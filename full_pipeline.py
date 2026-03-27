"""
Generic Object Decoding — Cross-Subject Generalizability Analysis
=================================================================
Research question:
    Can a model trained on one subject's brain responses decode
    what another subject is seeing — and WHERE does variability lie?

Stages:
    1. Within-subject baseline (all 5 subjects)
    2. Cross-subject decoding matrix
    3. ROI-level analysis (V1/V2/V3/V4/LOC/VC)
    4. Category-level variability analysis
"""

import h5py
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import pandas as pd
import time

warnings.filterwarnings('ignore')

# ── Config — UPDATE THESE PATHS for your local setup ────────────────────────────────────────────────────────────────────
DATA_DIR = Path('/home/saud/Desktop/bmi/Generic Object Decoding (fMRI) 7387130')
CAT_LABELS     = Path('data/ImageNetTest_synset.xlsx')
RESULTS_DIR    = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)  # creates results/ folder automatically
SUBJECTS = [f'Subject{i}.h5' for i in range(1, 6)]
ROI_NAMES = ['ROI_V1', 'ROI_V2', 'ROI_V3', 'ROI_V4', 'ROI_LOC', 'ROI_VC']

ROI_COLORS = {
    'ROI_V1' : '#4A90D9',
    'ROI_V2' : '#5BB85D',
    'ROI_V3' : '#F0A500',
    'ROI_V4' : '#E05252',
    'ROI_LOC': '#9B59B6',
    'ROI_VC' : '#1ABC9C',
}

plt.rcParams.update({
    'figure.facecolor' : '#0F1117',
    'axes.facecolor'   : '#1A1D27',
    'axes.edgecolor'   : '#2E3347',
    'axes.labelcolor'  : '#C8CCDA',
    'xtick.color'      : '#8A8FA8',
    'ytick.color'      : '#8A8FA8',
    'text.color'       : '#E0E3EF',
    'grid.color'       : '#2E3347',
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.5,
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})


# ── Helper Functions ──────────────────────────────────────────────────────────
def load_subject(subject_file):
    """Load and parse a GOD subject .h5 file."""
    path = DATA_DIR / subject_file
    with h5py.File(path, 'r') as f:
        dataset     = f['dataset'][:]
        meta_keys   = [k.decode('utf-8') for k in f['metadata/key'][:]]
        meta_values = f['metadata/value'][:]

    meta = {k: meta_values[i] for i, k in enumerate(meta_keys)}

    def find_col(key):
        cols = np.where(~np.isnan(meta[key]))[0]
        return cols[0] if len(cols) > 0 else None

    datatype_col = find_col('DataType')
    category_col = find_col('category_index')
    voxel_mask   = ~np.isnan(meta['VoxelData'])

    datatype = dataset[:, datatype_col].astype(int)
    category = dataset[:, category_col].astype(int)
    X_all    = dataset[:, voxel_mask]

    roi_masks = {}
    for roi in ROI_NAMES:
        if roi in meta:
            roi_full     = ~np.isnan(meta[roi])
            roi_masks[roi] = roi_full[voxel_mask]
        else:
            roi_masks[roi] = np.zeros(voxel_mask.sum(), dtype=bool)

    train_mask   = datatype == 1
    test_mask    = datatype == 2
    X_train_full = X_all[train_mask]
    y_train_full = category[train_mask]
    X_test       = X_all[test_mask]
    y_test       = category[test_mask]

    test_cats    = np.unique(y_test)
    train_filter = np.isin(y_train_full, test_cats)
    X_train      = X_train_full[train_filter]
    y_train      = y_train_full[train_filter]

    return {
        'X_train'  : X_train,
        'y_train'  : y_train,
        'X_test'   : X_test,
        'y_test'   : y_test,
        'roi_masks': roi_masks,
        'test_cats': test_cats,
    }


def roi_means(X, roi_masks):
    """Compress brain data to mean activation per ROI — common across subjects."""
    features = []
    for roi in ROI_NAMES:
        mask = roi_masks[roi]
        if mask.sum() > 0:
            features.append(X[:, mask].mean(axis=1, keepdims=True))
        else:
            features.append(np.zeros((X.shape[0], 1)))
    return np.hstack(features)  # (n_samples, 6)


def align_subjects(subjects_data):
    """
    Convert each subject's data to ROI-mean feature space (6 features).
    This creates a common representation across subjects with different
    voxel counts, enabling cross-subject training and testing.
    """
    aligned = {}
    for subj, d in subjects_data.items():
        aligned[subj] = {
            'X_train'     : roi_means(d['X_train'], d['roi_masks']),
            'X_test'      : roi_means(d['X_test'],  d['roi_masks']),
            'y_train'     : d['y_train'],
            'y_test'      : d['y_test'],
            'roi_masks'   : d['roi_masks'],
            'test_cats'   : d['test_cats'],
            'X_train_full': d['X_train'],
            'X_test_full' : d['X_test'],
        }
    return aligned


def average_trials(X, y):
    """Average repeated trials of same stimulus to reduce noise."""
    unique_labels = np.unique(y)
    X_avg = np.array([X[y == lbl].mean(axis=0) for lbl in unique_labels])
    return X_avg, unique_labels


def build_pipeline(n_components=100):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=n_components, random_state=42)),
        ('svm',    SVC(kernel='linear', C=1.0, probability=True, random_state=42)),
    ])


def evaluate(pipe, X_test, y_test):
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    top1 = accuracy_score(y_test, y_pred)
    top5 = top_k_accuracy_score(y_test, y_proba, k=5, labels=pipe.classes_)
    return top1, top5


def decode_with_roi(X_train, y_train, X_test, y_test, roi_mask, n_components=50):
    """Train and evaluate using only voxels in a given ROI."""
    if roi_mask.sum() < 10:
        return None, None
    n_comp = min(n_components, roi_mask.sum() - 1, len(X_train) - 1)
    pipe   = build_pipeline(n_components=n_comp)
    pipe.fit(X_train[:, roi_mask], y_train)
    return evaluate(pipe, X_test[:, roi_mask], y_test)


# ── Stage 1: Load all subjects ────────────────────────────────────────────────
print("=" * 60)
print("Loading all subjects...")
print("=" * 60)

subjects_data = {}
for subj in SUBJECTS:
    path = DATA_DIR / subj
    if not path.exists():
        print(f"  ✗ {subj} not found — skipping")
        continue
    print(f"  Loading {subj}...", end=' ')
    subjects_data[subj] = load_subject(subj)
    d = subjects_data[subj]
    print(f"train={d['X_train'].shape}, test={d['X_test'].shape}")

available = list(subjects_data.keys())
n_subj    = len(available)
print(f"\n{n_subj} subjects loaded.\n")

if n_subj == 0:
    print("No subjects found. Check DATA_DIR path.")
    exit()

# ── Stage 2: Within-subject baseline ─────────────────────────────────────────
print("=" * 60)
print("Stage 1: Within-subject decoding (with trial averaging)")
print("=" * 60)

within_top1 = {}
within_top5 = {}

for subj in available:
    d = subjects_data[subj]
    X_te_avg, y_te_avg = average_trials(d['X_test'], d['y_test'])
    n_comp = min(100, len(d['X_train']) - 1)
    pipe   = build_pipeline(n_components=n_comp)
    pipe.fit(d['X_train'], d['y_train'])
    top1, top5 = evaluate(pipe, X_te_avg, y_te_avg)
    within_top1[subj] = top1
    within_top5[subj] = top5
    print(f"  {subj}: Top-1={top1*100:.1f}%  Top-5={top5*100:.1f}%")

print(f"\n  Mean Top-1: {np.mean(list(within_top1.values()))*100:.1f}%")
print(f"  Mean Top-5: {np.mean(list(within_top5.values()))*100:.1f}%")
print(f"  Chance:     Top-1=2.0%  Top-5=10.0%")

# ── Stage 3: Cross-subject decoding ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 2: Cross-subject decoding matrix (ROI-mean feature space)")
print("=" * 60)

aligned      = align_subjects(subjects_data)
cross_top1   = np.full((n_subj, n_subj), np.nan)
cross_top5   = np.full((n_subj, n_subj), np.nan)

for i, train_subj in enumerate(available):
    for j, test_subj in enumerate(available):
        d_train = aligned[train_subj]
        d_test  = aligned[test_subj]

        shared   = np.intersect1d(d_train['test_cats'], d_test['test_cats'])
        if len(shared) == 0:
            continue

        tr_mask  = np.isin(d_train['y_train'], shared)
        te_mask  = np.isin(d_test['y_test'],   shared)

        X_tr = d_train['X_train'][tr_mask]
        y_tr = d_train['y_train'][tr_mask]
        X_te = d_test['X_test'][te_mask]
        y_te = d_test['y_test'][te_mask]

        if len(X_tr) < 10 or len(X_te) < 10:
            continue

        X_te_avg, y_te_avg = average_trials(X_te, y_te)

        n_comp = min(6, len(X_tr) - 1)
        pipe   = build_pipeline(n_components=n_comp)
        pipe.fit(X_tr, y_tr)
        top1, top5 = evaluate(pipe, X_te_avg, y_te_avg)

        cross_top1[i, j] = top1
        cross_top5[i, j] = top5

        tag = "(within)" if i == j else "(cross) "
        print(f"  Train={train_subj[:8]} → Test={test_subj[:8]} {tag}"
              f"  Top-1={top1*100:.1f}%  Top-5={top5*100:.1f}%")

# ── Stage 4: ROI-level analysis ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 3: ROI-level decoding (within-subject, Subject 1)")
print("=" * 60)

roi_results = {}
subj1       = available[0]
d1          = subjects_data[subj1]
X_te_avg, y_te_avg = average_trials(d1['X_test'], d1['y_test'])

for roi in ROI_NAMES:
    mask       = d1['roi_masks'][roi]
    top1, top5 = decode_with_roi(
        d1['X_train'], d1['y_train'],
        X_te_avg, y_te_avg, mask
    )
    roi_results[roi] = {
        'top1'    : top1,
        'top5'    : top5,
        'n_voxels': mask.sum(),
    }
    if top1 is not None:
        print(f"  {roi:10s} ({mask.sum():4d} voxels): "
              f"Top-1={top1*100:.1f}%  Top-5={top5*100:.1f}%")
    else:
        print(f"  {roi:10s}: insufficient voxels")

# ── Stage 5: Category-level variability ──────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 4: Per-category cross-subject variability")
print("=" * 60)

cat_accuracy = {}

for subj in available:
    d = subjects_data[subj]
    X_te_avg, y_te_avg = average_trials(d['X_test'], d['y_test'])
    n_comp = min(100, len(d['X_train']) - 1)
    pipe   = build_pipeline(n_components=n_comp)
    pipe.fit(d['X_train'], d['y_train'])
    y_pred = pipe.predict(X_te_avg)

    for cat, pred in zip(y_te_avg, y_pred):
        if cat not in cat_accuracy:
            cat_accuracy[cat] = []
        cat_accuracy[cat].append(1 if pred == cat else 0)

cat_mean = {c: np.mean(v) for c, v in cat_accuracy.items()}
cat_var  = {c: np.var(v)  for c, v in cat_accuracy.items()}

sorted_cats    = sorted(cat_mean, key=cat_mean.get, reverse=True)
sorted_by_var  = sorted(cat_var,  key=cat_var.get,  reverse=True)

print("\n  Top 10 most consistently decoded categories:")
for cat in sorted_cats[:10]:
    print(f"    Category {cat:3d}: "
          f"mean={cat_mean[cat]*100:.0f}%  var={cat_var[cat]:.3f}")

print("\n  Top 10 most variable categories (high variance across subjects):")
for cat in sorted_by_var[:10]:
    print(f"    Category {cat:3d}: "
          f"mean={cat_mean[cat]*100:.0f}%  var={cat_var[cat]:.3f}")

#Catagory Look up
cat_df = pd.read_excel(CAT_LABELS)
print("\n  Best decoded categories with names:")
for cat in sorted_cats[:10]:
    name = cat_df.iloc[cat-1]['category'] if cat <= len(cat_df) else 'unknown'
    print(f"    Category {cat:3d} ({name:20s}): "
          f"mean={cat_mean[cat]*100:.0f}%  var={cat_var[cat]:.3f}")

# ── Plotting ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Generating plots...")
print("=" * 60)

subj_labels = [f'S{i+1}' for i in range(n_subj)]

# Plot 1: Cross-subject accuracy heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0F1117')
fig.suptitle('Within vs Cross-Subject Decoding Accuracy',
             fontsize=14, fontweight='bold', color='#E0E3EF')

for ax, matrix, title in zip(
        axes,
        [cross_top1, cross_top5],
        ['Top-1 Accuracy', 'Top-5 Accuracy']):
    im = ax.imshow(matrix * 100, cmap='RdYlGn',
                   vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(n_subj))
    ax.set_yticks(range(n_subj))
    ax.set_xticklabels([f'Test {s}' for s in subj_labels], fontsize=8)
    ax.set_yticklabels([f'Train {s}' for s in subj_labels], fontsize=8)
    ax.set_title(title, fontsize=11, color='#C8CCDA', pad=8)
    for i in range(n_subj):
        for j in range(n_subj):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i,j]*100:.1f}%',
                        ha='center', va='center',
                        fontsize=9, color='black', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Accuracy %')

plt.tight_layout()
plt.savefig(RESULTS_DIR /'plot_cross_subject_matrix.png',
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("  ✓ plot_cross_subject_matrix.png")

# Plot 2: ROI-level accuracy
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0F1117')

valid_rois = [r for r in ROI_NAMES if roi_results[r]['top1'] is not None]
top1_vals  = [roi_results[r]['top1'] * 100 for r in valid_rois]
top5_vals  = [roi_results[r]['top5'] * 100 for r in valid_rois]
n_vox_vals = [roi_results[r]['n_voxels']   for r in valid_rois]
colors     = [ROI_COLORS[r] for r in valid_rois]
x          = np.arange(len(valid_rois))
width      = 0.35

ax.bar(x - width/2, top1_vals, width, label='Top-1',
       color=colors, alpha=0.85, linewidth=0)
ax.bar(x + width/2, top5_vals, width, label='Top-5',
       color=colors, alpha=0.45, linewidth=0)

for i, (nv, xpos) in enumerate(zip(n_vox_vals, x)):
    ax.text(xpos, max(top1_vals[i], top5_vals[i]) + 0.5,
            f'{nv}v', fontsize=7, color='#8A8FA8', ha='center')

ax.axhline(2,  color='#E05252', linewidth=1,
           linestyle='--', alpha=0.7, label='Chance Top-1 (2%)')
ax.axhline(10, color='#F0A500', linewidth=1,
           linestyle='--', alpha=0.7, label='Chance Top-5 (10%)')
ax.set_xticks(x)
ax.set_xticklabels([r.replace('ROI_', '') for r in valid_rois], fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=10, color='#8A8FA8')
ax.set_title(f'Decoding Accuracy by Brain ROI — {subj1}',
             fontsize=13, fontweight='bold', color='#E0E3EF', pad=12)
ax.legend(fontsize=9, framealpha=0.3, facecolor='#1A1D27',
          edgecolor='#2E3347', labelcolor='#C8CCDA')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR /'/plot_roi_accuracy.png',
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("  ✓ plot_roi_accuracy.png")

# Plot 3: Category variability
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor('#0F1117')
fig.suptitle('Per-Category Decoding: Mean Accuracy vs Cross-Subject Variability',
             fontsize=13, fontweight='bold', color='#E0E3EF')

cats  = sorted(cat_mean.keys())
means = [cat_mean[c] * 100 for c in cats]
varis = [cat_var[c]        for c in cats]

ax          = axes[0]
sorted_idx  = np.argsort(means)[::-1]
sorted_means= [means[i] for i in sorted_idx]
bar_colors  = ['#5BB85D' if m > 10 else
               '#F0A500' if m > 2  else
               '#E05252' for m in sorted_means]
ax.bar(range(len(sorted_means)), sorted_means,
       color=bar_colors, alpha=0.8, linewidth=0)
ax.axhline(2, color='#E05252', linewidth=1,
           linestyle='--', alpha=0.7, label='Chance (2%)')
ax.set_xlabel('Category (sorted by accuracy)', fontsize=9, color='#8A8FA8')
ax.set_ylabel('Mean Accuracy % (across subjects)', fontsize=9, color='#8A8FA8')
ax.set_title('Per-Category Mean Decoding Accuracy', fontsize=11, color='#C8CCDA')
green_p = mpatches.Patch(color='#5BB85D', alpha=0.8, label='Above chance (>10%)')
amber_p = mpatches.Patch(color='#F0A500', alpha=0.8, label='Near chance (2-10%)')
red_p   = mpatches.Patch(color='#E05252', alpha=0.8, label='Below chance (<2%)')
ax.legend(handles=[green_p, amber_p, red_p], fontsize=8,
          framealpha=0.3, facecolor='#1A1D27',
          edgecolor='#2E3347', labelcolor='#C8CCDA')
ax.grid(True, axis='y', alpha=0.3)

ax = axes[1]
sc = ax.scatter(means, varis, c=means, cmap='RdYlGn',
                s=60, alpha=0.8, vmin=0, vmax=100)
ax.set_xlabel('Mean Accuracy % across subjects', fontsize=9, color='#8A8FA8')
ax.set_ylabel('Variance across subjects', fontsize=9, color='#8A8FA8')
ax.set_title('Accuracy vs Variability per Category', fontsize=11, color='#C8CCDA')
plt.colorbar(sc, ax=ax, label='Mean accuracy %')
for c in sorted_cats[:5]:
    ax.annotate(f'Cat {c}',
                (cat_mean[c]*100, cat_var[c]),
                fontsize=7, color='#C8CCDA',
                xytext=(5, 5), textcoords='offset points')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR /'plot_category_variability.png',
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("  ✓ plot_category_variability.png")

# Plot 4: Within vs cross bar comparison
if n_subj > 1:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0F1117')

    within_vals = [within_top5.get(s, 0) * 100 for s in available]
    cross_vals  = []
    for i in range(n_subj):
        off_diag = [cross_top5[j, i] * 100
                    for j in range(n_subj)
                    if j != i and not np.isnan(cross_top5[j, i])]
        cross_vals.append(np.mean(off_diag) if off_diag else 0)

    x     = np.arange(n_subj)
    width = 0.35
    ax.bar(x - width/2, within_vals, width,
           label='Within-subject', color='#4A90D9', alpha=0.85, linewidth=0)
    ax.bar(x + width/2, cross_vals, width,
           label='Cross-subject (mean)', color='#E05252', alpha=0.85, linewidth=0)
    ax.axhline(10, color='#F0A500', linewidth=1,
               linestyle='--', alpha=0.7, label='Chance Top-5 (10%)')
    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels, fontsize=11)
    ax.set_ylabel('Top-5 Accuracy (%)', fontsize=10, color='#8A8FA8')
    ax.set_title('Within vs Cross-Subject Decoding — Top-5 Accuracy',
                 fontsize=13, fontweight='bold', color='#E0E3EF', pad=12)
    ax.legend(fontsize=9, framealpha=0.3, facecolor='#1A1D27',
              edgecolor='#2E3347', labelcolor='#C8CCDA')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR /'plot_within_vs_cross.png',
                dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print("  ✓ plot_within_vs_cross.png")

cat_df = pd.read_excel(CAT_LABELS)



print("\n" + "=" * 60)
print("All done.")
print("=" * 60)