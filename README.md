# Electrode Coverage Pipeline

Screen intracranial electrodes (iEEG/SEEG/ECoG) against any MNI-space brain atlas, using public data from OpenNeuro.

**What it does**: You provide a brain atlas (NIfTI file) for any region of interest. The pipeline automatically finds all publicly available iEEG datasets on OpenNeuro, downloads their electrode coordinates, and tells you which electrodes fall inside (or near) your region. It also extracts clinical annotations like seizure onset zone (SOZ) labels.

---

## Quick Start

### Option A: pip install (recommended)

```bash
git clone <repo-url>
cd unified_pipeline
pip install .
```

After installation, the `electrode-coverage` command is available globally:

```bash
electrode-coverage --help
```

### Option B: conda

```bash
git clone <repo-url>
cd unified_pipeline
conda env create -f environment.yml
conda activate electrode-coverage
```

### Option C: manual

```bash
git clone <repo-url>
cd unified_pipeline
pip install -r requirements.txt
python -m electrode_coverage --help
```

---

## Step-by-Step Usage Guide

### Step 1: Prepare Your Atlas

You need an MNI-space NIfTI file (`.nii` or `.nii.gz`). This can be:
- A binary mask (0/1 values) of your region of interest
- A labeled atlas where each region has a numeric label
- A probabilistic map (continuous values)

Common sources:
- [Harvard-Oxford atlas](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases) (via FSL or nilearn)
- [JHU white matter atlas](https://neurovault.org/collections/264/)
- Any custom mask you've created in MNI space

**Verify your atlas** before running the full pipeline:

```bash
electrode-coverage --roi-mask your_atlas.nii.gz --atlas-info
```

This prints shape, voxel size, MNI bounding box, labels, and a PASS/WARN/FAIL status.

### Step 2: Choose Your Data Source

**Scan ALL OpenNeuro iEEG datasets** (recommended first run):

```bash
electrode-coverage --roi-mask your_atlas.nii.gz --all-openneuro -o results.csv
```

**Scan specific datasets**:

```bash
electrode-coverage --roi-mask your_atlas.nii.gz --openneuro ds003708 ds004473 -o results.csv
```

**Use a local BIDS dataset**:

```bash
electrode-coverage --roi-mask your_atlas.nii.gz /path/to/bids_dataset -o results.csv
```

### Step 3: Use a Labeled Atlas (pick specific regions)

If your atlas has numeric labels (e.g., label 17 = hippocampus):

```bash
electrode-coverage --roi-atlas your_labeled_atlas.nii.gz \
  --roi-labels 17 18 \
  --roi-name "Hippocampus" \
  --all-openneuro \
  -o results.csv
```

### Step 4: Adjust Proximity Threshold

By default, only electrodes **exactly inside** the ROI are flagged. To also include electrodes within a distance:

```bash
electrode-coverage --roi-mask your_atlas.nii.gz \
  --all-openneuro \
  --proximity-mm 5 \
  -o results.csv
```

This flags electrodes within 5mm of the ROI boundary. The output CSV includes a `distance_mm` column showing the exact distance for every electrode.

### Step 5: Mask Dilation

You can also expand the mask itself before screening:

```bash
electrode-coverage --roi-mask your_atlas.nii.gz \
  --dilate-mm 3 \
  --all-openneuro \
  -o results.csv
```

**Dilation vs proximity**: Dilation permanently grows the mask. Proximity keeps the original mask but reports nearby electrodes. You can use both.

### Step 6: Filter by Channel Annotations

Only process datasets that have seizure onset zone labels:

```bash
electrode-coverage --roi-mask your_atlas.nii.gz \
  --all-openneuro \
  --require-soz \
  -o results.csv
```

Filter options:
- `--require-channels` — only datasets with `channels.tsv` files
- `--require-annotations` — only datasets with clinical annotation text
- `--require-soz` — only datasets where channels are labeled as SOZ

### Step 7: Generate 3D Visualization

```bash
electrode-coverage --roi-mask your_atlas.nii.gz \
  --all-openneuro \
  --proximity-mm 5 \
  -o results.csv \
  --visualize output.html
```

Open `output.html` in a browser. It shows:
- Semi-transparent brain surface
- Your ROI as a colored 3D mesh
- Electrodes color-coded by dataset
- SOZ electrodes as red diamonds
- Hover over any electrode for details

### Step 8: View the Search Funnel

See how many datasets were filtered at each stage:

```bash
electrode-coverage --roi-mask your_atlas.nii.gz \
  --all-openneuro \
  --require-soz \
  --search-report \
  -o results.csv
```

---

## Understanding the Output

### CSV Columns

| Column | Description |
|--------|-------------|
| `dataset` | OpenNeuro dataset ID or local name |
| `subject` | Subject identifier |
| `session` | Session identifier |
| `channel_name` | Electrode/channel name |
| `x`, `y`, `z` | Original coordinates |
| `mni_x`, `mni_y`, `mni_z` | MNI-transformed coordinates |
| `coordinate_system` | Original coordinate system string |
| `coord_space_family` | Classified space (mni152, talairach, acpc, surface, native, unknown) |
| `coord_status` | Transform applied (mni_native, mni_from_talairach, acpc_approximate, etc.) |
| `suspicious_units` | True if coordinates may be in wrong units |
| `distance_mm` | Distance in mm from electrode to nearest ROI boundary (0 = inside) |
| `in_roi` | True if electrode is inside ROI (or within proximity threshold) |
| `channel_type` | ECOG, SEEG, EKG, etc. (from channels.tsv) |
| `channel_status` | good/bad |
| `is_soz` | True if annotated as seizure onset zone |
| `is_irritative_zone` | True if annotated as irritative zone |
| `is_resected` | True if annotated as resected |
| `is_bad` | True if channel status is bad |
| `clinical_annotation` | Raw annotation text |

### Coordinate Spaces Explained

- **MNI152** (mni_native): Standard MNI space. Screened directly.
- **ACPC** (acpc_approximate): AC-PC aligned. Treated as approximate MNI with a warning. Results should be interpreted with caution.
- **Talairach** (mni_from_talairach): Transformed to MNI via the Lancaster et al. 2007 published affine matrix, applied using `nibabel.affines.apply_affine()`.
- **Surface** (surface_space): FreeSurfer surface coordinates (vertex indices). Cannot be screened against volumetric masks. Skipped.
- **Native** (native_space): Scanner-specific coordinates. Require subject-specific registration not available in public metadata. Skipped.

---

## GitHub API Rate Limits

The pipeline downloads electrode files from GitHub (OpenNeuro mirror). Without authentication: **60 requests/hour**. With a free GitHub token: **5,000 requests/hour**.

To avoid rate limits:

```bash
export GITHUB_TOKEN=ghp_your_token_here
electrode-coverage --all-openneuro ...
```

Get a token at: https://github.com/settings/tokens (no special permissions needed).

---

## Cache Management

Downloaded files are cached at `~/.electrode_coverage/cache/`. Cache auto-refreshes every 7 days.

```bash
# Force re-download everything
electrode-coverage --refresh-cache --all-openneuro ...

# Set cache TTL to 1 day
electrode-coverage --cache-ttl 1 --all-openneuro ...

# Always fresh (no cache)
electrode-coverage --cache-ttl 0 --all-openneuro ...
```

---

## Built-in Presets

For fornix-specific analysis:

```bash
# Fornix (FSL probabilistic template)
electrode-coverage --preset fornix --all-openneuro -o results.csv --visualize fornix.html

# Fornix (JHU white matter labels)
electrode-coverage --preset fornix-jhu --all-openneuro -o results.csv
```

---

## Python API

```python
from electrode_coverage import ElectrodeCoveragePipeline, PipelineConfig, ROIConfig

config = PipelineConfig(
    roi=ROIConfig(name="Hippocampus", mask_path="hippo_mask.nii.gz"),
    proximity_mm=5.0,
)
pipe = ElectrodeCoveragePipeline(config)
pipe.add_all_openneuro()
result = pipe.run()

result.print_summary()
result.save_csv("results.csv")
result.visualize("output.html")

# Access the data
print(result.in_roi)          # DataFrame of electrodes in ROI
print(result.dataset_summary) # Per-dataset stats
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| nibabel | >=4.0 | NIfTI file I/O, affine transforms |
| nilearn | >=0.10 | Brain surface meshes, atlas fetching |
| numpy | >=1.24 | Array operations |
| pandas | >=1.5 | DataFrame handling, TSV reading |
| scipy | >=1.10 | Mask dilation, distance transforms |
| plotly | >=5.0 | 3D interactive visualization |
| scikit-image | >=0.20 | Marching cubes for mesh generation |
