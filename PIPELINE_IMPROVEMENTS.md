# Pipeline Quality Improvement Plan

## Problem

Current pipeline produces suboptimal Gaussian Splats for real estate scanning. Key issues: dead training config, no scene normalization, no frame quality filtering, wasted LiDAR data, no quality gate.

---

## Phase 1 — Immediate Wins

### 1.1 Wire TrainingConfig to Nerfstudio CLI
- **File**: `pipeline/train.py` — `to_nerfstudio_args()` (line 39-44)
- **Problem**: `TrainingConfig` defines 16 params but only `max_iterations` is passed. All other params are dead code.
- **Fix**: Map all fields to nerfstudio CLI flags:
  - `--pipeline.model.densify-grad-thresh`
  - `--pipeline.model.cull-alpha-thresh`
  - `--pipeline.model.stop-split-at`
  - `--pipeline.model.refine-every`
  - `--pipeline.model.num-downscales 0`
  - `--pipeline.model.warmup-length`
  - `--optimizers.means.optimizer.lr`
  - `--pipeline.model.lambda-dssim`
- **Impact**: Very high | **Effort**: Low
- **Status**: [x] DONE

### 1.2 Auto-compute aabb_scale + Scene Centering
- **File**: `pipeline/convert_poses.py` — `create_transforms_json()` (line 134)
- **Problem**: `aabb_scale` hardcoded to 4. `compute_scene_bounds()` exists but result unused. Wrong aabb_scale = poor quality.
- **Fix**:
  - Compute scene bounds from all camera positions
  - Center poses by subtracting centroid
  - `aabb_scale = 2^ceil(log2(diagonal * 1.5))`, clamped [1, 128]
- **Impact**: High | **Effort**: Low
- **Status**: [x] DONE

### 1.3 Quality-Weighted Frame Selection
- **File**: `pipeline/extract_frames.py` — `downsample_frames()` (line 319)
- **Problem**: Selection purely by path distance. No quality awareness.
- **Fix**:
  - Compute per-frame quality: Laplacian variance (sharpness) + mean brightness
  - When multiple frames in same path segment, prefer highest quality
  - Note: iOS already does blur rejection (threshold 100), this catches remaining suboptimal frames
- **Impact**: Medium-high | **Effort**: Low
- **Status**: [x] DONE

### 1.4 Outlier Pose Detection
- **File**: `pipeline/extract_frames.py` or new `utils/pose_quality.py`
- **Problem**: Only zero-position and basic velocity checks. Tracking glitches near glass/mirrors pass through.
- **Fix**:
  - Sliding window smoothness: flag poses >2σ from local fit
  - Angular consistency: flag >15° rotation jumps between frames
  - Remove flagged poses before downsampling
- **Impact**: High | **Effort**: Medium
- **Status**: [x] DONE

### 1.5 Post-Training Quality Metrics
- **File**: New `pipeline/quality_assessment.py`, modify `process.py`
- **Problem**: Cannot distinguish good results from garbage. Bad scans ship to users.
- **Fix**:
  - Hold out every 10th frame as test set
  - After training, compute PSNR/SSIM on rendered test views
  - Gate: PSNR < 20 = fail, 20-25 = warn, >25 = pass
  - Include metrics in package metadata
- **Impact**: Very high | **Effort**: Medium
- **Status**: [x] DONE

---

## Phase 2 — Major Quality Improvements

### 2.1 Appearance Embedding for Lighting
- **File**: `pipeline/train.py`
- Add `--pipeline.model.use-appearance-embedding True` when brightness std > 30
- Handles windows + mixed lighting (real estate #1 visual issue)
- **Impact**: High | **Effort**: Low
- **Status**: [ ] TODO

### 2.2 Depth Supervision from LiDAR Mesh
- **Files**: New `pipeline/depth_supervision.py`, modify `convert_poses.py`, `train.py`
- Render depth maps from `mesh.ply` at each camera pose via trimesh ray casting
- Add depth paths to transforms.json, enable `--pipeline.model.depth-loss-mult 0.1`
- Biggest ROI for Pro devices (all have LiDAR)
- **Impact**: Very high | **Effort**: Medium-high
- **Status**: [ ] TODO

### 2.3 Scene-Adaptive Training Profiles
- **File**: `pipeline/train.py`
- Auto-detect from scene diagonal: Small (<5m) 20k iter, Medium (5-15m) 30k, Large (15-40m) 40k, XL (40m+) 50k
- Each profile adjusts densification, LR, and other params
- **Impact**: Medium-high | **Effort**: Low-medium
- **Status**: [ ] TODO

### 2.4 View Coverage Optimization
- **File**: `pipeline/extract_frames.py`
- Spatial grid + greedy set-cover: prefer frames that cover under-represented areas
- Prevents over-sampling corridors, under-sampling rooms
- **Impact**: High | **Effort**: Medium
- **Status**: [ ] TODO

### 2.5 Gaussian Splat Cleanup / Floater Removal
- **File**: New `pipeline/postprocess_splat.py`
- Statistical outlier removal, cull transparent splats, clamp extreme scales, crop outside bbox
- Reuse `read_gaussian_ply()` from `compress.py:230`
- **Impact**: Medium-high | **Effort**: Medium
- **Status**: [ ] TODO

---

## Phase 3 — Delivery & Polish

### 3.1 Progressive Multi-Resolution Splat
- New `pipeline/progressive_splat.py`, modify `package.py`
- Sort by size×opacity, split into LOD0 (10%), LOD1 (30%), LOD2 (60%)
- Time-to-first-render: 30-60s → 3-5s
- **Impact**: High (UX) | **Effort**: Medium
- **Status**: [ ] TODO

### 3.2 Adaptive Frame Count
- `target = clamp(50 * scene_diagonal, 100, 500)` instead of hardcoded 250
- **Impact**: Medium | **Effort**: Low
- **Status**: [ ] TODO

### 3.3 Exposure/White Balance Normalization
- New `pipeline/image_preprocessing.py`
- Gentle CLAHE + gray-world WB blend toward mean
- **Impact**: Medium | **Effort**: Medium
- **Status**: [ ] TODO

### 3.4 SPZ Compression as Default
- `pipeline/compress.py` — flip default, improve fallback quantization
- 50MB PLY → 5-15MB SPZ
- **Impact**: High (delivery) | **Effort**: Low-medium
- **Status**: [ ] TODO

### 3.5 Smarter Thumbnail Selection
- `pipeline/package.py` — score by brightness+sharpness+orientation, pick best
- **Impact**: Low-medium | **Effort**: Low
- **Status**: [ ] TODO

---

## Phase 4 — Future / Research

### 4.1 Optional COLMAP Pose Refinement
- Bundle adjustment for drift reduction on large scenes
- Significant complexity + time cost, opt-in only
- **Status**: [ ] TODO

### 4.2 Evaluate Newer Training Methods
- Make method configurable (splatfacto vs splatfacto-big vs 2DGS)
- Benchmark before switching default
- **Status**: [ ] TODO

---

## Verification

For each phase:
1. Run on 3 scans: small room, medium apartment, large office
2. Compare before/after PSNR/SSIM (once 1.5 is implemented)
3. Visual inspection for floaters, color consistency, coverage
4. Check file sizes and load times
5. Regression: existing scans still process correctly
