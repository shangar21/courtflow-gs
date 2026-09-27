# Deviations from the task spec (and from QuickCapture)

## Stage A (frame 0)

| Spec | Implementation | Reason |
| --- | --- | --- |
| Step 2: associate instances by projecting mask-centroid rays + epipolar distance + colour histogram. | Instances are 3D connected components of a court-wide **occupancy carving** of every camera's union person mask; each component is projected back to pick (Hungarian IoU) the SAM2 detector mask in each view, or to re-prompt SAM2 where the detector missed. | With 12 views 30 degrees apart and ~20x90 px players, centroid/epipolar chaining fragmented 68 masks into 60 IDs. Identity from 3D is consistent in every view by construction; the synthetic test checks it. Colour histograms are still used, but only to reject re-prompted masks that segment a static occluder (basket stanchion) instead of the person. |
| Step 2: SAM2 auto masks or detector boxes. | Full-resolution Faster R-CNN R50-FPN-v2 boxes (pluggable `BoxPrompter`) refined by SAM2.1-tiny, filtered to people whose feet hit the court floor. | The 320-px mobilenet detector missed small players. |
| Step 3: full-frame MASt3R on 24 pairs is the primary player geometry. | Full-frame 512-px MASt3R (24 pairs) is used for the **background** and the floor. Players use **per-instance crop matching**: a padded square crop around the instance in both images is resized to 512 (~3-5x magnification), matched with reciprocal NN seeded on the instance mask, mapped back, DLT-triangulated with the known P. | Players are ~6x25 px at 512 full-frame; matching there produced the "star" of rays. |
| Step 3: instance label by majority vote. | Foreground points are **carved** by the instance hull (must lie inside it) and must agree in colour + instance label in >= 1 extra view. | Majority voting admitted floor points lying on rays through the player (the radial "star" failure of the previous attempt). |
| Step 4: strict hull (inside the mask in every camera where in frustum). | Strict, plus two principled exemptions: a camera is skipped where a **nearer instance** covers the pixel (spec), or where **static geometry** (multi-view-consistent background DLT points, >= 0.15 m above the floor, splatted 4 px) is nearer than the voxel by 0.3 m. Voxels below the floor plane are clipped. | The stanchion/padding partially hides several people in some views; a strict hull removed heads and feet. `hull_allowed_misses` (default 0) exists; 1 inflates hulls ~2.5x. |
| Scene units "meters". | The supplied calibration is not metric (~0.078 units/m, estimated from standing heights of detected people with the floor plane). All metric thresholds in `config.py` are in meters and converted with `units_per_meter`. | Keeps the spec's metric defaults meaningful. |
| Step 7 alpha BCE against the instance mask everywhere. | Pixels of nearer instances and pixels where the instance hull projects onto background (the view that could not see that part: static occlusion) are ignored by the BCE. | Otherwise the alpha loss erases body parts hidden behind the stanchion. |
| DropGaussian p = 0.1. | Faithful to the paper (arXiv 2504.00773): drop with rate r and scale survivors' opacity by 1/(1-r); r grows linearly r_t = gamma * t / T. `drop_gaussian_rate` is gamma (default 0.1 per spec; the paper uses 0.2). Disabled at eval/export. | |
| Crop rendering. | Instance training renders tile-aligned crops of equal size per instance (K principal point shifted); pixel-identical to the full render. | Players cover <1% of a frame. |

Held-out images: the 24 non-training cameras are read only by `eval/heldout.py`, after models are frozen.
The target images are the supplied photometrically-normalized frames (the same source the
17.50 dB global baseline was evaluated on).

## Stage B vs QuickCapture (`third_party/QuickCapture`, read before implementing; nothing imported)

| Area | QuickCapture | This implementation | Reason |
| --- | --- | --- | --- |
| Weights | Inverse power `1/(d^alpha + eps)`, alpha = 7.2 (learnable), over *all* controls; covariance weighted by W^2 | Normalized Gaussian RBF `exp(-d^2 / 2 sigma_j^2)` over the K = 8 nearest controls of the same instance, sigma_j = mean distance to the control's 3 nearest controls; linear weights in the covariance | Task spec; compact support (O(NK) memory), no cross-instance coupling. |
| Rigid fit | `T = Vh @ U` (V^T U) without reflection fix; the fused path then overwrites T with identity | `R = V U^T` with the det fix, fused in one CUDA kernel (Jacobi eigen of M^T M, Gram-Schmidt U, signed sigma_3) | Correct proper rotation. |
| Orientation | Rotates only the quaternion's vector part by T | Proper product `R * q` | |
| SH | Not rotated | Band-1 SH rotated: `c' = A R A^T c`, A maps gsplat's (-y, z, -x) basis order (verified against gsplat's SH evaluation) | View-dependent colour follows the body. |
| Backward | Autograd through a batched SVD library | Analytic polar-factor gradient `dL/dM = U Y V^T`, `Y_ij = skew(U^T G V)_ij / (sigma_i + sigma_j)` (no `1/(sigma_j^2 - sigma_i^2)` blow-up for near-planar neighbourhoods); `eps` is relative to sigma_1; warp-segmented reduction of equal control indices before atomicAdd | Stable and fast; 460-1500x faster than the PyTorch reference at 100k-500k Gaussians. |
| Kernel scope | Fused CUDA builds the covariance only | Gather, centroids, covariance, SVD, outputs and backward fused | |
| CP rotation blend | None | Optional per-Gaussian twist `nlerp(I, normalize(sum w q_j), beta)` applied after R_i (`mls_blend_cp_rotation`, default off) | Spec extension for local twist. |
| Optimized state | Gaussian parameters trainable in the live loop | Only control translations/quaternions; canonical Gaussians and background frozen | Online contract; canonical export immutable. |
| Regularizer scale | — | ARAP residual divided by rest edge length^2, temporal acceleration by mean control spacing^2 (unit-free weights) | Scene units are ~0.078/m, raw residuals would make weights meaningless. |
| Nsight Compute | — | Not available without root (`ERR_NVGPUCTRPERM`); CUDA-event timings reported instead | |

## Stage B vs the task spec

| Spec | Implementation | Reason / evidence |
| --- | --- | --- |
| Optimize t_j, q_j per control. | t_j = T_person + delta_j: a per-person shared translation (lr 2 cm) plus per-control offsets (lr 3 mm), initialized each frame from the triangulated mask centroids (least-squares ray intersection over the 12 views) and softly tied to them (Huber, 0.3 m scale). | Per-control Adam alone lagged fast players and random-walked weakly observed controls (IoU 0.60 at frame 10 vs 0.71 with the split). The anchor re-acquires lost players. |
| ARAP with R_a from q_a. | Default: R_a from each control's local MLS fit (`tracking_optimize_rotations=False`); q still optimized when the CP-rotation blend is on. | With R from q, ARAP resisted rotation until q caught up (worse IoU). |
| Temporal weight not specified. | 0.01 on acceleration / control-spacing^2. | 0.1 over-damped basketball motion (frame-10 IoU 0.64 -> 0.71 when reduced). |
| 100 iterations. | <= 50 with window-mean early stop (>= 15). | 50 iterations reach the 12-camera / 300-iteration IoU within 0.01 on frame 1. |
| Full-resolution losses. | Tracking renders/losses at 0.5 image scale in a crop around all persons; evaluation is full resolution. | Speed. |
| Alpha BCE vs mask. | Per-instance BCE on one-hot identity channels rendered with RGB (14 extra channels), weight 1.0. | Identity-aware supervision when players overlap; union mode (`tracking_mask_mode=union`) is ~20% faster with equal frame-1 IoU. |
| SAM2 video predictor online. | Run per camera over the clip (causal propagation: frame f only sees frames <= f), prompted with the Stage A frame-0 labels; overlaps resolved by highest logit. | Equivalent to online use; cost logged as mask_propagation_s. |
| New-geometry spawning, new-instance Stage A, appearance refinement. | Not implemented. | Time; see report. |
| Novel-view rendering. | Near plane at 0.8x the nearest triangulated seed depth for videos; orbit interpolates between physical ring poses (slerp). | Fitted-circle orbits leave the constrained region and show near-camera floaters. |
