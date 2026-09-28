# Archived early draft — do not use for final results

This document records the pre-PRO-6000, 300-frame development work and is retained only for
historical context. Its measurements, commands, and conclusions have been superseded.

Use [the final report](report/courtflow_gs_report.pdf) for the submission: it covers the final
700-frame, 12-input-view reconstruction, held-out 24-view evaluation, runtime, inference, and
limitations. The final 1080p result is 20.43 dB mean held-out PSNR on one RTX PRO 6000, completed
in approximately 119 minutes end to end.

---

# CourtFlow-GS

## Online multi-view dynamic Gaussian reconstruction with fused MLS tracking

### Executive summary

CourtFlow-GS reconstructs a frame-0 canonical Gaussian scene from a calibrated 12-camera ring,
then tracks a basketball play online using group-aware control points, fused rigid moving-least-
squares (MLS), mask/centroid constraints, focused refinement, and periodic full keyframes.  The
design is intended for high-quality offline reconstruction and fast baked-state novel-view
rendering; it is not a realtime tracker for unseen games.

The final diagnostic run completed 300 frames with a ball model and produced held-out and orbit
videos.  It is visually useful and substantially improves on the original person-only tracker,
but crowded late-game identity association remains the principal failure mode.

### Task and data

- 36 synchronized 4K cameras arranged as a ring.
- Twelve evenly spaced views (`0, 3, ..., 33`) are training cameras; the other 24 are held out.
- Camera poses are fixed known inputs. No camera or SfM optimization is used.
- The raw-distribution adapter accepts `cameras/` and `calibration/`, creates half-resolution
  pinhole training/evaluation inputs, and preserves the fixed-camera contract.

### Method

#### 1. Canonical frame-0 scene

Stage A builds a static canonical Gaussian scene.

1. Undistort calibrated images and estimate floor/scale.
2. Detect people, prompt SAM2, and associate masks in 3D through occupancy carving.
3. Build occlusion-aware visual hulls and obtain per-instance crop MASt3R correspondences.
4. Triangulate/filter/fuse points, initialize one Gaussian model per person plus background.
5. Train people with crop L1 + D-SSIM + alpha/depth regularization, then train background while
   compositing frozen people in front.

At frame 0, Stage A achieved 19.32 dB held-out PSNR, 0.651 SSIM, 0.243 LPIPS, 18.34 dB person
region PSNR, and 0.788 person IoU. This exceeds the prior global 3DGS baseline of 17.50 dB
held-out PSNR.

#### 2. Online control-point deformation

The live model contains background, person/referee/staff groups, and an optional ball group.
Controls are sampled throughout the background and per movable instance. Every Gaussian is bound
to its eight nearest same-group controls with fixed radial-basis weights. Each frame optimizes a
per-instance shared translation plus a local offset per control.

The fused CUDA MLS kernel maps offsets to rigid local transforms for Gaussian means, quaternions,
and degree-1 SH coefficients. An ARAP term uses the corresponding local MLS rotations. The
deformed model is baked as the next frame's state rather than repeatedly warping frame 0.

Dynamic-only deformation keeps the background controls fixed between keyframes and composites
dynamic splats over a cached background render. Masks are re-prompted from the current render
every five frames; their multi-view centroids provide robust bulk-motion anchors when identities
remain consistent. The ball follows a triangulated, gravity-aware trajectory anchor.

#### 3. Focused inter-keyframe refinement

Pure deformation loses quality after a keyframe. After each non-keyframe deformation,
CourtFlow-GS projects all dynamic Gaussians into each camera, tile-aligns a padded union crop, and
runs 100 Gaussian refinement iterations over those crops. Every tenth frame receives a 500-step,
full-image keyframe refinement with bounded densification, after which Gaussian-control bindings
are refreshed.

This crop refinement eliminates the severe early person-quality sawtooth, while keeping the
runtime close to the previous setup. It deliberately favors dynamic content; full-frame held-out
quality remains a tradeoff to improve in future work.

### Pseudocode

```text
canonical = build_stage_a(frame0_images, fixed_ring_calibration)
controls  = sample_controls(canonical.background, canonical.instances, ball)
state     = canonical

for frame in video_frames:
    images = decode_undistort_normalize_all_training_views(frame)

    if frame % 5 == 0:
        masks = SAM2.reprompt(images, project(state.dynamic_gaussians))
        anchors = triangulate_mask_centroids(masks)
    else:
        masks, anchors = None, no_anchor

    offsets = optimize_control_offsets(
        render_dynamic_over_cached_background(state, controls), images,
        mask_bce=masks,
        centroid_anchor=anchors,
        arap=local_mls_rotations(controls),
        temporal_acceleration=control_history)

    state = bake(fused_rigid_mls(state, controls, offsets))

    if frame % 10 != 0:
        crops = dynamic_content_crops(state, images)
        state = gaussian_refine(state, crops, iterations=100, densify=False)
    else:
        state = gaussian_refine(state, images, iterations=500, densify=bounded)
        controls = rebind_gaussians_to_controls(state, controls)
        cached_background = render_background(state)

    save_metrics_checkpoint_and_novel_view_video(state)
```

### Results

| Method / measurement | Full PSNR | Person PSNR | Person IoU | Time/frame |
|---|---:|---:|---:|---:|
| Stage A held-out frame 0 | 19.32 | 18.34 | 0.788 | — |
| Previous global 3DGS baseline, held-out frame 0 | 17.50 | — | — | — |
| Stage B v1, keyframe-held-out mean | 18.29 | 15.00 | 0.70 → 0.30 | 2.6 s |
| Stage B v2, old 300-frame keyframe-held-out mean | 18.87 | 19.62 | 0.71 → 0.39 | 12.0 s shared GPU |
| Dynamic-only deformation + crop refinement, frames 1–20 | 22.42 train | 26.76 train | 0.746 train | 6.94 s |
| Final 300-frame diagnostic, held-out mean | 20.07 | 22.44 | — | 7.61 s |

The final diagnostic starts at 0.789 training IoU and ends at 0.001 by frame 299. Its held-out
person PSNR falls from 23.86 to 20.59 dB. Thus a high crop-region PSNR near the end must not be
misread as successful person tracking: it can reflect fitting to incorrectly associated masks.

### Performance

Final 300-frame diagnostic averages:

| Component | Time |
|---|---:|
| MLS deformation and control optimization | 1.95 s/frame |
| Focused refinement | 3.41 s/non-keyframe |
| 500-step full keyframe | 23.16 s/keyframe |
| Mask refresh amortized | 0.33 s/frame |
| End-to-end | 7.61 s/frame (0.131 FPS) |

The fused MLS/SVD path is not the bottleneck: it occupies about 2.7% of deformation time.
Rasterization forward/backward dominates tracking optimization. For baked inference, a frame-299
2.01M-Gaussian checkpoint rasterizes at 317 FPS, or 229 FPS including view-dependent SH colour,
at 937×527 on RTX 3080. Python-to-ffmpeg output is the video-export bottleneck; a production
realtime stream should use GPU-resident NVENC handoff.

### CUDA and portability

- Fused rigid MLS forward/backward and MLS local rotations are custom CUDA.
- Fused SSIM is vendored under its MIT license.
- SelectiveAdam includes a bias-correction repair needed for short keyframe training.
- `python -m ring_init.doctor --compile` builds CUDA extensions for the detected GPU architecture.
  The source package is architecture-neutral and can compile on an RTX PRO 6000 when paired with
  a compatible CUDA-enabled PyTorch and matching nvcc toolkit.

### Limitations and next steps

1. Long crowded plays cause per-camera SAM2 identities to swap or disappear. Detection-led,
   multi-view keyframe association is the primary quality improvement opportunity.
2. Densification grows the final diagnostic model to 2.01M Gaussians. A quality-preserving
   checkpoint consolidation/LOD pass would improve storage and video-export cost.
3. Held-out frames occur every ten frames, coincident with keyframes. Add inter-keyframe held-out
   frames for an unbiased novel-view temporal metric.
4. Replace the Python raw-frame pipe with GPU-resident NVENC/Video Codec SDK integration for
   realtime encoded inference output.
