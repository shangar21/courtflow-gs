from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json


# Full-image and tracking-image pixel lengths; both grids scale with capture_scale.
PIXEL_LENGTH_FIELDS = ("instance_box_padding_px", "crop_min_size_px", "reprojection_px", "occluder_splat_px",
                       "mask_prune_dilate_px", "crop_render_padding_px", "bg_person_mask_dilation_px",
                       "v2_keyframe_focus_pad_px", "ball_crop_px", "ball_reproj_px",
                       "tracking_loss_margin_px", "tracking_box_margin_px")


@dataclass
class Config:
    """Every threshold of the pipeline. Metric lengths are in meters and converted to scene
    units with `units_per_meter` (auto-estimated from the floor step when None)."""
    # Input/output and fixed-camera contract.
    data_root: str = "data"
    out_root: str = "out"
    calibration: str = "calibration.json"
    camera_count: int = 12
    device: str = "cuda"
    units: str | None = "scene"
    units_per_meter: float | None = None
    image_pattern: str = "images/cam_{cam:02d}.png"
    seed: int = 0
    # Floor / scale estimation.
    floor_ransac_threshold: float = 0.01          # scene units, before scale is known
    floor_ransac_iterations: int = 2000
    person_height_m: float = 1.9                  # used to estimate units_per_meter
    court_margin_m: float = 1.0                   # occupancy box margin around the floor extent
    court_floor_percentile: float = 2.0           # robust floor extent percentile
    # Person detection + SAM2 masks.
    detector: str = "fasterrcnn_resnet50_fpn_v2"
    detector_min_size: int = 2108
    detector_max_size: int = 3800
    detector_score_threshold: float = 0.5
    detector_max_detections: int = 300
    sam2_checkpoint: str | None = None
    sam2_config: str | None = None
    min_mask_area_px: int = 64
    # Occupancy-based instance association.
    occupancy_voxel_m: float = 0.05
    occupancy_height_m: float = 2.4
    occupancy_min_views: int = 4
    occupancy_allowed_misses: int = 2
    instance_min_height_m: float = 1.2
    instance_min_voxels: int = 30
    instance_split_footprint_m: float = 1.1       # footprint extent above which a component is split
    instance_box_padding_px: int = 6
    instance_min_iou: float = 0.2                 # detector mask vs projected occupancy silhouette
    instance_max_hist_distance: float = 0.5       # Bhattacharyya, re-prompted mask vs other views
    # Geometry.
    mast3r_model: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    mast3r_size: int = 512
    pair_confidence: float = 1.5                  # MASt3R descriptor confidence threshold
    matches_per_pair: int = 50_000
    crop_padding_fraction: float = 0.35
    crop_min_size_px: int = 96
    crop_seed_stride: int = 2                     # reciprocal-NN seeds on the instance mask (512 crop px)
    reprojection_px: float = 1.5
    min_triangulation_angle_deg: float = 3.0
    multi_view_color_l1: float = 0.12
    multi_view_min_extra_views: int = 1
    visual_hull_voxel_m: float = 0.01
    hull_carve_dilation_voxels: int = 1
    use_hull: bool = True                         # ablation: False seeds from triangulated points only, no hull prune
    hull_allowed_misses: int = 0                  # cameras allowed to disagree (1 inflates hulls ~2.5x)
    occluder_min_extra_views: int = 2
    occluder_splat_px: int = 4
    occluder_margin_m: float = 0.3
    occluder_min_height_m: float = 0.5            # noisy floor points below this act as fake occluders
    hull_fill_distance_voxels: float = 2.0
    player_voxel_m: float = 0.005
    background_voxel_m: float = 0.10
    background_hull_margin_m: float = 0.10
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    normal_radius_voxels: float = 3.0
    # Gaussian initialization.
    sh_degree: int = 1
    opacity_init: float = 0.5
    knn_scale_k: int = 3
    normal_scale_ratio: float = 0.1
    # t0 optimization (players).
    t0_iterations: int = 7000
    t0_views_per_step: int = 4
    t0_lr_means: float = 1.6e-4                   # multiplied by the instance extent
    t0_lr_scales: float = 5e-3
    t0_lr_quats: float = 1e-3
    t0_lr_opacity: float = 5e-2
    t0_lr_sh0: float = 2.5e-3
    t0_lr_shN: float = 2.5e-3 / 20
    t0_lr_means_final_factor: float = 0.01
    l1_weight: float = 1.0
    dssim_weight: float = 0.2
    ssim_window: int = 11
    alpha_weight: float = 0.1
    depth_weight: float = 0.05
    depth_decay_fraction: float = 0.5
    scale_ratio_max: float = 10.0
    scale_reg_weight: float = 0.01
    drop_gaussian: bool = True
    drop_gaussian_rate: float = 0.1
    densify_grad_threshold: float = 0.0002        # 3DGS default; multiplied below
    densify_gradient_multiplier: float = 2.0
    densify_start_iter: int = 500
    densify_every: int = 100
    densify_until_fraction: float = 0.5
    max_gaussians_per_instance: int = 150_000
    opacity_prune: float = 0.01
    mask_prune_min_views: int = 2
    mask_prune_every: int = 500
    mask_prune_dilate_px: int = 3
    hull_prune_dilation_voxels: int = 2           # prune person Gaussians whose centre leaves the dilated hull
    crop_render_padding_px: int = 16
    # Background (instance 0).
    background: bool = True
    bg_iterations: int = 10_000
    bg_max_gaussians: int = 600_000
    bg_views_per_step: int = 2
    bg_person_mask_dilation_px: int = 2
    bg_composite_persons: bool = True             # frozen person models rendered in front during bg training
    # Stage B frame data (reproduces the frame-0 preprocessing).
    capture_scale: float = 0.5                    # raw 4K capture decode scale (0.5 -> 1920x1080); pixel params are tuned at 0.5
    videos_dir: str | None = None                 # cameras/view_XXX.mp4
    distorted_cameras_txt: str | None = None      # COLMAP OPENCV intrinsics of the 1080p frames
    distorted_images_txt: str | None = None
    photometric_json: str | None = None
    frame_workers: int = 6
    frame_format: str = "png"                     # training frames: png (lossless) | jpg (~18x faster writes; for 4K capture)
    frame_jpeg_quality: int = 95
    frame_gpu_preprocess: bool = False            # undistortion + colour table on the GPU in the extraction workers
    sam2_workers: int = 1                         # cameras propagated concurrently by SAM2 (capped by host memory)
    eval_frame_stride: int = 10                   # held-out cameras decoded/evaluated every N frames
    # Stage B v2 (QuickCapture-style whole-scene deformation + keyframe retraining).
    stage_b_mode: str = "scene"                  # scene (v2) | persons (v1: persons only, frozen background)
    control_points_background: int = 1000
    v2_mask_weight: float = 0.3                   # per-person SAM2 mask BCE in the deformation loss
    reassoc_max_move_m: float = 3.0               # keyframe identity repair: max 3D move between keyframes
    reassoc_distance_scale_m: float = 1.0
    reassoc_appearance_weight: float = 1.0
    reassoc_min_iou: float = 0.1
    reassoc_seed_margin_m: float = 0.3            # split merged components among people whose last position lies within
    v2_mask_source: str = "reassoc"               # mask frames: reprompt (SAM2 prompted from tracked people) | video (SAM2 video labels) | reassoc (video labels + 3D identity repair; production default)
    v2_mask_every: int = 5                        # 0: SAM2 video masks every frame; N: SAM2 re-prompted from tracked people every N frames
    v2_touchup_iterations: int = 5                # SH DC + opacity iterations between keyframes
    # Optional dense refinement experiment.  Unlike touch-up this optimizes the complete
    # Gaussian model, closing the quality gap between expensive keyframes.  A value of zero
    # retains the lightweight touch-up-only path.
    v2_refine_iterations: int = 100
    v2_refine_views_per_step: int = 2
    v2_refine_focus: bool = True                   # train dynamic-content crops rather than full frames
    v2_refine_dynamic_only: bool = False           # allow visible background splats to adapt in focused refinement
    v2_keyframe_every: int = 10
    v2_keyframe_iterations: int = 500               # sweep "G": +2.9 dB held-out person PSNR vs 200 @ small lr
    v2_keyframe_views_per_step: int = 2
    v2_keyframe_lr_means: float = 1.6e-4          # x scene extent (Stage A level; 1.6e-5 could not move Gaussians)
    v2_keyframe_densify_start: int = 50
    v2_keyframe_densify_every: int = 50
    v2_keyframe_densify_until: float = 0.6
    v2_keyframe_growth: float = 0.05              # max Gaussian growth per keyframe
    selective_adam: bool = True                   # keyframe/background training: update only visible Gaussians
    v2_prefetch: bool = True                      # load frame f+1 on a background thread while tracking frame f
    v2_dynamic_only_deform: bool = True           # rasterize only persons/ball during deformation over a cached background
    v2_keyframe_focus: bool = False               # experiment: keyframes train crops around dynamic content ...
    v2_keyframe_full_every: int = 25              # ... with a full-frame keyframe every N frames
    v2_keyframe_focus_pad_px: int = 48
    v2_keyframe_protect: bool = True              # no opacity pruning at keyframes (lost players must not be deleted)
    v2_health_iou: float = 0.10                   # keyframe re-init below this IoU; enabled only with identity-repaired masks
    v2_boost_iou: float = 0.5                     # persons below this IoU get a stronger anchor until the next keyframe
    v2_anchor_boost: float = 10.0
    v2_video_views: tuple[int, ...] = (13, 26)    # held-out source views rendered to video online
    v2_video_scale: float = 0.5
    v2_save_ply_every: int = 25
    # Ball (own instance + control points).
    ball: bool = True
    ball_radius_m: float = 0.1215                 # size-7 basketball
    ball_score_threshold: float = 0.3
    ball_crop_px: int = 256                       # detection crop around the predicted projection
    ball_crop_detector_size: int = 800
    ball_reproj_px: float = 12.0
    ball_max_jump_m: float = 2.0                  # per-frame gate on the triangulated position
    ball_max_lost: int = 5                        # full-frame re-detection after this many misses
    ball_init_min_views: int = 3
    ball_lost_damping: float = 0.9                # velocity damping per unobserved frame
    ball_restitution: float = 0.8
    ball_gaussians: int = 800
    ball_clear_factor: float = 1.3                # remove other Gaussians within this x radius
    ball_control_points: int = 16
    ball_anchor_weight: float = 10.0
    video_fps: float = 25.0
    # Online MLS (Stage B).
    control_points: int = 1024                    # per player (v1 used 512)
    control_knn: int = 6
    gaussian_control_knn: int = 8
    mls_sigma_knn: int = 3
    mls_svd_epsilon: float = 1e-6
    mls_blend_cp_rotation: bool = False
    mls_rotation_blend_weight: float = 0.0
    tracking_iterations: int = 50                 # spec default 100; 50 reaches the 12-camera/300-iteration IoU within 0.01
    tracking_lr_translation_m: float = 0.003      # Adam lr for per-control offsets (meters)
    tracking_lr_rotation: float = 0.005           # Adam lr for control quaternions
    tracking_optimize_rotations: bool = False     # False: ARAP uses each control's local MLS rotation (q still used by the blend)
    tracking_warm_start: bool = True
    tracking_loss_margin_px: int = 16             # foreground loss band around the propagated masks
    tracking_alpha_weight: float = 1.0            # mask BCE weight in tracking (large motion: masks are the robust cue)
    tracking_mask_mode: str = "instance"          # instance: per-instance one-hot channels | union: single alpha
    tracking_min_iterations: int = 15             # no early stop before this
    tracking_adam_eps: float = 1e-8
    tracking_global_translation: bool = True      # per-person shared translation + per-control offsets
    tracking_centroid_anchor: bool = True         # init + soft-constrain each person's translation to triangulated mask centroids
    tracking_anchor_weight: float = 0.1
    tracking_anchor_scale_m: float = 0.3          # Huber scale of the anchor residual
    tracking_lr_global_m: float = 0.02            # Adam lr of the per-person translation (meters)
    tracking_box_margin_px: int = 48              # render crop around all persons per camera
    tracking_scale: float = 0.5                   # image scale for tracking renders/losses (eval is always full res)
    tracking_profile: bool = True                 # synchronize for the per-stage timing breakdown
    tracking_batch_cameras: bool = False          # pad selected tracking cameras and rasterize them in one gsplat call
    tracking_radius_clip: float = 0.0             # px; cull sub-pixel splats during deformation (0 disables)
    training_radius_clip: float = 0.0             # px; same culling threshold for 3DGS training/keyframes
    tracking_eval_cameras: int = 24               # held-out cameras evaluated per eval frame
    tracking_save_ply: bool = False               # deformed per-frame .ply export (disk heavy)
    render_near_plane_fraction: float = 0.8       # novel-view videos: near plane = fraction of nearest seed depth
    cameras_per_iteration: int = 4
    arap_weight: float = 1.0                      # ARAP residual / rest edge length^2
    temporal_weight: float = 0.01                 # acceleration / control spacing^2 (0.1 over-damps fast motion)
    early_stop_delta: float = 2e-3                # relative loss change over early_stop_patience iterations
    early_stop_patience: int = 10
    new_geometry_alpha_miss: float = 0.2
    refine_appearance_every: int = 0
    # Evaluation (held-out cameras are read only by eval/heldout.py).
    eval_sparse: str | None = None
    eval_images: str | None = None
    eval_training_views: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)
    heldout_cameras: tuple[int, ...] = (0, 4, 8)
    baseline_heldout_psnr_db: float = 17.5002

    def validate(self) -> None:
        if self.camera_count != 12:
            raise ValueError("This pipeline requires exactly 12 calibrated ring cameras.")
        if self.units is None or not str(self.units).strip():
            raise ValueError("Config.units must state a consistent scene unit.")
        if not 0 <= self.drop_gaussian_rate < 1:
            raise ValueError("drop_gaussian_rate must be in [0, 1).")
        if self.gaussian_control_knn < 3:
            raise ValueError("gaussian_control_knn must be at least 3 for rigid MLS.")
        if self.frame_format not in ("png", "jpg"):
            raise ValueError("frame_format must be 'png' or 'jpg'.")
        if self.sh_degree not in (0, 1):
            raise ValueError("sh_degree must be 0 or 1 (higher degrees overfit with 12 views).")

    def rescale_pixel_params(self, reference_scale: float = 0.5) -> None:
        """Keep image-pixel thresholds at the same physical footprint when ``capture_scale``
        differs from the resolution they were tuned at. Call once on a freshly loaded config."""
        f = self.capture_scale / reference_scale
        if f == 1: return
        for name in PIXEL_LENGTH_FIELDS:
            value = getattr(self, name)
            setattr(self, name, type(value)(round(value * f)) if isinstance(value, int) else value * f)
        self.min_mask_area_px = int(round(self.min_mask_area_px * f * f))

    def m(self, meters: float) -> float:
        """Convert a metric length to scene units."""
        if self.units_per_meter is None:
            raise RuntimeError("units_per_meter unknown: run the floor step or set it in the config.")
        return meters * self.units_per_meter

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            config = cls()
        else:
            values = json.loads(Path(path).read_text())
            known = {f.name for f in fields(cls)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"Unknown config keys: {sorted(unknown)}")
            for key in ("heldout_cameras", "eval_training_views", "v2_video_views"):
                if key in values:
                    values[key] = tuple(values[key])
            config = cls(**values)
        config.validate()
        return config

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=list) + "\n")
