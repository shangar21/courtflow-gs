# Method figure assets

| Requested figure | Asset | Source |
| --- | --- | --- |
| IMG A2 | `img_a2_sam2_mask_overlay.png` | Frame-0 training camera 00 and accepted instance masks. |
| IMG A3 | `img_a3_label_image.png` | Frame-0 mutually-exclusive label image $L_c$. |
| IMG A4 | `img_a4_hull_comparison.png` | Recomputed strict silhouette intersection versus stored occlusion-aware hull. |
| IMG A5 | `img_a5_crop_mast3r_matches.png` | Highest-count saved crop-MASt3R pair. |
| IMG A6 | `img_a6_canonical_heldout_render.png` | Stage-A held-out view 13 render. |
| IMG B0 | `img_b0_controls_by_group.png` | Frame-0 controls and subsampled canonical splats. |
| IMG B2 | `img_b2_prompt_boxes_and_masks.png` | Saved frame-150 SAM2 labels over prompt boxes, camera 11. |
| IMG B5 | `img_b5_composited_render_f150.png` | Frame-150 held-out view-13 diagnostic render. |

Regenerate with `python tools/export_report_figures.py` from the repository root.  The script
reads the cached diagnostic run at `/media/storage/peripheral_frame0_1080p/ring_init_out_f0/basketball`
by default; use `--run-root` to select another completed run.
