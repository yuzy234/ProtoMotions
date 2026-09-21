# EasyMimic SMPL tracking

This integration keeps the EasyMimic world coordinates unchanged:

- right-handed, Z-up;
- meters;
- the PLY mesh remains at its original transform;
- reference motion reset does not sample a new XY terrain location.

For parallel Isaac Gym training, the canonical PLY and motion are translated
together onto spatial collision tiles.  The translation is invisible to local
observations and rewards: every humanoid keeps exactly the same pose relative
to its reconstructed scene.  The number of tiles is selected automatically so
that no tile contains more than 16 environments.  This replaces the accidental
all-environments-at-one-XY layout that overloaded GPU PhysX broadphase.  A
single-environment viewer still uses one unshifted canonical scene.

The tracking experiment also disables the interactive projectile pool.  Those
actors are only used by the viewer's `J` key and otherwise waste pinned memory
during headless training.

The source file contains two people (`global_id` 1 and 2), each with 91 frames.
The packaged MotionLib therefore contains two 30 FPS clips and full evaluation
runs both clips once.

## Environment

The working local Isaac Gym environment is `igym`, not `myenv`:

```bash
conda activate igym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /home/yzy/part1/workspace/test/baseline/ProtoMotions
export PYTHONPATH="$PWD"
```

## Rebuild the MotionLib

```bash
python data/scripts/convert_easymimic_smpl_to_motionlib.py \
  /home/yzy/part1/workspace/test/easymimic/outputs/easymimic_2026_6_3_sync_videos_frames60_150_vggtomega_twostage_sam3maskinit_mainline_default_fullopt/isaac_zup_export/optimized_fused_global_smpl_isaac_zup.pt \
  data/easymimic/optimized_fused_global_smpl_isaac_zup_motionlib.pt \
  --mesh-path /home/yzy/part1/workspace/test/easymimic/outputs/easymimic_2026_6_3_sync_videos_frames60_150_vggtomega_twostage_sam3maskinit_mainline_default_fullopt/isaac_zup_export/heightfield_mesh_isaac_zup.ply \
  --fps 30 \
  --per-motion-smpl-shape
```

Pass `--global-id 1` or `--global-id 2` to package only one person.
With `--per-motion-smpl-shape`, each clip uses its own EasyMimic betas and
generates a matching CRISP/SMPLSim MJCF under `data/easymimic/assets/mjcf`.
EasyMimic does not export gender here, so generated bodies use neutral SMPL.
The converter follows CRISP's SMPL convention and sets the simulation root to
`trans_world + local_pos[0]`, because SMPL `trans` is the model translation, not
the Pelvis joint position.

## Motion Tracker

```bash
python protomotions/inference_agent.py \
  --checkpoint data/pretrained_models/motion_tracker/smpl-terrains/last.ckpt \
  --motion-file data/easymimic/optimized_fused_global_smpl_isaac_zup_motionlib.pt \
  --terrain-mesh /home/yzy/part1/workspace/test/easymimic/outputs/easymimic_2026_6_3_sync_videos_frames60_150_vggtomega_twostage_sam3maskinit_mainline_default_fullopt/isaac_zup_export/heightfield_mesh_isaac_zup.ply \
  --simulator isaacgym \
  --num-envs 1 \
  --loop-motion \
  --motion-id 0 \
  --preserve-reference-world-position \
  --smpl-shape-from-motion
```

## MaskedMimic

```bash
python protomotions/inference_agent.py \
  --checkpoint data/pretrained_models/masked_mimic/smpl/last.ckpt \
  --motion-file data/easymimic/optimized_fused_global_smpl_isaac_zup_motionlib.pt \
  --terrain-mesh /home/yzy/part1/workspace/test/easymimic/outputs/easymimic_2026_6_3_sync_videos_frames60_150_vggtomega_twostage_sam3maskinit_mainline_default_fullopt/isaac_zup_export/heightfield_mesh_isaac_zup.ply \
  --simulator isaacgym \
  --num-envs 1 \
  --loop-motion \
  --motion-id 0 \
  --preserve-reference-world-position \
  --masked-mimic-full-conditioning \
  --smpl-shape-from-motion
```

Full conditioning exposes translation and rotation for all bodies supported by
the checkpoint:

- Pelvis
- L_Ankle
- R_Ankle
- L_Hand
- R_Hand
- Head

The five target times are deterministic consecutive control steps. No random
body mask, constraint type, future time, motion ID, motion fragment, or reset
location is used.

Use `--motion-id 0` for `global_id=1` and `--motion-id 1` for `global_id=2`.
At the last frame, the same motion is reset to time zero at its original world
position and starts again.

Add `--headless` to either command to disable the viewer.
