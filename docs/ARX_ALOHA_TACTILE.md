# Isolated ARX X5 + ALOHA tactile RoboDojo

This variant keeps the ARX X5 arm kinematics and the existing two-arm 14D
Cartesian action interface, but replaces each stock gripper with ALOHA/VX300
finger geometry. A VT-Refine-size tactile pad is mounted to the inner face of
each finger, producing four tactile streams in total. The original RoboDojo
checkout and its assets are not modified.

## Isolated installation

On the evaluation host the source integration lives at:

- `/home/gracexu/robodojo-work/robodojosim-arx-aloha`
- `/home/gracexu/robodojo-work/RoboDojo-arx-aloha`

The original checkout remains at `/home/gracexu/robodojo-work/RoboDojo`.
Recreate or refresh the isolated overlay with:

```bash
cd /home/gracexu/robodojo-work/robodojosim-arx-aloha
scripts/install_arx_aloha_tactile.sh \
  /home/gracexu/robodojo-work/RoboDojo \
  /home/gracexu/robodojo-work/RoboDojo-arx-aloha
```

The installer retains the large Isaac Lab and cuRobo dependencies as read-only
links, makes independent copies of XPolicyLab and calibrated layouts, creates a
separate asset overlay, and converts the hybrid URDF to
`ARX_ALOHA_calibrated.usd`. Existing artifacts are preserved.

## Fixed-camera tactile variant

A third, independent setup replaces the two wrist cameras with fixed left and
right workspace cameras while retaining the primary overhead camera and the
same ARX+ALOHA tactile arms:

```bash
cd /home/gracexu/robodojo-work/robodojosim-arx-aloha
scripts/install_arx_aloha_fixed_cameras.sh \
  /home/gracexu/robodojo-work/RoboDojo \
  /home/gracexu/robodojo-work/RoboDojo-arx-aloha-fixed-cameras
```

Launch it with `ROBODOJOSIM_ENV_CFG_TYPE=arx_x5_aloha_fixed_cameras`.
Its vision keys are `cam_head`, `cam_left_fixed`, and `cam_right_fixed`.
The fixed-camera asset overlay omits only the wrist-camera declaration from
`robot_config.yml`; the tactile arm USD, four 12x32 VT-Refine streams, 14D
Cartesian action contract, and task geometry are unchanged.

## Mechanical and control changes

- ALOHA/VX300 finger meshes replace the stock ARX finger meshes without
  changing the ARX arm, bases, cameras, or workspace.
- Each gripper finger carries one `26 x 66.65 x 3 mm` tactile pad. The logical
  resolution stays `12 x 32` taxels, matching the requested physical size.
- The right finger axis is reversed in the URDF so both finger joints use the
  same positive-opening convention. Both joint limits are `0..0.042 m`; the
  mimic relation is `+1`.
- RoboDojo registers the copy as `arx_x5_aloha` / `dual_x5_aloha`. The action
  schema is unchanged: two Cartesian end-effector poses plus one scalar
  gripper command per arm (14 values total).
- The two-bottle success condition requires both bottles on the usable bin
  floor and both grippers released. The calibrated dustbin asset includes its
  support below the interior floor, so its `is_A_on_B_bottom` tolerance is
  `0.30 m`; the observed settled gap is about `0.25 m`. The standard RoboDojo
  gripper-open threshold of `0.60` is used.

No ACT architecture, observation encoder, weights, or training code was
changed.

## Tactile observations and HDF5

Four observation streams are emitted:

- `left_left_pad`
- `left_right_pad`
- `right_left_pad`
- `right_right_pad`

Each stored frame contains:

- `tactile/<pad>/normal_forces`: `float32 [12, 32]`, raw VT-Refine
  virtual-spring response
- `tactile/<pad>/positions`: `float32 [12, 32, 3]`, world-space meters

The implementation now follows the VT-Refine/easysim-envs spatial contact
model rather than an aggregate PhysX pad force. On each observation it:

1. Reconciles the tactile mesh registry with the active RoboDojo
   `LayoutManager` records. Newly spawned `Rigid`, `Dynamic`, and `Geometry`
   objects are triangulated from their USD mesh descendants; stale objects are
   removed when the layout changes.
2. Transforms all 384 taxels on each pad into each active object's mesh frame
   and queries signed distance plus the outward surface normal using Warp's
   winding-number mesh query.
3. Converts negative signed distance to penetration depth and applies the
   VT-Refine virtual spring (`KN=1.0`), tangential velocity spring (`KT=0.1`),
   and Coulomb cap (`MU=2.0`). Object and taxel point velocities are evaluated
   at the closest surface point.
4. Projects the resulting world-space force into the taxel frame and stores
   its pad-normal component. Contact therefore occupies the actual affected
   taxels, not a fixed central block.

The values are the raw VT-Refine sensor response, not calibrated real-world
newtons. The native Isaac contact sensors remain only as reliable pad-pose
trackers; they do not determine the taxel pressure field. The implementation is
adapted from `easysim-envs` commit
`5512c3389685b217fae046a32b0b95f2f3020c39`, with its MIT license copied to
`integration/robodojo/arx_aloha/EASYSIM_ENVS_LICENSE`.

## Validation and evaluation

Run the source/schema validator with the Python 3.11 policy environment:

```bash
cd /home/gracexu/robodojo-work/robodojosim-arx-aloha
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/gracexu/miniforge3/envs/RoboDojoPolicy/bin/python \
  scripts/validate_arx_aloha_tactile.py \
  --episode /data/gracexu/robodojo-work/arx-aloha-tactile/vt-refine-sdf-smoke-attempt3-20261027/episode_000000.hdf5
```

The full spatial-backend held-out scripted evaluation uses physical GPU 1 (an
RTX 4090) and writes its successful 250-step tactile episode under:

`/data/gracexu/robodojo-work/arx-aloha-tactile/vt-refine-sdf-smoke-attempt3-20261027`

Its videos are under the matching RoboDojo `eval_result` run ID
`arx-aloha-vt-refine-sdf-smoke-attempt3-20261027`. All four pads recorded
finite spatial fields outside the old central `2 x 2` placeholder; peak
simultaneous activation ranged from 15 to 76 taxels. The task result counts as
success only after both bottles have been released and satisfy the calibrated
bottom-floor predicate.
