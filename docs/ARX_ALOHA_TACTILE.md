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

- `tactile/<pad>/normal_forces`: `float32 [12, 32]`, newtons
- `tactile/<pad>/positions`: `float32 [12, 32, 3]`, world-space meters

The Isaac integration uses native PhysX contact sensors on the four pad links.
Because RoboDojo creates task objects after Isaac Lab initializes filtered
contact buffers, dynamically filtered per-contact taxel assignment is not
available in this version. The stable implementation records the aggregate pad
normal force and encodes it in the central `2 x 2` cells of the unchanged
`12 x 32` grid; positions cover the full taxel map. This is a useful tactile
contact signal and preserves the target interface, but it is not the full
VT-Refine SDF/virtual-spring spatial pressure model.

## Validation and evaluation

Run the source/schema validator with the Python 3.11 policy environment:

```bash
cd /home/gracexu/robodojo-work/robodojosim-arx-aloha
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/gracexu/miniforge3/envs/RoboDojoPolicy/bin/python \
  scripts/validate_arx_aloha_tactile.py
```

The definitive held-out scripted evaluation uses physical GPU 1 (an RTX 4090)
and writes its tactile episode under:

`/data/gracexu/robodojo-work/arx-aloha-tactile/settled-final-20261027/data`

Its videos are under the matching RoboDojo `eval_result` run ID
`arx-aloha-tactile-settled-final-20261027`. The result counts as success only
after both bottles have been released and satisfy the calibrated bottom-floor
predicate.
