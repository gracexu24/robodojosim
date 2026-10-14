# Deprecated controller configurations

The supported production controller is defined by `bottle_task.json` and
`policy_data.json`. It uses the brown/yellow two-bottle layout and direct
per-arm placement into the shared tabletop bin.

The following calibration files are retained only to reproduce the earlier
single-bottle and cross-arm-handoff experiments:

- `bottle_calibration.json`
- `bottle_drop_calibration.json`
- `bottle_handoff_calibration.json`
- `bottle_left_drop_calibration.json`
- `bottle_right_calibration.json`
- `bottle_right_drop_calibration.json`
- `bottle_right_horizontal_drop_calibration.json`

They are not supported starting points for new datasets. Setting
`direct_right_drop` to `false` invokes the deprecated handoff branch and emits
a Python `DeprecationWarning`. Setting `ROBODOJOSIM_TWO_BOTTLE_TASK=0` also
prints a shell warning before restoring the old four-bottle task.
