# Trained two-bottle policies

The repository tracks lightweight metadata for the two trained policies in
[`configs/trained_policies.json`](../configs/trained_policies.json). Model weights and datasets remain on lab storage
and are intentionally excluded from Git.

| Policy ID | Family | Training data | Steps | Status |
|---|---|---:|---:|---|
| `bottle-act-two-color-50` | ACT | 50 successful brown/yellow episodes | 20,000 | trained |
| `bottle-diffusion-two-color-50` | Diffusion Policy | the same 50 episodes | 20,000 | trained |

Both checkpoints use the three-camera plus 16-value Cartesian robot-state interface exposed by
`bottle_lerobot`. Their exact lab paths live in the registry. Historical evaluation reports created before the
bottle-only reward correction used the stricter legacy definition that also required both arms to return home; do
not compare those percentages directly with new bottle-only evaluations.

To evaluate either policy, pass its registered checkpoint to `scripts/evaluate_policy.sh`. The checkpoint family is
detected from its saved LeRobot configuration, so the same adapter supports both models.
