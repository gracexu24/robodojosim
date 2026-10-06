import json

from robodojosim.layouts import normalize_bottle_mass


def test_normalize_bottle_mass_only_changes_exact_asset_typo(tmp_path):
    layout_dir = tmp_path / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    path = layout_dir / "put_bottles_into_dustbin_0.json"
    data = {
        "Rigid": {
            "bottle": [
                {"category_idx": 22, "physics": {"mass": 22}},
                {"category_idx": 22, "physics": {"mass": 0.22}},
                {"category_idx": 25, "physics": {"mass": 22}},
            ]
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    changed = normalize_bottle_mass(tmp_path)

    assert changed == [path]
    updated = json.loads(path.read_text(encoding="utf-8"))
    masses = [item["physics"]["mass"] for item in updated["Rigid"]["bottle"]]
    assert masses == [0.22, 0.22, 22]
    assert path.with_suffix(".json.robodojosim-original").exists()
    assert normalize_bottle_mass(tmp_path) == []
