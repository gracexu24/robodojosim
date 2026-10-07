import json

from robodojosim.layouts import centralize_dustbin, normalize_bottle_mass


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


def test_centralize_dustbin_only_changes_public_bottle_task_position(tmp_path):
    layout_dir = tmp_path / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    path = layout_dir / "put_bottles_into_dustbin_0.json"
    data = {
        "Geometry": {
            "dustbin": [
                {
                    "category_idx": 0,
                    "label": "dustbin",
                    "xlim": [-0.63, -0.63],
                    "default_pos": [-0.63, -0.1, 0.4],
                },
                {
                    "category_idx": 9,
                    "label": "dustbin",
                    "xlim": [-0.63, -0.63],
                    "default_pos": [-0.63, 0.2, 0.4],
                },
            ]
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    changed = centralize_dustbin(tmp_path)

    assert changed == [path]
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["Geometry"]["dustbin"][0]["default_pos"] == [0.0, -0.1, 0.4]
    assert updated["Geometry"]["dustbin"][0]["xlim"] == [0.0, 0.0]
    assert updated["Geometry"]["dustbin"][1]["default_pos"] == [-0.63, 0.2, 0.4]
    assert centralize_dustbin(tmp_path) == []
