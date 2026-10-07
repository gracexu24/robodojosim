import json

from robodojosim.layouts import centralize_dustbin, configure_training_dustbin, normalize_bottle_mass


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


def test_training_dustbin_is_shortened_and_placed_on_table(tmp_path):
    layout_dir = tmp_path / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    path = layout_dir / "put_bottles_into_dustbin_0.json"
    data = {
        "Rigid": {
            "bottle": [
                {"xlim": [0.05, 0.45], "default_pos": [0.25, -0.1, 0.86]},
                {"xlim": [-0.35, 0.05], "default_pos": [-0.15, -0.1, 0.86]},
            ]
        },
        "Geometry": {
            "dustbin": [
                {
                    "category_idx": 0,
                    "label": "dustbin",
                    "xlim": [-0.63, -0.63],
                    "ylim": [-0.1, -0.1],
                    "zlim": [0.35, 0.35],
                    "relative_plane": "Ground",
                    "scale": [1.0, 1.0, 1.0],
                    "physics": {"type": "geometry"},
                    "default_pos": [-0.63, -0.1, 0.4],
                },
                {
                    "category_idx": 9,
                    "label": "dustbin",
                    "xlim": [-0.63, -0.63],
                    "default_pos": [-0.63, 0.2, 0.4],
                },
            ]
        },
        "Table": {"default_pos": [0.09, -0.05, 0.74], "scale": [1.0, 1.1, 0.05]},
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    changed = configure_training_dustbin(tmp_path)

    assert changed == [path]
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["Geometry"]["dustbin"][0]["default_pos"] == [0.0, -0.1, 0.83]
    assert updated["Geometry"]["dustbin"][0]["xlim"] == [0.0, 0.0]
    assert updated["Geometry"]["dustbin"][0]["ylim"] == [-0.1, -0.1]
    assert updated["Geometry"]["dustbin"][0]["zlim"] == [0.83, 0.83]
    assert updated["Geometry"]["dustbin"][0]["relative_plane"] == "Table"
    assert updated["Geometry"]["dustbin"][0]["scale"] == [1.0, 0.6, 0.2]
    assert updated["Geometry"]["dustbin"][0]["physics"]["collision"] is True
    assert updated["Geometry"]["dustbin"][1]["default_pos"] == [-0.63, 0.2, 0.4]
    assert updated["Rigid"]["bottle"][0]["default_pos"][0] == 0.39
    assert updated["Rigid"]["bottle"][0]["xlim"] == [0.3, 0.48]
    assert updated["Rigid"]["bottle"][1]["default_pos"][0] == -0.34
    assert updated["Rigid"]["bottle"][1]["xlim"] == [-0.38, -0.3]
    assert centralize_dustbin(tmp_path) == []


def test_training_dustbin_migrates_old_centered_floor_variant(tmp_path):
    layout_dir = tmp_path / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    path = layout_dir / "put_bottles_into_dustbin_0.json"
    path.write_text(
        json.dumps(
            {
                "Geometry": {
                    "dustbin": [
                        {
                            "category_idx": 0,
                            "label": "dustbin",
                            "relative_plane": "Ground",
                            "default_pos": [0.0, -0.1, 0.4],
                        }
                    ]
                },
                "Table": {"default_pos": [0.0, 0.0, 0.74], "scale": [1.0, 1.0, 0.05]},
            }
        ),
        encoding="utf-8",
    )

    assert centralize_dustbin(tmp_path) == [path]
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["Geometry"]["dustbin"][0]["default_pos"] == [0.0, -0.1, 0.83]


def test_training_dustbin_migrates_old_half_height_tabletop_variant(tmp_path):
    layout_dir = tmp_path / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    path = layout_dir / "put_bottles_into_dustbin_0.json"
    path.write_text(
        json.dumps(
            {
                "Geometry": {
                    "dustbin": [
                        {
                            "category_idx": 0,
                            "label": "dustbin",
                            "relative_plane": "Table",
                            "scale": [1.0, 1.0, 0.5],
                            "default_pos": [0.0, 0.3, 0.9275],
                        }
                    ]
                },
                "Table": {"default_pos": [0.0, 0.0, 0.74], "scale": [1.0, 1.0, 0.05]},
            }
        ),
        encoding="utf-8",
    )

    assert centralize_dustbin(tmp_path) == [path]
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["Geometry"]["dustbin"][0]["scale"] == [1.0, 0.6, 0.2]
    assert centralize_dustbin(tmp_path) == []
