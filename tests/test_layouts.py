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
                {"xlim": [0.05, 0.45], "ylim": [-0.25, 0.02], "default_pos": [0.25, -0.1, 0.86]},
                {"xlim": [-0.35, 0.05], "ylim": [-0.25, 0.02], "default_pos": [-0.15, -0.1, 0.86]},
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
    assert updated["Geometry"]["dustbin"][0]["default_pos"] == [0.0, -0.1, 0.81375]
    assert updated["Geometry"]["dustbin"][0]["xlim"] == [0.0, 0.0]
    assert updated["Geometry"]["dustbin"][0]["ylim"] == [-0.1, -0.1]
    assert updated["Geometry"]["dustbin"][0]["zlim"] == [0.81375, 0.81375]
    assert updated["Geometry"]["dustbin"][0]["relative_plane"] == "Table"
    assert updated["Geometry"]["dustbin"][0]["scale"] == [0.65, 0.6, 0.15]
    assert updated["Geometry"]["dustbin"][0]["physics"]["collision"] is True
    assert updated["Geometry"]["dustbin"][1]["default_pos"] == [-0.63, 0.2, 0.4]
    assert updated["Rigid"]["bottle"][0]["default_pos"][:2] == [-0.35, -0.17]
    assert updated["Rigid"]["bottle"][0]["xlim"] == [-0.35, -0.35]
    assert updated["Rigid"]["bottle"][0]["ylim"] == [-0.17, -0.17]
    assert updated["Rigid"]["bottle"][1]["default_pos"][:2] == [-0.35, 0.01]
    assert updated["Rigid"]["bottle"][1]["xlim"] == [-0.35, -0.35]
    assert updated["Rigid"]["bottle"][1]["ylim"] == [0.01, 0.01]
    assert "robodojosim" not in updated
    assert centralize_dustbin(tmp_path) == []


def test_training_dustbin_removes_legacy_top_level_marker(tmp_path):
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
                            "xlim": [0.0, 0.0],
                            "ylim": [-0.1, -0.1],
                            "zlim": [0.78125, 0.78125],
                            "relative_plane": "Table",
                            "scale": [0.8, 0.6, 0.05],
                            "physics": {"collision": True},
                            "default_pos": [0.0, -0.1, 0.78125],
                        }
                    ]
                },
                "Table": {"default_pos": [0.0, 0.0, 0.74], "scale": [1.0, 1.0, 0.05]},
                "robodojosim": {"training_layout_version": 1},
            }
        ),
        encoding="utf-8",
    )

    assert configure_training_dustbin(tmp_path) == [path]
    assert "robodojosim" not in json.loads(path.read_text(encoding="utf-8"))
    assert configure_training_dustbin(tmp_path) == []


def test_training_layout_uses_calibrated_right_bottle_lanes(tmp_path):
    layout_dir = tmp_path / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    path = layout_dir / "put_bottles_into_dustbin_0.json"
    bottles = [
        {"default_pos": [0.0, 0.0, 0.8], "xlim": [-0.35, 0.45], "ylim": [-0.25, 0.02]}
        for _ in range(4)
    ]
    path.write_text(
        json.dumps(
            {
                "Rigid": {"bottle": bottles},
                "Geometry": {"dustbin": [{"category_idx": 0, "label": "dustbin"}]},
                "Table": {"default_pos": [0.0, 0.0, 0.74], "scale": [1.0, 1.0, 0.05]},
            }
        ),
        encoding="utf-8",
    )

    configure_training_dustbin(tmp_path)

    updated = json.loads(path.read_text(encoding="utf-8"))["Rigid"]["bottle"]
    assert [item["default_pos"][:2] for item in updated] == [
        [-0.35, -0.17],
        [-0.35, 0.01],
        [0.38, 0.01],
        [0.38, -0.17],
    ]


def test_training_layout_uses_only_upright_graspable_plastic_bottles(tmp_path):
    layout_dir = tmp_path / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    path = layout_dir / "put_bottles_into_dustbin_3.json"
    bottles = [
        {
            "category_idx": category,
            "default_pos": [0.0, 0.0, 0.8],
            "default_ori": [0.5, 0.5, 0.5, 0.5],
            "rotate_rand": True,
            "physics": {"mass": 22.0},
        }
        for category in (5, 50, 47, 22)
    ]
    path.write_text(
        json.dumps(
            {
                "Rigid": {"bottle": bottles},
                "Geometry": {"dustbin": [{"category_idx": 0, "label": "dustbin"}]},
                "Table": {"default_pos": [0.0, 0.0, 0.74], "scale": [1.0, 1.0, 0.05]},
            }
        ),
        encoding="utf-8",
    )

    configure_training_dustbin(tmp_path)

    updated = json.loads(path.read_text(encoding="utf-8"))["Rigid"]["bottle"]
    assert [item["category_idx"] for item in updated] == [55, 0, 2, 25]
    assert [item["default_pos"][2] for item in updated] == [0.805278, 0.796709, 0.796733, 0.797006]
    assert [item["physics"]["mass"] for item in updated] == [0.15, 0.19, 0.19, 0.15]
    assert [item["default_ori"] for item in updated] == [
        [0.5006, -0.4993, 0.4995, 0.5006],
        [0.4995, -0.5003, 0.5005, 0.4997],
        [-0.4993, 0.5005, -0.5004, -0.4998],
        [0.5001, -0.5, 0.5, 0.4999],
    ]
    assert [item["qpos"] for item in updated] == [item["default_ori"] for item in updated]
    assert all(item["rotate_rand"] is False for item in updated)


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
    assert updated["Geometry"]["dustbin"][0]["default_pos"] == [0.0, -0.1, 0.81375]


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
    assert updated["Geometry"]["dustbin"][0]["scale"] == [0.65, 0.6, 0.15]
    assert centralize_dustbin(tmp_path) == []


def test_training_dustbin_migrates_failed_outer_and_rear_lane_variants(tmp_path):
    layout_dir = tmp_path / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    path = layout_dir / "put_bottles_into_dustbin_0.json"
    path.write_text(
        json.dumps(
            {
                "Rigid": {
                    "bottle": [
                        {
                            "xlim": [0.30, 0.48],
                            "ylim": [0.10, 0.20],
                            "default_pos": [0.39, 0.155556, 0.86],
                        }
                    ]
                },
                "Geometry": {
                    "dustbin": [
                        {
                            "category_idx": 0,
                            "label": "dustbin",
                            "relative_plane": "Table",
                            "scale": [1.0, 0.6, 0.2],
                            "default_pos": [0.0, -0.1, 0.83],
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
    bottle = updated["Rigid"]["bottle"][0]
    assert bottle["default_pos"][:2] == [-0.35, -0.17]
    assert bottle["xlim"] == [-0.35, -0.35]
    assert bottle["ylim"] == [-0.17, -0.17]
    assert centralize_dustbin(tmp_path) == []
