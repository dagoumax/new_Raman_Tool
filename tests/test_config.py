from pathlib import Path

from raman_tool.config import load_config, save_config


def test_config_defaults_include_safety_limits():
    config = load_config(Path("missing-config.toml"))

    assert config["safety"]["max_input_file_mb"] == 512
    assert config["safety"]["max_image_pixels"] >= 2048 * 2048


def test_save_and_load_config_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    save_config(
        {
            "safety": {
                "max_input_file_mb": 256,
                "max_text_file_mb": 64,
                "max_data_points": 123456,
                "min_supported_image_pixels": 2048 * 2048,
                "max_image_pixels": 4096 * 4096,
                "max_image_channels": 3,
                "max_baseline_points": 50000,
            }
        },
        path,
    )

    loaded = load_config(path)

    assert loaded["safety"]["max_input_file_mb"] == 256
    assert loaded["safety"]["max_image_channels"] == 3
    assert loaded["safety"]["max_baseline_points"] == 50000
