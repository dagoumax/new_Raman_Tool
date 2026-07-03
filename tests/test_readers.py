"""测试文件读取器."""

import pytest
import numpy as np
from pathlib import Path
from raman_tool.readers import read_txt, read_asc, read_tif, read_bmp, detect_format, read_file
from raman_tool.readers.base import SUPPORTED_FORMATS

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


@pytest.fixture
def sample_txt_file(tmp_path):
    """创建测试用 TXT 文件."""
    content = (
        "# Raman spectrum data\n"
        "100.0\t10.0\n"
        "200.0\t50.0\n"
        "300.0\t30.0\n"
        "400.0\t20.0\n"
        "500.0\t5.0\n"
    )
    filepath = tmp_path / "test_spectrum.txt"
    filepath.write_text(content)
    return filepath


@pytest.fixture
def sample_asc_file(tmp_path):
    """创建测试用 ASC 文件."""
    content = (
        "Some header text\n"
        "100.0, 10.0\n"
        "200.0, 50.0\n"
        "300.0, 30.0\n"
        "400.0, 20.0\n"
        "500.0, 5.0\n"
    )
    filepath = tmp_path / "test_spectrum.asc"
    filepath.write_text(content)
    return filepath



@pytest.fixture
def sample_sif_file(tmp_path):
    """创建测试用模拟 SIF 文件 (匹配实际 Andor SIF 格式: uint16 LE)."""
    import struct
    intensity = np.array([9539, 9795, 9283, 9027, 9027, 9283, 8771, 9283, 9027, 9539], dtype=np.uint16)
    filepath = tmp_path / "test_spectrum.sif"
    header = b"Andor Technology Multi-Channel File\n\n65538 %d\n" % (len(intensity) * 2)
    footer = b"<?xml version=\"1.0\"?><Signals><Signature>test</Signature></Signals>\n\nSIFX"
    filepath.write_bytes(header + intensity.tobytes() + footer)
    return filepath


class TestTXTReader:
    def test_read_basic(self, sample_txt_file):
        spec = read_txt(sample_txt_file)
        assert spec.size == 5
        assert spec.raman_shift[0] == 100.0
        assert spec.raman_shift[-1] == 500.0
        assert spec.intensity[1] == 50.0
        assert spec.filename == "test_spectrum.txt"

    def test_read_comments_skipped(self, sample_txt_file):
        spec = read_txt(sample_txt_file)
        assert spec.size == 5

    def test_read_empty_raises(self, tmp_path):
        filepath = tmp_path / "empty.txt"
        filepath.write_text("")
        with pytest.raises(Exception):
            read_txt(filepath)


class TestASCReader:
    def test_read_basic(self, sample_asc_file):
        spec = read_asc(sample_asc_file)
        assert spec.size == 5
        assert spec.raman_shift[0] == 100.0

    def test_metadata_filepath(self, sample_asc_file):
        spec = read_asc(sample_asc_file)
        assert "filepath" in spec.metadata


class TestTifReader:
    @pytest.fixture
    def sample_tif(self, tmp_path):
        """创建测试用 2D TIFF 图像."""
        from PIL import Image
        arr = np.random.default_rng(42).integers(0, 255, (10, 50), dtype=np.uint8)
        filepath = tmp_path / "test.tif"
        Image.fromarray(arr).save(filepath)
        return filepath, arr

    def test_read_basic(self, sample_tif):
        filepath, arr = sample_tif
        spec = read_tif(filepath)
        assert spec.size == 50
        assert "image_shape" in spec.metadata

    def test_read_with_row_groups(self, sample_tif):
        filepath, arr = sample_tif
        spec = read_tif(filepath, row_groups="1-5")
        assert spec.metadata["selected_rows"] == 5
        assert spec.size == 50

    def test_read_with_col_merge(self, sample_tif):
        filepath, arr = sample_tif
        spec = read_tif(filepath, col_merge=2)
        assert spec.size == 25

    def test_read_with_row_groups_and_col_merge(self, sample_tif):
        filepath, arr = sample_tif
        spec = read_tif(filepath, row_groups="1-3, 7-9", col_merge=5)
        assert spec.metadata["selected_rows"] == 6
        assert spec.size == 10

    def test_read_1d_tif(self, tmp_path):
        from PIL import Image
        arr = np.array([10, 20, 30, 40, 50], dtype=np.uint8)
        filepath = tmp_path / "test1d.tif"
        Image.fromarray(arr).save(filepath)
        spec = read_tif(filepath)
        assert spec.size == 5

    def test_read_rgb_tif(self, tmp_path):
        from PIL import Image
        arr = np.random.default_rng(42).integers(0, 255, (10, 20, 3), dtype=np.uint8)
        filepath = tmp_path / "test_rgb.tif"
        Image.fromarray(arr).save(filepath)
        spec = read_tif(filepath)
        assert spec.size == 20

    def test_row_groups_parse(self):
        from raman_tool.readers.tif_reader import _parse_row_groups
        groups = _parse_row_groups("1-40, 91-130")
        assert len(groups) == 2
        assert groups[0] == (1, 40)
        assert groups[1] == (91, 130)

        groups = _parse_row_groups("5~10 15-20")
        assert len(groups) == 2
        assert groups[0] == (5, 10)

    def test_col_merge(self):
        from raman_tool.readers.tif_reader import _merge_columns
        arr = np.arange(20).reshape(2, 10).astype(np.float64)
        result = _merge_columns(arr, 2)
        assert result.shape == (2, 5)
        assert result[0, 0] == 0.5  # avg of 0 and 1
        assert result[0, 1] == 2.5  # avg of 2 and 3


class TestDetectFormat:
    def test_detect_txt(self):
        assert detect_format("data.txt") == ".txt"

    def test_detect_asc(self):
        assert detect_format("data.asc") == ".asc"

    def test_detect_sif(self):
        assert detect_format("data.sif") == ".sif"

    def test_detect_tif(self):
        assert detect_format("data.tif") == ".tif"
        assert detect_format("data.tiff") == ".tiff"

    def test_detect_bmp(self):
        assert detect_format("data.bmp") == ".bmp"

    def test_detect_unsupported_raises(self):
        with pytest.raises(ValueError):
            detect_format("data.xyz")

    def test_detect_case_insensitive(self):
        assert detect_format("data.TXT") == ".txt"
        assert detect_format("data.SIF") == ".sif"


class TestSupportedFormats:
    def test_all_formats_have_description(self):
        for ext, desc in SUPPORTED_FORMATS.items():
            assert len(desc) > 0


class TestSIFReader:
    def test_read_basic(self, sample_sif_file):
        from raman_tool.readers.sif_reader import read_sif
        spec = read_sif(sample_sif_file)
        assert spec.size == 10
        assert spec.filename == "test_spectrum.sif"
        assert spec.metadata["format"] == "SIF"

    def test_intensity_values(self, sample_sif_file):
        from raman_tool.readers.sif_reader import read_sif
        spec = read_sif(sample_sif_file)
        assert spec.intensity[0] == 9539.0
        assert spec.intensity[1] == 9795.0
        assert spec.intensity[0] == 9539.0

    def test_raman_shift_computed(self, sample_sif_file):
        from raman_tool.readers.sif_reader import read_sif
        spec = read_sif(sample_sif_file)
        assert len(spec.raman_shift) == 10
        assert spec.raman_shift[0] > 0  # computed from calibration

    def test_read_real_sif(self):
        """测试读取实际的 SIF 数据文件."""
        from raman_tool.readers.sif_reader import read_sif
        sif_path = TEST_DATA_DIR / "10_1.sif"
        if not sif_path.exists():
            pytest.skip("SIF test data not found")
        spec = read_sif(sif_path)
        assert spec.size > 0
        assert spec.metadata["format"] == "SIF"
        assert spec.metadata["num_pixels"] > 0
        assert np.all(spec.intensity > 0)
        # 验证数据在合理范围
        assert np.max(spec.intensity) < 1e6
        assert np.min(spec.intensity) >= 0

    def test_read_fallback_2d(self, tmp_path):
        """测试 2D 图像回退格式."""
        from raman_tool.readers.sif_reader import read_sif
        import struct
        vals = [165.0, 166.0, 164.0, 163.0, 162.0, 164.0, 165.0, 162.0, 166.0, 169.0]
        filepath = tmp_path / "test_2d.sif"
        header = b"DU420_BVF\n 1024 255 25\n" + b" " * 40
        header += b"0\n0\n"
        intensity = struct.pack("<%df" % len(vals), *vals)
        footer = b"<?xml version=\"1.0\"?><Signals/></Signals>\n"
        filepath.write_bytes(header + intensity + footer)
        spec = read_sif(filepath)
        assert spec.size == 10
        assert spec.intensity[0] == pytest.approx(165.0)

    def test_metadata(self, sample_sif_file):
        from raman_tool.readers.sif_reader import read_sif
        spec = read_sif(sample_sif_file)
        meta = spec.metadata
        assert "num_pixels" in meta
        assert "wavelengths" in meta
        assert "laser_wavelength" in meta
