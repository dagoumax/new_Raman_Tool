import pytest

from raman_tool.models import Spectrum
from raman_tool.processing.baseline import subtract_baseline
from raman_tool.readers.asc_reader import read_asc
from raman_tool.safety import MAX_BASELINE_POINTS, MAX_TEXT_FILE_BYTES, InputLimitError, refresh_limits


def setup_function():
    refresh_limits()


def test_asc_reader_rejects_oversized_text_file(monkeypatch, tmp_path):
    path = tmp_path / "huge.asc"
    path.write_text("1 2\n", encoding="utf-8")

    class FakeStat:
        st_size = MAX_TEXT_FILE_BYTES + 1

    monkeypatch.setattr(type(path), "stat", lambda self: FakeStat())

    with pytest.raises(InputLimitError):
        read_asc(path)


def test_arpls_rejects_too_many_points():
    n = MAX_BASELINE_POINTS + 1
    spec = Spectrum(
        raman_shift=list(range(n)),
        intensity=[1.0] * n,
        filename="huge.asc",
    )

    with pytest.raises(InputLimitError):
        subtract_baseline(spec)



def test_image_pixel_limit_rejects_large_dimensions():
    from raman_tool.safety import MAX_IMAGE_PIXELS, check_image_pixels

    with pytest.raises(InputLimitError):
        check_image_pixels(MAX_IMAGE_PIXELS + 1, 1)



def test_image_array_limit_allows_2048_square_rgb():
    from raman_tool.safety import check_image_array_values, check_image_pixels

    check_image_pixels(2048, 2048)
    check_image_array_values(2048 * 2048 * 3)


def test_image_array_limit_rejects_excessive_values():
    from raman_tool.safety import MAX_IMAGE_ARRAY_VALUES, check_image_array_values

    with pytest.raises(InputLimitError):
        check_image_array_values(MAX_IMAGE_ARRAY_VALUES + 1)
