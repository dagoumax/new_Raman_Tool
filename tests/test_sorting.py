from pathlib import Path

from raman_tool.sorting import natural_sorted


def test_natural_sorted_orders_embedded_numbers():
    files = [
        Path("sample_100.sif"),
        Path("sample_10.sif"),
        Path("sample_2.sif"),
        Path("sample_1.sif"),
    ]

    assert [f.name for f in natural_sorted(files)] == [
        "sample_1.sif",
        "sample_2.sif",
        "sample_10.sif",
        "sample_100.sif",
    ]


def test_natural_sorted_uses_file_name_for_paths():
    files = [
        r"C:\data\10.sif",
        r"C:\data\1.sif",
        r"C:\data\2.sif",
    ]

    assert natural_sorted(files) == [
        r"C:\data\1.sif",
        r"C:\data\2.sif",
        r"C:\data\10.sif",
    ]
