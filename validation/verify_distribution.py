"""Verify built artifacts and run the reference CLI from the wheel itself."""

from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root / "dist").glob("raman_tool-*.whl"), key=lambda p: p.stat().st_mtime)
    sources = sorted((root / "dist").glob("raman_tool-*.tar.gz"), key=lambda p: p.stat().st_mtime)
    if not wheels or not sources:
        raise SystemExit("Build a wheel and source distribution first")
    wheel, source = wheels[-1], sources[-1]
    required_modules = [
        "raman_tool/qt_workers.py", "raman_tool/qt_dialogs.py", "raman_tool/history.py",
        "raman_tool/reference_validation.py", "raman_tool/validation.py",
        "raman_tool/readers/text_reader.py", "raman_tool/readers/image_reader.py",
    ]
    with zipfile.ZipFile(wheel) as archive:
        missing = set(required_modules) - set(archive.namelist())
        if missing:
            raise SystemExit(f"Missing wheel modules: {sorted(missing)}")
    with tarfile.open(source, "r:gz") as archive:
        names = {name.partition("/")[2] for name in archive.getnames()}
        required = {
            "tests/fixtures/standards/synthetic-mixture.txt",
            "tests/fixtures/standards/synthetic-mixture.json",
            "tests/test_entrypoint_consistency.py", "docs/VALIDATION.md",
        }
        if required - names:
            raise SystemExit(f"Missing source distribution files: {sorted(required - names)}")
        if any("local_sample_import_audit" in name or "user_reference_strategy_comparison" in name for name in names):
            raise SystemExit("Private measured-reference reports must not be in the distribution")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import raman_tool; assert '.whl' in raman_tool.__file__, raman_tool.__file__; "
        "from raman_tool.cli import main; "
        "raise SystemExit(main(['validate-standard', sys.argv[2]]))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(wheel),
         str(root / "tests/fixtures/standards/synthetic-mixture.json")],
        timeout=60, check=False,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"Verified wheel modules, source fixtures/docs, and wheel CLI: {wheel.name}")


if __name__ == "__main__":
    main()
