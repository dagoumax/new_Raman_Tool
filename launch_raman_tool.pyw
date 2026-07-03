from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
SRC = ROOT / "src"
SITE_PACKAGES = VENV / "Lib" / "site-packages"
SCRIPTS = VENV / "Scripts"

os.environ.setdefault("VIRTUAL_ENV", str(VENV))
os.environ["PATH"] = str(SCRIPTS) + os.pathsep + os.environ.get("PATH", "")

for path in (SITE_PACKAGES, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from raman_tool.qt_gui import main


if __name__ == "__main__":
    raise SystemExit(main())