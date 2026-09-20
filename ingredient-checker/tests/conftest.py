import sys
from pathlib import Path

# chain.py and barcode_detector.py use bare top-level imports, so the package
# directory (not just the tests folder) must be importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
