import hashlib
import subprocess
import sys
from pathlib import Path


EXPECTED_MD5 = "e0dcd19471b6761feaf6b50f294a7ff8"


def test_fig1_checksum():
    subprocess.run([sys.executable, "run.py", "fig1"], check=True)
    p = Path("figures/fig01_test.pgm")
    assert p.exists()
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    assert md5 == EXPECTED_MD5
