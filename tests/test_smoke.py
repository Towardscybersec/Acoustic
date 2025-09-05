import subprocess, sys


def test_cli_help():
    subprocess.run([sys.executable, "run.py", "--help"], check=True)
