"""Install an IPython kernel for the current project environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def infer_project_name() -> str:
    return Path(__file__).resolve().parents[1].name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=infer_project_name())
    parser.add_argument("--display-name", default=None)
    args = parser.parse_args()

    display_name = args.display_name or f"Python ({args.name})"
    cmd = [
        sys.executable,
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name",
        args.name,
        "--display-name",
        display_name,
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
