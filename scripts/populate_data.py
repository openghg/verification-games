"""Placeholder for project-specific data download or staging logic.

Populate this script with the commands needed to fetch or prepare the datasets
used by the project.
"""

from __future__ import annotations

from verification_games.paths import data_dir


def main() -> None:
    print("Populate data under:", data_dir())


if __name__ == "__main__":
    main()
