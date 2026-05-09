# data/

Use this directory for local project data and generated artefacts.

Suggested subdirectories:

- `raw/` for immutable downloaded inputs
- `interim/` for partially processed data
- `processed/` for derived outputs that can be regenerated

Large `.zarr` and `.nc` datasets are ignored by default via `.gitignore`.
