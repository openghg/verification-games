# verification-games

Forward modelling for PARIS verification games 2026.

A notebook-first project layout for work that depends on
[OpenGHG](https://github.com/openghg/openghg) and
[OpenGHG Inversions](https://github.com/openghg/openghg_inversions).

## Repository structure

```text
verification-games/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
├── notebooks/
│   └── 00_experiment_template.ipynb
├── scripts/
│   ├── install_kernel.py
│   └── populate_data.py
├── src/
│   └── verification_games/
│       ├── __init__.py
│       ├── paths.py
│       └── notebook.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Setup

### Preferred: `uv`

Create the environment and install dependencies:

```bash
uv sync --extra jupyter --extra dev
```

Later, add packages with:

```bash
uv add PACKAGE_NAME
uv add --optional notebook jupytext nbdime
```

Sync again after editing `pyproject.toml`:

```bash
uv sync --extra jupyter --extra dev --extra notebook
```

Install the project into the current environment if you want imports from another
virtual environment or a pre-existing kernel:

```bash
python -m pip install -e .
```

That is often the easiest route when you already have a conda or mamba
environment on a cluster.

### Conda / mamba

If you already have a conda or mamba environment with OpenGHG-related packages,
you can usually keep that environment and do:

```bash
python -m pip install -e .
python -m pip install jupyterlab ipykernel jupyter-server-proxy dask-labextension
```

For more detail on environment setup for OpenGHG itself, see the upstream
repository:

- https://github.com/openghg/openghg

## Jupyter kernel for the project environment

Install a dedicated kernel from the `uv`-managed environment:

```bash
uv run python scripts/install_kernel.py
```

Then choose the new kernel in JupyterLab for new notebooks.

This is often smoother than relying on some unrelated default kernel. It also
reduces ambiguity about which environment your notebook is using.

## Suggested workflow

- Put reusable code in `src/verification_games/`.
- Keep notebooks in `notebooks/`.
- Treat notebooks as clients of your package rather than the place where core
  logic lives.
- Put large data in `data/` and keep it out of Git.
- Commit code, small text files, configs, and notebooks you want to preserve.
- Do not commit large generated datasets, Zarr stores, NetCDF files, caches, or
  notebook checkpoints.

A good pattern is:

1. prototype in a notebook
2. move anything reusable into `src/verification_games/`
3. import that code back into the notebook
4. commit the code and the notebook when the notebook is a useful artefact

## Accessing the data directory from notebooks

Use the helper functions instead of hard-coding relative paths:

```python
from verification_games.paths import data_dir

raw_dir = data_dir("raw")
processed_dir = data_dir("processed")
```

That keeps notebook path logic consistent if you later move files around.

## Notebook bootstrap vs editable install

There are three common ways to make `src/` importable from notebooks.

### 1. Project kernel
Use the project kernel installed from the local environment.

Pros:
- cleanest imports
- most reproducible
- no notebook path hacks

Cons:
- needs a kernel install step

### 2. `pip install -e .`
Install the project editable into another existing environment.

Pros:
- good if you already have a working conda environment
- no bootstrap code in notebooks

Cons:
- you need to remember to install into the right environment

### 3. Bootstrap cell
A notebook can add `src/` to `sys.path` at runtime.

Pros:
- simple
- works immediately
- useful on HPC when you do not want to rebuild environments

Cons:
- notebook-specific
- easier to forget or duplicate
- hides environment problems that an editable install or dedicated kernel would
  surface earlier

For one-off exploratory notebooks, bootstrap is often the least disruptive. For
longer-lived work, a dedicated kernel or editable install is cleaner.

## Autoreload for code in `src/`

In IPython or Jupyter you can use:

```python
%load_ext autoreload
%autoreload 2
```

Then import your module normally:

```python
from verification_games.paths import data_dir
```

This will usually pick up changes in Python modules under `src/` without
restarting the kernel.

### Common pitfalls with autoreload

- It does not always behave well with stateful objects created before the code
  change.
- Existing class instances may still use old method definitions or stale state.
- `from module import name` can be more confusing than `import module` when you
  are debugging reload behaviour.
- Some libraries with compiled extensions or global side effects do not reload
  cleanly.
- If you change function signatures, dataclass fields, inheritance structure, or
  module-level constants, a full kernel restart is often safer.

A practical rule:

- small function edits: autoreload is convenient
- structural changes: restart the kernel

Bootstrap can be easier than autoreload-related debugging when the main problem
is simply “make `src/` importable right now”.

## What to commit

Usually commit:

- code in `src/`
- `pyproject.toml`
- `README.md`
- small scripts in `scripts/`
- notebooks that are useful records of an experiment
- small text metadata files in `data/`

Usually do not commit:

- large `.zarr` stores
- large `.nc` files
- generated outputs
- caches
- `.ipynb_checkpoints/`

If you need to ignore additional local data, add repo-specific paths to
`.gitignore`.

## Optional notebook extras

These are not required on day one, but are useful later:

```bash
uv sync --extra notebook
```

This installs:

- `jupytext` for text notebook representations
- `nbdime` for notebook-aware diffs and merges
