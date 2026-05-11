---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
      format_version: '0.13'
      jupytext_version: 1.17.0
kernelspec:
  display_name: verification-games
  language: python
  name: verification-games
---

# Forward-model staging pipeline

This notebook stages the PARIS fluxes as a source-aware Zarr store, registers
that store in `games_catalog`, and runs resumable `fp_x_flux` site/month jobs.

```{code-cell} ipython3
from pathlib import Path

from dask import compute as dask_compute
import pandas as pd
import xarray as xr

from verification_games.cluster import (
    load_slurm_account,
    make_slurm_cluster,
    wait_for_workers,
    warm_forward_model_numba,
)
from verification_games.flux_stage import (
    DEFAULT_CATALOG_PATH,
    build_staged_flux_dataset,
    find_flux_records,
    open_games_catalog,
    register_staged_flux_zarr,
    stage_flux_zarr,
    validate_staged_flux_zarr,
)
from verification_games.run_forward_model import (
    AVAILABLE_SHARED_STORE_SITES,
    compute_fp_x_flux,
    get_month_footprint,
    iter_site_month_runs,
    open_staged_flux,
    run_site_month,
)
```

```{code-cell} ipython3
VG_PATH = Path("/group/chem/acrg/verification_games_round_2")
CATALOG_PATH = DEFAULT_CATALOG_PATH
WORK_ROOT = VG_PATH / "forward_model_intermediates"

STAGED_FLUX_ZARR = (
    VG_PATH
    / "games_catalog"
    / "data"
    / "flux"
    / "derived"
    / "PARIS_CTE-HR_filled_fluxes_staged"
    / "zarr_v1"
    / "paris_cte-hr_filled_fluxes_EUROPE_flux_stage_filled_nan_zero_forward_model_input_202012-202112.zarr"
)
STAGED_FLUX_TEMP = WORK_ROOT / "staging_tmp"
FP_X_FLUX_OUTPUT_DIR = WORK_ROOT / "fp_x_flux"
LOG_DIR = WORK_ROOT / "dask_logs"

games_cat = open_games_catalog(CATALOG_PATH)
```

## Stage all fluxes

Set `RUN_STAGING = True` when ready. This writes to a temporary Zarr directory,
validates the staged structure, then moves it into place.

```{code-cell} ipython3
RUN_STAGING = False
REGISTER_STAGED_ZARR = False

if RUN_STAGING:
    staged_path, input_records = stage_flux_zarr(
        games_cat,
        STAGED_FLUX_ZARR,
        temp_parent=STAGED_FLUX_TEMP,
        overwrite=False,
    )
else:
    staged_path = STAGED_FLUX_ZARR
    input_records = None

if staged_path.exists():
    summary = validate_staged_flux_zarr(staged_path, full_nan_check=False)
else:
    print(f"Staged flux Zarr does not exist yet: {staged_path}")
    summary = None
summary
```

```{code-cell} ipython3
if REGISTER_STAGED_ZARR:
    if not staged_path.exists():
        raise FileNotFoundError(staged_path)
    if input_records is None:
        input_records = find_flux_records(games_cat)
    staged_record = register_staged_flux_zarr(games_cat, staged_path, input_records)
    staged_record
```

## Optional full NaN check

This scans the whole staged Zarr and is intentionally separate from the
structure validation.

```{code-cell} ipython3
RUN_FULL_NAN_CHECK = False

if RUN_FULL_NAN_CHECK:
    if not staged_path.exists():
        raise FileNotFoundError(staged_path)
    validate_staged_flux_zarr(staged_path, full_nan_check=True)
```

## Cluster

```{code-cell} ipython3
START_CLUSTER = False

if START_CLUSTER:
    account = load_slurm_account("slurm.toml")
    cluster, client = make_slurm_cluster(
        log_path=LOG_DIR,
        account=account,
        walltime="08:00:00",
        n_workers=8,
        cores_per_worker=1,
        memory_per_worker_gb=48,
    )
    wait_for_workers(client, 8)
    warm_forward_model_numba(client)
    client
```

## Smoke run

The smoke run uses all staged sources for MHD in January 2021 and writes one
`time x source` Zarr output.

```{code-cell} ipython3
RUN_SMOKE = False

if RUN_SMOKE:
    smoke = run_site_month(
        site="MHD",
        start_date="2021-01-01",
        end_date="2021-02-01",
        flux_zarr_path=staged_path,
        output_dir=FP_X_FLUX_OUTPUT_DIR,
        time_chunk=24,
        source_chunk=4,
        fillna_zero=False,
    )
    smoke
```

## Smoke comparison

This compares the staged, already-filled Zarr with a direct lazy stack of the
original NetCDF flux records where NaNs are filled by the forward-model helper.

```{code-cell} ipython3
RUN_SMOKE_COMPARISON = False

if RUN_SMOKE_COMPARISON:
    fp = get_month_footprint(site="MHD", start_date="2021-01-01", end_date="2021-02-01")

    staged_flux = open_staged_flux(staged_path)
    staged_expr = compute_fp_x_flux(
        fp,
        staged_flux,
        time_chunk=24,
        source_chunk=4,
        fillna_zero=False,
    )

    raw_flux_ds, _ = build_staged_flux_dataset(games_cat, fill_value=None)
    raw_expr = compute_fp_x_flux(
        fp,
        raw_flux_ds["flux"],
        time_chunk=24,
        source_chunk=4,
        fillna_zero=True,
    )

    staged_result, raw_result = dask_compute(staged_expr, raw_expr)
    xr.testing.assert_allclose(staged_result, raw_result, rtol=1e-5, atol=1e-8)
    staged_result
```

## Production loop

This skips `PUY` because it is not currently available in the shared footprint
Zarr store. Existing outputs are skipped by default, so interrupted runs can be
continued by re-running this cell.

```{code-cell} ipython3
RUN_PRODUCTION = False

if RUN_PRODUCTION:
    completed = []
    for site, start_date, end_date in iter_site_month_runs(sites=AVAILABLE_SHARED_STORE_SITES):
        run = run_site_month(
            site=site,
            start_date=start_date,
            end_date=end_date,
            flux_zarr_path=staged_path,
            output_dir=FP_X_FLUX_OUTPUT_DIR,
            time_chunk=24,
            source_chunk=4,
            fillna_zero=False,
            skip_existing=True,
        )
        completed.append(run)

    pd.DataFrame([run.__dict__ for run in completed])
```
