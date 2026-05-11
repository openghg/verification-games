"""Dask/SLURM helpers for verification-games forward modelling."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from forward_model_tests.mod_obs_functions import warm_numba_block_kernel_with_source_worker


THREAD_LIMIT_PROLOGUE = [
    "export OMP_NUM_THREADS=1",
    "export OPENBLAS_NUM_THREADS=1",
    "export MKL_NUM_THREADS=1",
    "export VECLIB_MAXIMUM_THREADS=1",
    "export NUMEXPR_NUM_THREADS=1",
]


def load_slurm_account(path: str | Path = "slurm.toml") -> str:
    """Load the SLURM account name from a small TOML file."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing SLURM config file: {config_path}")

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    try:
        return str(data["acct"])
    except KeyError as exc:
        raise KeyError(f"'acct' not found in {config_path}") from exc


def make_slurm_cluster(  # noqa: PLR0913
    *,
    log_path: str | Path,
    account: str,
    walltime: str = "08:00:00",
    n_workers: int = 8,
    cores_per_worker: int = 1,
    memory_per_worker_gb: int = 48,
    local_directory: str | Path | None = None,
    queue: str | None = None,
    death_timeout: int = 60,
):
    """Create a conservative one-process-per-worker SLURM Dask cluster."""
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    if cores_per_worker <= 0:
        raise ValueError("cores_per_worker must be positive")
    if memory_per_worker_gb <= 0:
        raise ValueError("memory_per_worker_gb must be positive")

    log_dir = Path(log_path).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = Path(local_directory).expanduser() if local_directory else log_dir / "dask-worker-space"
    worker_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {}
    if queue is not None:
        kwargs["queue"] = queue

    cluster = SLURMCluster(
        processes=1,
        cores=cores_per_worker,
        memory=f"{memory_per_worker_gb}GB",
        walltime=walltime,
        account=account,
        log_directory=str(log_dir),
        local_directory=str(worker_dir),
        death_timeout=death_timeout,
        job_script_prologue=THREAD_LIMIT_PROLOGUE,
        **kwargs,
    )
    cluster.scale(jobs=n_workers)
    return cluster, Client(cluster)


def wait_for_workers(client, n_workers: int, timeout_s: int = 600) -> None:
    """Block until at least ``n_workers`` Dask workers have connected."""
    client.wait_for_workers(n_workers, timeout=timeout_s)


def warm_forward_model_numba(client) -> dict[str, str]:
    """Compile the forward-model Numba kernels on every connected worker."""
    return client.run(warm_numba_block_kernel_with_source_worker)
