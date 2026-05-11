"""Tests for forward-model orchestration helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr

from verification_games.run_forward_model import (
    AVAILABLE_SHARED_STORE_SITES,
    _harmonize_spatial_coords,
    footprint_dataset_from_openghg,
    month_windows,
    open_staged_flux,
    select_flux_sources,
)

EXPECTED_SHARED_STORE_SITE_COUNT = 31
EXPECTED_RESOLVED_H_BACK_COUNT = 2


def test_available_sites_skip_puy() -> None:
    """The default production list omits the unavailable PUY footprint."""
    assert "PUY" not in AVAILABLE_SHARED_STORE_SITES
    assert len(AVAILABLE_SHARED_STORE_SITES) == EXPECTED_SHARED_STORE_SITE_COUNT


def test_month_windows_returns_exclusive_month_bounds() -> None:
    """Month windows use inclusive starts and exclusive ends."""
    windows = month_windows(start="2021-01-01", end="2021-04-01")
    assert [(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")) for start, end in windows] == [
        ("2021-01-01", "2021-02-01"),
        ("2021-02-01", "2021-03-01"),
        ("2021-03-01", "2021-04-01"),
    ]


def test_select_flux_sources_uses_source_metadata() -> None:
    """Flux source filtering uses source-level metadata coordinates."""
    flux = xr.DataArray(
        np.zeros((2, 1, 1, 4), dtype=np.float32),
        dims=("time", "lat", "lon", "source"),
        coords={
            "time": [0, 1],
            "lat": [50.0],
            "lon": [-2.0],
            "source": ["co2_BASE_GPP", "co2_BASE_FF", "o2_BASE_GPP", "o2_ATEN_GPP"],
            "species": ("source", ["co2", "co2", "o2", "o2"]),
            "games_scenario": ("source", ["BASE", "BASE", "BASE", "ATEN"]),
            "sector": ("source", ["GPP", "FF", "GPP", "GPP"]),
        },
    )

    selected = select_flux_sources(flux, species=("o2",), scenarios=("BASE",), sectors=("GPP",))
    assert list(selected["source"].values) == ["o2_BASE_GPP"]


def test_footprint_dataset_from_openghg_splits_residual_h_back() -> None:
    """OpenGHG-style H_back footprints are split for the Numba kernel."""
    ds = xr.Dataset(
        {
            "fp_HiTRes": (
                ("time", "lat", "lon", "H_back"),
                np.ones((2, 1, 1, 3), dtype=np.float32),
            )
        },
        coords={
            "time": np.array(["2021-01-01T00", "2021-01-01T01"], dtype="datetime64[h]"),
            "lat": [50.0],
            "lon": [-2.0],
            "H_back": [0.0, 1.0, 2.0],
        },
    )

    prepared = footprint_dataset_from_openghg(ds)

    assert set(prepared.data_vars) == {"fp_time_resolved", "fp_residual"}
    assert prepared["fp_time_resolved"].sizes["H_back"] == EXPECTED_RESOLVED_H_BACK_COUNT
    assert "H_back" not in prepared["fp_residual"].dims


def test_harmonize_spatial_coords_accepts_same_grid_float_noise() -> None:
    """Tiny coordinate dtype/noise differences should not trigger regridding."""
    fp = xr.Dataset(
        coords={
            "lat": np.array([50.0, 51.0], dtype=np.float64),
            "lon": np.array([-2.0, -1.0], dtype=np.float64),
        }
    )
    flux = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        dims=("lat", "lon"),
        coords={
            "lat": np.array([50.0, 51.0000005], dtype=np.float32),
            "lon": np.array([-2.0, -1.0000005], dtype=np.float32),
        },
    )

    aligned = _harmonize_spatial_coords(flux, fp)

    assert aligned["lat"].identical(fp["lat"])
    assert aligned["lon"].identical(fp["lon"])


def test_harmonize_spatial_coords_rejects_different_grid() -> None:
    """Forward modelling should fail explicitly rather than interpolate fluxes."""
    fp = xr.Dataset(coords={"lat": [50.0, 51.0], "lon": [-2.0, -1.0]})
    flux = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        dims=("lat", "lon"),
        coords={"lat": [50.0, 51.1], "lon": [-2.0, -1.0]},
    )

    try:
        _harmonize_spatial_coords(flux, fp)
    except ValueError as exc:
        assert "regrid before forward modelling" in str(exc)
    else:
        raise AssertionError("expected spatial grid mismatch to raise ValueError")


def test_open_staged_flux_defaults_to_dask_backed_zarr(tmp_path) -> None:
    """Default Zarr opening should preserve lazy Dask arrays."""
    path = tmp_path / "staged_flux.zarr"
    ds = xr.Dataset(
        {
            "flux": (
                ("time", "lat", "lon", "source"),
                np.zeros((2, 2, 2, 1), dtype=np.float32),
                {"units": "mol m-2 s-1"},
            )
        },
        coords={"time": [0, 1], "lat": [50.0, 51.0], "lon": [-2.0, -1.0], "source": ["co2_BASE_GPP"]},
    ).chunk({"time": 1, "lat": 2, "lon": 2, "source": 1})
    ds.to_zarr(path)

    flux = open_staged_flux(path)

    assert hasattr(flux.data, "__dask_graph__")
