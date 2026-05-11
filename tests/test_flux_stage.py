"""Tests for flux staging helpers."""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr

from verification_games import flux_stage
from verification_games.flux_stage import (
    FluxDatasetInput,
    _flux_data_array,
    convert_flux_units,
    source_label,
    stack_flux_sources,
)
from verification_games.units import cf_ureg

EXPECTED_O2_ATEN_GPP_MEAN = 11.0


def _flux_dataset(offset: float = 0.0) -> xr.Dataset:
    coords = {
        "time": np.array(["2021-01-01T00", "2021-01-01T01"], dtype="datetime64[h]"),
        "lat": np.array([50.0, 51.0], dtype=np.float32),
        "lon": np.array([-2.0, -1.0], dtype=np.float32),
    }
    shape = (2, 2, 2)
    ds = xr.Dataset(
        {
            "GPP": (("time", "lat", "lon"), np.full(shape, offset + 1, dtype=np.float32)),
            "TER": (("time", "lat", "lon"), np.full(shape, np.nan, dtype=np.float32)),
        },
        coords=coords,
    )
    ds["GPP"].attrs["units"] = "mol m-2 s-1"
    ds["TER"].attrs["units"] = "mol m-2 s-1"
    return ds


def test_source_label_is_stable() -> None:
    """Source labels normalize species/scenario text."""
    assert source_label("O2", "base", "GPP") == "o2_BASE_GPP"


def test_cf_registry_parses_cf_units_and_pint_accessor_is_available() -> None:
    """The lightweight CF registry activates pint-xarray without OpenGHG."""
    assert "openghg" not in sys.modules

    source = cf_ureg.parse_expression("mol m-2 s-1")
    target = cf_ureg.parse_expression("mol/m2/s")
    assert np.isclose(float(source.to(target.units).magnitude), float(target.magnitude))
    assert hasattr(xr.DataArray([1.0]), "pint")


def test_convert_flux_units_uses_cf_registry() -> None:
    """Flux units are converted with the CF/Pint registry when needed."""
    data = xr.DataArray([1.0], attrs={"units": "micromol m-2 s-1"})

    converted = convert_flux_units(data)

    assert converted.attrs["units"] == "mol m-2 s-1"
    assert converted.attrs["source_units_before_staging"] == "micromol m-2 s-1"
    assert np.isclose(float(converted.values[0]), 1e-6)


def test_flux_data_array_repairs_duplicate_dimension_names() -> None:
    """Malformed HDF5 dimension scales can be normalised from shape and coords."""
    ds = _flux_dataset()
    broken = xr.DataArray(
        ds["GPP"].data,
        dims=("time", "time", "time"),
        attrs=dict(ds["GPP"].attrs),
        name="GPP",
    )
    ds["GPP"] = broken

    repaired = _flux_data_array(ds, "GPP")

    assert repaired.dims == ("time", "lat", "lon")
    assert repaired.shape == ds["GPP"].shape
    assert repaired.attrs["units"] == "mol m-2 s-1"


def test_stack_flux_sources_stacks_metadata_and_fills_nans() -> None:
    """Stacking creates a source dimension and replaces NaNs with zero."""
    inputs = [
        FluxDatasetInput(
            species="co2",
            games_scenario="BASE",
            record_id="1",
            path="/tmp/base_co2.nc",
            dataset=_flux_dataset(0),
        ),
        FluxDatasetInput(
            species="o2",
            games_scenario="ATEN",
            record_id="2",
            path="/tmp/aten_o2.nc",
            dataset=_flux_dataset(10),
        ),
    ]

    ds = stack_flux_sources(
        inputs,
        sectors=("GPP", "TER"),
        output_chunks={"time": 1, "lat": 2, "lon": 2, "source": 2},
    )
    flux = ds["flux"].compute()

    assert flux.dims == ("time", "lat", "lon", "source")
    assert list(flux["source"].values) == [
        "co2_BASE_GPP",
        "co2_BASE_TER",
        "o2_ATEN_GPP",
        "o2_ATEN_TER",
    ]
    assert list(flux["species"].values) == ["co2", "co2", "o2", "o2"]
    assert list(flux["games_scenario"].values) == ["BASE", "BASE", "ATEN", "ATEN"]
    assert list(flux["sector"].values) == ["GPP", "TER", "GPP", "TER"]
    assert not bool(flux.isnull().any())
    assert float(flux.sel(source="co2_BASE_TER").max()) == 0.0
    assert float(flux.sel(source="o2_ATEN_GPP").mean()) == EXPECTED_O2_ATEN_GPP_MEAN
    assert flux.attrs["units"] == "mol m-2 s-1"
    assert "history" in ds.attrs


def test_stack_flux_sources_can_preserve_nans_for_reference_path() -> None:
    """The smoke comparison can keep original NaNs until compute-time fill."""
    inputs = [
        FluxDatasetInput(
            species="co2",
            games_scenario="BASE",
            record_id="1",
            path="/tmp/base_co2.nc",
            dataset=_flux_dataset(0),
        )
    ]

    ds = stack_flux_sources(inputs, sectors=("TER",), fill_value=None)

    assert bool(ds["flux"].isnull().any())
    assert ds.attrs["nan_policy"] == "NaNs preserved."


def test_stage_flux_zarr_dry_run_skips_write(monkeypatch, tmp_path) -> None:
    """Dry runs build the lazy dataset but do not materialise Zarr."""
    ds = stack_flux_sources(
        [
            FluxDatasetInput(
                species="co2",
                games_scenario="BASE",
                record_id="1",
                path="/tmp/base_co2.nc",
                dataset=_flux_dataset(0),
            )
        ],
        sectors=("GPP",),
    )
    records = [object()]
    calls = []

    def fake_build(*args, **kwargs):
        calls.append(("build", args, kwargs))
        return ds, records

    def fake_write(*_args, **_kwargs):
        raise AssertionError("dry_run should not write Zarr")

    monkeypatch.setattr(flux_stage, "build_staged_flux_dataset", fake_build)
    monkeypatch.setattr(flux_stage, "write_staged_flux_zarr", fake_write)

    target_path, returned_records = flux_stage.stage_flux_zarr(
        object(),
        tmp_path / "staged.zarr",
        dry_run=True,
    )

    assert calls
    assert target_path == tmp_path / "staged.zarr"
    assert returned_records == records
    assert not target_path.exists()
