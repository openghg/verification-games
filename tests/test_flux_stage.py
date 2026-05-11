"""Tests for flux staging helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr

from verification_games.flux_stage import FluxDatasetInput, source_label, stack_flux_sources

EXPECTED_O2_ATEN_GPP_MEAN = 11.0


def _flux_dataset(offset: float = 0.0) -> xr.Dataset:
    coords = {
        "time": np.array(["2021-01-01T00", "2021-01-01T01"], dtype="datetime64[h]"),
        "lat": np.array([50.0, 51.0], dtype=np.float32),
        "lon": np.array([-2.0, -1.0], dtype=np.float32),
    }
    shape = (2, 2, 2)
    return xr.Dataset(
        {
            "GPP": (("time", "lat", "lon"), np.full(shape, offset + 1, dtype=np.float32)),
            "TER": (("time", "lat", "lon"), np.full(shape, np.nan, dtype=np.float32)),
        },
        coords=coords,
    )


def test_source_label_is_stable() -> None:
    """Source labels normalize species/scenario text."""
    assert source_label("O2", "base", "GPP") == "o2_BASE_GPP"


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
