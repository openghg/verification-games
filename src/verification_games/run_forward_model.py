"""Run footprint-times-flux forward-model intermediates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from uuid import uuid4

from forward_model_tests.mod_obs_functions import fp_x_flux_sum_space_numba
import numpy as np
from numcodecs import Blosc
import pandas as pd
import xarray as xr

from verification_games.metadata import append_history


SITE_LIST = (
    "BIR",
    "BSD",
    "CBW",
    "CMN",
    "GAT",
    "HEI",
    "HEL",
    "HFD",
    "HPB",
    "HTM",
    "HUN",
    "JFJ",
    "KIT",
    "KRE",
    "LIN",
    "LUT",
    "MHD",
    "NOR",
    "OPE",
    "OXK",
    "PAL",
    "PUY",
    "RGL",
    "SAC",
    "SSL",
    "STE",
    "TAC",
    "TOH",
    "TRN",
    "UTO",
    "WAO",
    "WES",
)
AVAILABLE_SHARED_STORE_SITES = tuple(site for site in SITE_LIST if site != "PUY")
FP_X_FLUX_UNITS = "1"
EXPECTED_FLUX_UNITS = "mol m-2 s-1"
EXPECTED_FOOTPRINT_UNITS = "m2 s mol-1"
SPATIAL_COORD_TOLERANCE = 1e-5
DEFAULT_ZARR_COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
FP_X_FLUX_MODIFICATIONS = (
    "Computed footprint times staged flux and summed over latitude and longitude; "
    "baseline contribution not applied."
)


@dataclass(frozen=True)
class ForwardModelRun:
    """Metadata for one site/month fp_x_flux output."""

    site: str
    start_date: str
    end_date: str
    output_path: str
    flux_zarr_path: str
    footprint_store: str
    footprint_domain: str
    footprint_species: str
    inlet: str | None
    time_chunk: int
    source_chunk: int
    fillna_zero: bool
    created_at: str
    status: str


def month_windows(
    *,
    start: str | pd.Timestamp = "2021-01-01",
    end: str | pd.Timestamp = "2022-01-01",
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return month start/end pairs with an exclusive end timestamp."""
    starts = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS", inclusive="left")
    return [(month_start, month_start + pd.DateOffset(months=1)) for month_start in starts]


def open_staged_flux(zarr_path: str | Path, *, chunks: dict[str, int] | None = None) -> xr.DataArray:
    """Open the staged flux Zarr and return its ``flux`` DataArray."""
    open_kwargs = {} if chunks is None else {"chunks": chunks}
    ds = xr.open_zarr(Path(zarr_path), **open_kwargs)
    if "flux" not in ds:
        raise ValueError(f"Staged flux Zarr is missing variable 'flux': {zarr_path}")
    flux = ds["flux"]
    units = flux.attrs.get("units")
    if units != EXPECTED_FLUX_UNITS:
        raise ValueError(f"Expected staged flux units {EXPECTED_FLUX_UNITS!r}, found {units!r}")
    return flux


def select_flux_sources(
    flux: xr.DataArray,
    *,
    species: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    sectors: Sequence[str] | None = None,
) -> xr.DataArray:
    """Select staged flux sources by metadata coordinates."""
    mask = xr.ones_like(flux["source"], dtype=bool)
    if species is not None:
        mask = mask & flux["species"].isin([value.lower() for value in species])
    if scenarios is not None:
        mask = mask & flux["games_scenario"].isin([value.upper() for value in scenarios])
    if sectors is not None:
        mask = mask & flux["sector"].isin(list(sectors))
    return flux.sel(source=flux["source"].where(mask, drop=True))


def footprint_dataset_from_openghg(data) -> xr.Dataset:
    """Convert an OpenGHG footprint object or Dataset to the kernel input form."""
    ds = data.data if hasattr(data, "data") else data
    if not isinstance(ds, xr.Dataset):
        raise TypeError("Expected an xarray Dataset or an object with a Dataset .data attribute")
    if {"fp_time_resolved", "fp_residual"} <= set(ds.data_vars):
        return ds[["fp_time_resolved", "fp_residual"]]
    if "fp_HiTRes" not in ds:
        raise ValueError("Footprint Dataset must contain 'fp_HiTRes' or prepared footprint variables")

    hitres = ds["fp_HiTRes"]
    out = xr.Dataset(
        {
            "fp_time_resolved": hitres.isel(H_back=slice(None, -1)),
            "fp_residual": hitres.isel(H_back=-1, drop=True),
        },
        attrs=ds.attrs,
    )
    return out


def _validate_footprint_units(fp: xr.Dataset) -> None:
    for name in ("fp_time_resolved", "fp_residual"):
        units = fp[name].attrs.get("units")
        if units is not None and units != EXPECTED_FOOTPRINT_UNITS:
            raise ValueError(f"Expected footprint units {EXPECTED_FOOTPRINT_UNITS!r} for {name}, found {units!r}")


def get_month_footprint(  # noqa: PLR0913
    *,
    site: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    domain: str = "europe",
    store: str = "shared_store_zarr",
    species: str = "co2",
    inlet: str | None = None,
) -> xr.Dataset:
    """Retrieve one site/month footprint from OpenGHG."""
    from openghg.retrieve import get_footprint  # noqa: PLC0415

    kwargs = {
        "site": site.lower(),
        "domain": domain,
        "store": store,
        "species": species,
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
    }
    if inlet is not None:
        kwargs["inlet"] = inlet
    print(f"Fetching footprint: {kwargs}")
    fp = footprint_dataset_from_openghg(get_footprint(**kwargs))
    _validate_footprint_units(fp)
    return fp


def _flux_slice_for_footprint(flux: xr.DataArray, fp: xr.Dataset) -> xr.DataArray:
    h_back = np.rint(fp["fp_time_resolved"]["H_back"].values).astype("int64")
    max_lag = int(h_back.max()) if h_back.size else 0
    start = pd.Timestamp(fp["fp_time_resolved"]["time"].values[0]) - pd.Timedelta(hours=max_lag)
    end = pd.Timestamp(fp["fp_time_resolved"]["time"].values[-1]) + pd.Timedelta(hours=1)
    return flux.sel(time=slice(start, end))


def _coordinate_values_match(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    tolerance: float = SPATIAL_COORD_TOLERANCE,
) -> bool:
    """Return whether two 1D coordinates have the same values for alignment."""
    if left.shape != right.shape:
        return False

    left_values = left.values
    right_values = right.values
    if np.issubdtype(left_values.dtype, np.number) and np.issubdtype(right_values.dtype, np.number):
        return bool(np.allclose(left_values, right_values, rtol=0.0, atol=tolerance, equal_nan=True))
    return bool(np.array_equal(left_values, right_values))


def _harmonize_spatial_coords(flux: xr.DataArray, fp: xr.Dataset) -> xr.DataArray:
    """Assign footprint spatial coordinates to flux when grids match numerically."""
    updates: dict[str, xr.DataArray] = {}
    for dim in ("lat", "lon"):
        if dim not in flux.coords or dim not in fp.coords:
            continue
        if flux.coords[dim].identical(fp.coords[dim]):
            continue
        if not _coordinate_values_match(flux.coords[dim], fp.coords[dim]):
            raise ValueError(
                f"Flux and footprint {dim!r} coordinates do not match; regrid before forward modelling."
            )
        updates[dim] = fp.coords[dim]

    if not updates:
        return flux
    return flux.assign_coords(updates)


def compute_fp_x_flux(
    fp: xr.Dataset,
    flux: xr.DataArray,
    *,
    time_chunk: int = 24,
    source_chunk: int = 4,
    fillna_zero: bool = False,
) -> xr.DataArray:
    """Build the lazy ``(footprint * flux).sum(lat, lon)`` expression."""
    flux_slice = _flux_slice_for_footprint(flux, fp)
    flux_slice = _harmonize_spatial_coords(flux_slice, fp)
    lat_chunk = int(fp.sizes["lat"])
    lon_chunk = int(fp.sizes["lon"])
    result = fp_x_flux_sum_space_numba(
        fp,
        flux_slice.to_dataset(name="flux"),
        cast_float32=True,
        fillna_zero=fillna_zero,
        time_chunk=time_chunk,
        lat_chunk=lat_chunk,
        lon_chunk=lon_chunk,
        source_chunk=source_chunk,
    )
    result.name = "fp_x_flux"
    result.attrs.update(
        {
            "description": "(footprint * flux).sum('lat', 'lon') intermediate without baseline contribution.",
            "baseline_status": "not_applied",
            "units": FP_X_FLUX_UNITS,
            "flux_units": EXPECTED_FLUX_UNITS,
            "footprint_units": EXPECTED_FOOTPRINT_UNITS,
            "modifications": FP_X_FLUX_MODIFICATIONS,
        }
    )
    result.attrs = append_history(result.attrs, FP_X_FLUX_MODIFICATIONS)
    return result


def zarr_encoding(
    ds: xr.Dataset,
    *,
    compressor=DEFAULT_ZARR_COMPRESSOR,
) -> dict[str, dict[str, object]]:
    """Return Zarr encoding for fp_x_flux outputs."""
    return {name: {"compressor": compressor} for name in ds.data_vars}


def _stringify_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _zarr_safe_string_coords(ds: xr.Dataset) -> xr.Dataset:
    """Convert object/string coordinates to plain Python strings before Zarr writes."""
    updates = {}
    for name, coord in ds.coords.items():
        if coord.dtype.kind not in {"O", "S", "U"}:
            continue
        values = np.asarray(coord.values, dtype=object)
        string_values = np.array([_stringify_value(value) for value in values.ravel()], dtype=object).reshape(
            values.shape
        )
        updates[name] = xr.DataArray(string_values, dims=coord.dims, attrs=dict(coord.attrs), name=name)

    if not updates:
        return ds
    return ds.assign_coords(updates)


def fp_x_flux_output_path(output_dir: str | Path, *, site: str, start_date: str | pd.Timestamp) -> Path:
    """Return the canonical site/month output path."""
    month = pd.Timestamp(start_date).strftime("%Y%m")
    return Path(output_dir).expanduser() / site.lower() / f"{site.lower()}_{month}_fp_x_flux.zarr"


def write_fp_x_flux_zarr(
    result: xr.DataArray,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    consolidated: bool = True,
    compressor=DEFAULT_ZARR_COMPRESSOR,
) -> Path:
    """Write one fp_x_flux output via a temporary sibling directory."""
    target = Path(output_path).expanduser()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {target}")

    tmp = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    target.parent.mkdir(parents=True, exist_ok=True)
    ds = _zarr_safe_string_coords(result.to_dataset())
    print(f"Writing fp_x_flux temporary Zarr: {tmp}")
    try:
        ds.to_zarr(tmp, mode="w", consolidated=consolidated, encoding=zarr_encoding(ds, compressor=compressor))
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise

    if target.exists():
        print(f"Removing existing fp_x_flux Zarr before overwrite: {target}")
        shutil.rmtree(target)

    print(f"Moving fp_x_flux Zarr into place: {target}")
    shutil.move(str(tmp), str(target))
    return target


def write_manifest(run: ForwardModelRun, manifest_path: str | Path | None = None) -> Path:
    """Write a JSON manifest for one site/month output."""
    output = Path(run.output_path)
    path = Path(manifest_path) if manifest_path is not None else output.with_suffix(".manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(run), indent=2, sort_keys=True) + "\n")
    return path


def run_site_month(  # noqa: PLR0913
    *,
    site: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    flux_zarr_path: str | Path,
    output_dir: str | Path,
    inlet: str | None = None,
    footprint_domain: str = "europe",
    footprint_store: str = "shared_store_zarr",
    footprint_species: str = "co2",
    species: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    sectors: Sequence[str] | None = None,
    time_chunk: int = 24,
    source_chunk: int = 4,
    fillna_zero: bool = False,
    skip_existing: bool = True,
    overwrite: bool = False,
) -> ForwardModelRun:
    """Run one site/month and write a resumable fp_x_flux Zarr output."""
    site_upper = site.upper()
    target = fp_x_flux_output_path(output_dir, site=site_upper, start_date=start_date)
    if target.exists() and skip_existing and not overwrite:
        print(f"Skipping existing fp_x_flux output: {target}")
        return ForwardModelRun(
            site=site_upper,
            start_date=str(pd.Timestamp(start_date).date()),
            end_date=str(pd.Timestamp(end_date).date()),
            output_path=str(target),
            flux_zarr_path=str(flux_zarr_path),
            footprint_store=footprint_store,
            footprint_domain=footprint_domain,
            footprint_species=footprint_species,
            inlet=inlet,
            time_chunk=time_chunk,
            source_chunk=source_chunk,
            fillna_zero=fillna_zero,
            created_at=datetime.now(UTC).isoformat(),
            status="skipped_existing",
        )

    print(f"Running fp_x_flux for {site_upper} {pd.Timestamp(start_date):%Y-%m}")
    fp = get_month_footprint(
        site=site_upper,
        start_date=start_date,
        end_date=end_date,
        domain=footprint_domain,
        store=footprint_store,
        species=footprint_species,
        inlet=inlet,
    )
    flux = open_staged_flux(flux_zarr_path)
    flux = select_flux_sources(flux, species=species, scenarios=scenarios, sectors=sectors)
    result = compute_fp_x_flux(
        fp,
        flux,
        time_chunk=time_chunk,
        source_chunk=source_chunk,
        fillna_zero=fillna_zero,
    )
    result = result.assign_attrs(
        {
            "site": site_upper,
            "start_date": str(pd.Timestamp(start_date).date()),
            "end_date": str(pd.Timestamp(end_date).date()),
            "footprint_store": footprint_store,
            "footprint_domain": footprint_domain,
            "footprint_species": footprint_species,
            "inlet": "" if inlet is None else inlet,
        }
    )
    write_fp_x_flux_zarr(result, target, overwrite=overwrite)
    run = ForwardModelRun(
        site=site_upper,
        start_date=str(pd.Timestamp(start_date).date()),
        end_date=str(pd.Timestamp(end_date).date()),
        output_path=str(target),
        flux_zarr_path=str(flux_zarr_path),
        footprint_store=footprint_store,
        footprint_domain=footprint_domain,
        footprint_species=footprint_species,
        inlet=inlet,
        time_chunk=time_chunk,
        source_chunk=source_chunk,
        fillna_zero=fillna_zero,
        created_at=datetime.now(UTC).isoformat(),
        status="written",
    )
    write_manifest(run)
    return run


def iter_site_month_runs(
    *,
    sites: Iterable[str] = AVAILABLE_SHARED_STORE_SITES,
    start: str | pd.Timestamp = "2021-01-01",
    end: str | pd.Timestamp = "2022-01-01",
) -> Iterable[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Yield the default 31-site, 12-month production run keys."""
    for site in sites:
        for month_start, month_end in month_windows(start=start, end=end):
            yield site, month_start, month_end
