"""Stage PARIS verification-games fluxes into a source-aware Zarr store."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
from numcodecs import Blosc
from ogcat import ArtifactLocator, Catalog
import pandas as pd
import xarray as xr

from verification_games.metadata import append_history
from verification_games.units import cf_ureg


DEFAULT_VG_PATH = Path("/group/chem/acrg/verification_games_round_2")
DEFAULT_CATALOG_PATH = DEFAULT_VG_PATH / "games_catalog"
DEFAULT_SCENARIOS = ("ATEN", "BASE", "DFIN", "HFRA", "HGER", "PTEN")
DEFAULT_SPECIES = ("co2", "o2")
DEFAULT_SECTORS = ("GPP", "TER", "FF", "ocean")
DEFAULT_FLUX_CHUNKS = {"time": 24, "lat": 293, "lon": 391, "source": 4}
DEFAULT_STAGE_KEYWORDS = ("flux_stage", "filled_nan_zero", "forward_model_input")
DEFAULT_FLUX_UNITS = "mol m-2 s-1"
DEFAULT_PINT_FLUX_UNITS = "mol/m2/s"
DEFAULT_ZARR_COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
STAGED_FLUX_MODIFICATIONS = (
    "Stacked PARIS species/scenario/sector fluxes into a single source dimension, "
    "converted flux variables to mol m-2 s-1 where necessary, filled flux NaNs "
    "with zero, and persisted as chunked Zarr for forward modelling."
)


@dataclass(frozen=True)
class FluxDatasetInput:
    """One species/scenario flux dataset and its catalog provenance."""

    species: str
    games_scenario: str
    record_id: str
    path: str
    dataset: xr.Dataset


def source_label(species: str, games_scenario: str, sector: str) -> str:
    """Return a stable source coordinate label."""
    return f"{species.lower()}_{games_scenario.upper()}_{sector}"


def zarr_compressor_metadata(compressor=DEFAULT_ZARR_COMPRESSOR) -> dict[str, object]:
    """Return a JSON-friendly description of a Zarr compressor."""
    if hasattr(compressor, "get_config"):
        return dict(compressor.get_config())
    return {"repr": repr(compressor)}


def zarr_encoding(
    ds: xr.Dataset,
    *,
    compressor=DEFAULT_ZARR_COMPRESSOR,
) -> dict[str, dict[str, object]]:
    """Return Zarr encoding for data variables."""
    return {name: {"compressor": compressor} for name in ds.data_vars}


def open_games_catalog(catalog_path: str | Path = DEFAULT_CATALOG_PATH):
    """Open the verification games ogcat catalog."""
    return Catalog.open(Path(catalog_path))


def find_flux_records(
    catalog,
    *,
    species: Sequence[str] = DEFAULT_SPECIES,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    university_abbr: str = "UOB",
    record_type: str = "verification_games_flux",
) -> list:
    """Find the 12 source NetCDF records used to build the staged flux Zarr."""
    records = []
    for sp in species:
        for scenario in scenarios:
            result = catalog.search(
                where={
                    "record_type": record_type,
                    "species": sp,
                    "games_scenario": scenario,
                    "university_abbr": university_abbr,
                },
                as_record_set=True,
            )
            if len(result) != 1:
                raise ValueError(
                    "Expected exactly one flux record for "
                    f"species={sp!r}, scenario={scenario!r}; found {len(result)}."
                )
            records.append(result[0])
    return records


def open_flux_inputs(
    records: Iterable,
    *,
    input_chunks: Mapping[str, int] | None = None,
) -> list[FluxDatasetInput]:
    """Open catalog flux records lazily as xarray datasets."""
    chunks = dict(input_chunks or {key: value for key, value in DEFAULT_FLUX_CHUNKS.items() if key != "source"})
    inputs: list[FluxDatasetInput] = []

    for record in records:
        species = str(record.user_metadata["species"]).lower()
        scenario = str(record.user_metadata["games_scenario"]).upper()
        path = str(record.locator.value)
        print(f"Opening flux record {record.id}: {species} {scenario} -> {path}")
        ds = xr.open_dataset(path, chunks=chunks)
        inputs.append(
            FluxDatasetInput(
                species=species,
                games_scenario=scenario,
                record_id=str(record.id),
                path=path,
                dataset=ds,
            )
        )

    return inputs


def _is_target_flux_units(units: str, target_units: str = DEFAULT_FLUX_UNITS) -> bool:
    """Return whether source units are already exactly in the target unit scale."""
    source = cf_ureg.parse_expression(units)
    target = cf_ureg.parse_expression(target_units)
    converted = source.to(target.units)
    return bool(np.isclose(float(converted.magnitude), float(target.magnitude)))


def convert_flux_units(
    data: xr.DataArray,
    *,
    target_units: str = DEFAULT_FLUX_UNITS,
    pint_target_units: str = DEFAULT_PINT_FLUX_UNITS,
) -> xr.DataArray:
    """Convert a flux DataArray to the target units using pint-xarray if needed."""
    source_units = data.attrs.get("units")
    if not source_units:
        raise ValueError(f"Flux variable {data.name!r} has no 'units' attribute.")

    if _is_target_flux_units(str(source_units), target_units):
        out = data
    else:
        quantified = data.pint.quantify(unit_registry=cf_ureg)
        with xr.set_options(keep_attrs=True):
            out = quantified.pint.to(pint_target_units).pint.dequantify()

    out.attrs = dict(data.attrs)
    out.attrs["units"] = target_units
    out.attrs["source_units_before_staging"] = str(source_units)
    return out


def stack_flux_sources(
    inputs: Sequence[FluxDatasetInput],
    *,
    sectors: Sequence[str] = DEFAULT_SECTORS,
    output_chunks: Mapping[str, int] | None = None,
    fill_value: float | None = 0.0,
) -> xr.Dataset:
    """Stack species/scenario/sector fluxes into one ``flux`` variable."""
    arrays: list[xr.DataArray] = []
    source_labels: list[str] = []
    source_species: list[str] = []
    source_scenarios: list[str] = []
    source_sectors: list[str] = []
    input_records: list[str] = []
    input_paths: list[str] = []

    for item in inputs:
        input_records.append(item.record_id)
        input_paths.append(item.path)
        for sector in sectors:
            if sector not in item.dataset:
                raise KeyError(f"{item.path} is missing expected sector variable {sector!r}")
            label = source_label(item.species, item.games_scenario, sector)
            print(f"Stacking source {label}")
            da = convert_flux_units(item.dataset[sector], target_units=DEFAULT_FLUX_UNITS).astype(np.float32)
            if fill_value is not None:
                da = da.fillna(fill_value)
            arrays.append(da)
            source_labels.append(label)
            source_species.append(item.species)
            source_scenarios.append(item.games_scenario)
            source_sectors.append(sector)

    flux = xr.concat(arrays, dim=pd.Index(source_labels, name="source"))
    flux = flux.transpose("time", "lat", "lon", "source")
    flux = flux.assign_coords(
        species=("source", source_species),
        games_scenario=("source", source_scenarios),
        sector=("source", source_sectors),
    )
    chunks = dict(output_chunks or DEFAULT_FLUX_CHUNKS)
    flux = flux.chunk({dim: chunks[dim] for dim in chunks if dim in flux.dims})
    flux.name = "flux"
    flux.attrs.update(
        {
            "description": "PARIS verification-games fluxes stacked by species, scenario and sector.",
            "units": DEFAULT_FLUX_UNITS,
            "nan_policy": _nan_policy_text(fill_value),
        }
    )

    out = flux.to_dataset()
    out.attrs.update(
        {
            "title": "PARIS verification-games staged fluxes",
            "source_record_ids": ",".join(input_records),
            "source_paths": "\n".join(input_paths),
            "units": DEFAULT_FLUX_UNITS,
            "nan_policy": _nan_policy_text(fill_value),
            "modifications": STAGED_FLUX_MODIFICATIONS,
        }
    )
    out.attrs = append_history(out.attrs, STAGED_FLUX_MODIFICATIONS)
    return out


def build_staged_flux_dataset(  # noqa: PLR0913
    catalog,
    *,
    species: Sequence[str] = DEFAULT_SPECIES,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    sectors: Sequence[str] = DEFAULT_SECTORS,
    university_abbr: str = "UOB",
    input_chunks: Mapping[str, int] | None = None,
    output_chunks: Mapping[str, int] | None = None,
    fill_value: float | None = 0.0,
) -> tuple[xr.Dataset, list]:
    """Build the lazy staged-flux dataset and return it with input records."""
    records = find_flux_records(
        catalog,
        species=species,
        scenarios=scenarios,
        university_abbr=university_abbr,
    )
    inputs = open_flux_inputs(records, input_chunks=input_chunks)
    ds = stack_flux_sources(inputs, sectors=sectors, output_chunks=output_chunks, fill_value=fill_value)
    return ds, records


def write_staged_flux_zarr(  # noqa: PLR0913
    ds: xr.Dataset,
    target_path: str | Path,
    *,
    temp_parent: str | Path | None = None,
    overwrite: bool = False,
    consolidated: bool = True,
    compressor=DEFAULT_ZARR_COMPRESSOR,
) -> Path:
    """Write staged flux to a temporary Zarr directory, then move into place."""
    target = Path(target_path).expanduser()
    temp_root = Path(temp_parent).expanduser() if temp_parent else target.parent
    temp_root.mkdir(parents=True, exist_ok=True)
    tmp = temp_root / f".{target.name}.{uuid4().hex}.tmp"

    if target.exists() and not overwrite:
        raise FileExistsError(f"Target Zarr already exists: {target}")

    print(f"Writing staged flux Zarr to temporary path: {tmp}")
    ds.to_zarr(tmp, mode="w", consolidated=consolidated, encoding=zarr_encoding(ds, compressor=compressor))
    validation = validate_staged_flux_zarr(tmp, full_nan_check=False)
    print(f"Temporary staged flux validation: {validation}")

    if target.exists():
        print(f"Removing existing target Zarr before overwrite: {target}")
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Moving staged flux Zarr into place: {target}")
    shutil.move(str(tmp), str(target))
    return target


def stage_flux_zarr(  # noqa: PLR0913
    catalog,
    target_path: str | Path,
    *,
    temp_parent: str | Path | None = None,
    overwrite: bool = False,
    output_chunks: Mapping[str, int] | None = None,
    fill_value: float | None = 0.0,
    compressor=DEFAULT_ZARR_COMPRESSOR,
) -> tuple[Path, list]:
    """Build and write the all-source staged flux Zarr."""
    ds, records = build_staged_flux_dataset(catalog, output_chunks=output_chunks, fill_value=fill_value)
    path = write_staged_flux_zarr(
        ds,
        target_path,
        temp_parent=temp_parent,
        overwrite=overwrite,
        compressor=compressor,
    )
    return path, records


def _nan_policy_text(fill_value: float | None) -> str:
    if fill_value is None:
        return "NaNs preserved."
    return f"NaNs filled with {fill_value:g} during Zarr staging."


def validate_staged_flux_zarr(
    zarr_path: str | Path,
    *,
    expected_source_count: int = 48,
    expected_chunks: Mapping[str, int] = DEFAULT_FLUX_CHUNKS,
    full_nan_check: bool = False,
) -> dict[str, object]:
    """Validate staged Zarr structure, with optional full missing-data scan."""
    path = Path(zarr_path)
    ds = xr.open_zarr(path)
    if "flux" not in ds:
        raise ValueError(f"{path} does not contain a 'flux' variable")

    flux = ds["flux"]
    if tuple(flux.dims) != ("time", "lat", "lon", "source"):
        raise ValueError(f"Unexpected flux dims: {flux.dims}")
    if flux.sizes["source"] != expected_source_count:
        raise ValueError(f"Expected {expected_source_count} sources, found {flux.sizes['source']}")

    chunk_summary = {dim: tuple(int(c) for c in chunks) for dim, chunks in flux.chunksizes.items()}
    for dim, expected in expected_chunks.items():
        if dim in chunk_summary and max(chunk_summary[dim]) != expected:
            raise ValueError(f"Unexpected max chunk size for {dim}: {chunk_summary[dim]} != {expected}")

    summary: dict[str, object] = {
        "path": str(path),
        "sizes": dict(flux.sizes),
        "chunks": chunk_summary,
        "sources": [str(value) for value in flux["source"].values],
    }
    if full_nan_check:
        print("Computing full staged-flux NaN count")
        summary["nan_count"] = int(flux.isnull().sum().compute())  # noqa: PD003
    return summary


def _relative_path_if_inside(path: Path, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def register_staged_flux_zarr(  # noqa: PLR0913
    catalog,
    zarr_path: str | Path,
    input_records: Sequence,
    *,
    product: str = "PARIS_CTE-HR_filled_fluxes_staged",
    version: str = "zarr_v1",
    year: str = "202012-202112",
    university_abbr: str = "UOB",
    keywords: Sequence[str] = DEFAULT_STAGE_KEYWORDS,
) -> object:
    """Register a staged Zarr directory in ogcat without writing through ogcat."""
    path = Path(zarr_path).expanduser().resolve()
    relative_path = _relative_path_if_inside(path, Path(catalog.root))
    locator = ArtifactLocator.from_path(path, relative_path=relative_path)
    input_ids = [str(record.id) for record in input_records]
    input_paths = [str(record.locator.value) for record in input_records]
    validation = validate_staged_flux_zarr(path, full_nan_check=False)

    metadata = {
        "title": "PARIS verification-games staged flux Zarr for forward modelling.",
        "product": product,
        "version": version,
        "species": list(DEFAULT_SPECIES),
        "sector": list(DEFAULT_SECTORS),
        "domain": "EUROPE",
        "provenance": "derived",
        "format": "zarr",
        "data_format": "zarr",
        "year": year,
        "keywords": list(keywords),
        "inputs": input_ids,
        "university_abbr": university_abbr,
        "modifications": "Stacked species/scenario/sector fluxes into a source dimension and filled flux NaNs with zero.",
    }
    derived_metadata = {
        "reader_hint": "xarray.open_zarr",
        "output_variables": ["flux"],
        "sizes": validation["sizes"],
        "chunks": {dim: max(chunks) for dim, chunks in validation["chunks"].items()},
        "source_count": len(validation["sources"]),
        "sources": validation["sources"],
        "input_record_ids": input_ids,
        "source_paths": input_paths,
        "nan_policy": "Flux NaNs filled with 0.0 during staging.",
        "units": DEFAULT_FLUX_UNITS,
        "zarr_compressor": zarr_compressor_metadata(),
        "modifications": STAGED_FLUX_MODIFICATIONS,
    }
    naming_metadata = {
        "target_kind": "directory",
        "storage_relative_path": relative_path,
        "resolved_directory": str(path.parent),
        "resolved_filename": path.name,
    }
    print(f"Registering staged flux Zarr reference in ogcat: {path}")
    return catalog.add_artifact(
        record_type="derived_flux",
        locator=locator,
        metadata=metadata,
        storage_mode="reference",
        original_path=str(path),
        original_filename=path.name,
        suffixes=[".zarr"],
        derived_metadata=derived_metadata,
        naming_metadata=naming_metadata,
    )
