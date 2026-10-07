"""Utilities to build an aind-data-schema ``processing.json`` for camera alignment.

These helpers construct a ``Processing`` document that captures the
``IMAGE_CROSS_IMAGE_ALIGNMENT`` step produced by this package and registers the
pipeline (``aind-Z1-pipeline-1.0.0-production``) that produced it. Resource usage
is intentionally omitted; only the core process metadata and optional parameters
are recorded. Designed for programmatic use (no CLI).
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from __init__ import __maintainers__, __pipeline_version__, __version__, __url__

try:
    from aind_data_schema.components.identifiers import Code, Database, CombinedData, DataAsset
    from aind_data_schema.core.processing import (
        DataProcess,
        Processing,
        ProcessName,
        ProcessStage,
    )
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "aind-data-schema is required to generate processing.json. Install with\n"
        "  pip install aind-data-schema aind-data-schema-models\n"
        "and re-run this script."
    ) from exc


LOGGER = logging.getLogger(__name__)


def _git_metadata() -> Dict[str, Optional[str]]:
    """Return git commit hash and remote URL if available.

    Returns
    -------
    Dict[str, Optional[str]]
        Mapping with keys ``"commit"`` and ``"repo_url"``. Values are ``None`` if
        the information cannot be retrieved (e.g., not a git checkout).
    """
    
    # Hardcoded metadata for release builds (edit as needed).
    CODE_URL = __url__
    CODE_VERSION = __version__

    metadata: Dict[str, Optional[str]] = {"repo_url": CODE_URL, "commit": CODE_VERSION}
    try:
        metadata["commit"] = (
            subprocess.check_output([
                "git",
                "rev-parse",
                "HEAD",
            ], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        pass

    try:
        metadata["repo_url"] = (
            subprocess.check_output(
                [
                    "git",
                    "config",
                    "--get",
                    "remote.origin.url",
                ], text=True, stderr=subprocess.DEVNULL
            )
            .strip()
        )
    except Exception:
        pass

    return metadata


def build_processing_document(
    output_path: Path,
    dataset_name: Optional[str] = None,
    experimenters: Optional[List[str]] = None,
    parameters: Optional[Dict[str, object]] = None,
    pipeline_code_name: str = "aind-Z1-pipeline-1.0.0-production",
    pipeline_code_url: str = "https://codeocean.allenneuraldynamics.org/capsule/6036323/tree",
    pipeline_code_version: str = __pipeline_version__,
) -> Processing:
    """Construct a ``Processing`` document for image cross-image alignment.

    Parameters
    ----------
    output_path : Path
        Logical output location for the process (recorded in the schema).
    pipeline_name : str, default "Z1 camera alignment"
        Pipeline name to embed in the document.
    dataset_name : str, optional
        Dataset identifier to embed in the input data asset.
    experimenters : list of str, optional
        Names of experimenters to record; defaults to empty list.
    parameters : dict, optional
        Arbitrary parameter map to embed in the ``Code`` component.
    pipeline_code_name : str, default "aind-Z1-pipeline-1.0.0-production"
        Name of the pipeline that produced this process.
    pipeline_code_url : str
        URL for the pipeline capsule or source.
    pipeline_code_version : str
        Version string for the pipeline code.

    Returns
    -------
    Processing
        Fully populated ``Processing`` model with a single ``DataProcess`` entry
        of type ``IMAGE_CROSS_IMAGE_ALIGNMENT`` and a pipeline graph.
    """
    t = datetime.now(timezone.utc)
    meta = _git_metadata()

    input_data = None
    # if dataset_name:
    #     input_data = [
    #         CombinedData(
    #             assets=[
    #                 DataAsset(
    #                     url=f"s3://aind-open-data/{dataset_name}/image_radial_correction/",
    #                 )
    #             ],
    #             name="input_data",
    #             description="Input data for camera alignment",
    #         )
    #     ]

    code = Code(
        name="aind-z1-camera-alignment",
        version=meta.get("commit") or "unknown",
        url=meta.get("repo_url"),
        language="python",
        language_version="3.12.4",
        input_data=input_data,
        container=None, #not registered in docker hub yet
        parameters=parameters or {},
    )

    pipeline_code = Code(
        name=pipeline_code_name,
        version=pipeline_code_version,
        url=pipeline_code_url,
    )

    data_process = DataProcess(
        process_type=ProcessName.IMAGE_CROSS_IMAGE_ALIGNMENT,
        experimenters=experimenters or [],
        stage=ProcessStage.PROCESSING,
        start_date_time=t,
        end_date_time=t,
        output_path=str(output_path),
        pipeline_name=pipeline_code_name,
        code=code,
        resources=None,  # Intentionally omitted per request
    )
    return Processing.create_with_sequential_process_graph(
        pipelines=[pipeline_code],
        data_processes=[data_process],
    )


def write_processing_json(processing: Processing, destination: Path) -> None:
    """Write a ``Processing`` document to disk as JSON.

    Parameters
    ----------
    processing : Processing
        The document to serialize.
    destination : Path
        Target file path; parent directories are created if missing.
    """
    serialized = processing.model_dump_json(indent=2)
    Processing.model_validate_json(serialized)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized)


def _is_v2_processing(existing: Dict[str, Any]) -> bool:
    """Heuristically determine whether a processing dict uses the v2 schema.

    Parameters
    ----------
    existing : dict
        Parsed processing.json content.

    Returns
    -------
    bool
        True if the document appears to follow the aind-data-schema v2 layout
        (top-level ``data_processes``), False if it looks like the v1 layout
        (nested ``processing_pipeline.data_processes``).
    """
    if "data_processes" in existing:
        return True
    if "processing_pipeline" in existing:
        return False
    schema_version = str(existing.get("schema_version", ""))
    return schema_version.startswith("2")


def merge_processing(
    existing: Dict[str, Any],
    new_processing: Processing,
) -> Dict[str, Any]:
    """Append the new document's data processes to an existing processing dict.

    The existing pipelines, notes, and other metadata are preserved. When the
    existing document uses the same v2 schema version it is merged with the
    schema's native ``Processing.__add__`` operator (which de-duplicates process
    names and links dependency graphs). Otherwise the new data processes are
    appended to the raw list in place (v1 ``processing_pipeline.data_processes``
    or an incompatible v2 version), leaving existing content untouched.

    Parameters
    ----------
    existing : dict
        Parsed processing.json already present in the dataset.
    new_processing : Processing
        Freshly built document whose data processes should be appended.

    Returns
    -------
    dict
        The merged processing document as a JSON-serializable dict.
    """
    new_dict = new_processing.model_dump(mode="json")

    if _is_v2_processing(existing):
        existing_version = existing.get("schema_version")
        if existing_version == new_processing.schema_version:
            try:
                existing_model = Processing.model_validate(existing)
                combined = existing_model + new_processing
                return combined.model_dump(mode="json")
            except Exception as exc:  # pragma: no cover - defensive fallback
                LOGGER.warning(
                    "Failed to merge via Processing model (%s); "
                    "falling back to raw append.",
                    exc,
                )
        else:
            LOGGER.warning(
                "Existing processing schema_version %s differs from new %s; "
                "appending to data_processes without model validation.",
                existing_version,
                new_processing.schema_version,
            )

        merged = dict(existing)
        merged["data_processes"] = list(existing.get("data_processes") or []) + list(
            new_dict.get("data_processes") or []
        )
        merged_pipelines = list(existing.get("pipelines") or [])
        for pipeline in new_dict.get("pipelines") or []:
            if pipeline not in merged_pipelines:
                merged_pipelines.append(pipeline)
        if merged_pipelines:
            merged["pipelines"] = merged_pipelines
        return merged

    # v1 layout: append to the nested processing_pipeline.data_processes list.
    LOGGER.warning(
        "Existing processing.json appears to use the v1 schema; appending the "
        "v2 data process(es) to processing_pipeline.data_processes. The merged "
        "document will contain mixed schema versions."
    )
    merged = dict(existing)
    pipeline = dict(merged.get("processing_pipeline") or {})
    pipeline["data_processes"] = list(pipeline.get("data_processes") or []) + list(
        new_dict.get("data_processes") or []
    )
    merged["processing_pipeline"] = pipeline
    return merged


def generate_processing_json(
    output_dir: Path = Path("/results"),
    filename: str = "processing.json",
    dataset_name: Optional[str] = None,
    experimenters: Optional[List[str]] = None,
    parameters: Optional[Dict[str, object]] = None,
    pipeline_code_name: str = "aind-Z1-pipeline-1.0.0-production",
    pipeline_code_url: str = "https://codeocean.allenneuraldynamics.org/capsule/6036323/tree",
    pipeline_code_version: str = "1.0.0-production",
) -> Path:
    """Build and write ``processing.json`` for the alignment step.

    Parameters
    ----------
    output_dir : Path, default ``/results``
        Directory where the JSON will be written and the logical output path recorded
        in the ``DataProcess``.
    filename : str, default ``processing.json``
        File name for the JSON artifact inside ``output_dir``.
    pipeline_name : str, default "Z1 camera alignment"
        Human-friendly pipeline name to record.
    dataset_name : str, optional
        Dataset identifier to embed in the input data asset.
    experimenters : list of str, optional
        Names to attach to the process; defaults to an empty list.
    parameters : dict, optional
        Arbitrary parameter map to embed in the ``Code`` component.
    pipeline_code_name : str, default "aind-Z1-pipeline-1.0.0-production"
        Name of the pipeline producing this process.
    pipeline_code_url : str
        URL of the pipeline capsule/source.
    pipeline_code_version : str
        Version string for the pipeline code.

    Returns
    -------
    Path
        The destination path that was written.
    """
    destination = output_dir / filename
    processing = build_processing_document(
        output_path=output_dir,
        dataset_name=dataset_name,
        experimenters=experimenters,
        parameters=parameters,
        pipeline_code_name=pipeline_code_name,
        pipeline_code_url=pipeline_code_url,
        pipeline_code_version=pipeline_code_version,
    )
    write_processing_json(processing, destination)
    return destination


def update_processing_json_on_s3(
    dataset_name: str,
    output_dir: Path = Path("/results"),
    filename: str = "processing.json",
    bucket: str = "aind-open-data",
    experimenters: Optional[List[str]] = None,
    parameters: Optional[Dict[str, object]] = None,
    pipeline_code_name: str = "aind-Z1-pipeline-1.0.0-production",
    pipeline_code_url: str = "https://codeocean.allenneuraldynamics.org/capsule/6036323/tree",
    pipeline_code_version: str = __pipeline_version__,
) -> str:
    """Append this pipeline's data process to the dataset's S3 processing.json.

    Reads the existing ``processing.json`` from
    ``s3://{bucket}/{dataset_name}/processing.json`` (if present), appends the
    newly built camera-alignment data process to its ``data_processes`` list
    (preserving existing pipelines and metadata), writes the merged document
    locally to ``output_dir/filename``, and uploads it back to the same S3 key.

    If no existing document is found, the freshly built document is written as-is.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier used to locate the S3 processing.json.
    output_dir : Path, default ``/results``
        Directory where the merged JSON is written locally before upload.
    filename : str, default ``processing.json``
        File name for the local and S3 artifact.
    bucket : str, default ``aind-open-data``
        S3 bucket hosting the dataset.
    experimenters : list of str, optional
        Names to attach to the new process.
    parameters : dict, optional
        Parameter map to embed in the ``Code`` component.
    pipeline_code_name : str
        Name of the pipeline producing this process.
    pipeline_code_url : str
        URL of the pipeline capsule/source.
    pipeline_code_version : str
        Version string for the pipeline code.

    Returns
    -------
    str
        The S3 URI that was written.
    """
    # Imported lazily so building a document locally does not require s3fs.
    from s3_writer import read_json_from_s3, write_json_to_s3

    s3_key = f"{bucket}/{dataset_name}/{filename}"
    s3_uri = f"s3://{s3_key}"

    new_processing = build_processing_document(
        output_path=output_dir,
        dataset_name=dataset_name,
        experimenters=experimenters,
        parameters=parameters,
        pipeline_code_name=pipeline_code_name,
        pipeline_code_url=pipeline_code_url,
        pipeline_code_version=pipeline_code_version,
    )

    existing = read_json_from_s3(s3_uri)
    if existing:
        merged = merge_processing(existing, new_processing)
    else:
        LOGGER.info("No existing processing.json at %s; writing new document.", s3_uri)
        merged = new_processing.model_dump(mode="json")

    destination = output_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(merged, indent=2))

    write_json_to_s3(merged, s3_uri)
    return s3_uri

