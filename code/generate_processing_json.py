"""Utilities to build an aind-data-schema ``processing.json`` for camera alignment.

These helpers construct a ``Processing`` document that captures the
``IMAGE_CROSS_IMAGE_ALIGNMENT`` step produced by this package and registers the
pipeline (``aind-Z1-pipeline-1.0.0-production``) that produced it. Resource usage
is intentionally omitted; only the core process metadata and optional parameters
are recorded. Designed for programmatic use (no CLI).
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
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
