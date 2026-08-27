import argparse
import pathlib
import logging
import os
import sys
from typing import List, Dict, Any

from cam_affine_cuda import main as run_2d_camera_correction
from generate_processing_json import generate_processing_json
import camera_alignment_qc
from utils import (
    load_data_description,
    list_zarr_tiles_from_s3,
    get_project_name
) 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_zarr_datasets():
    """
    Process zarr datasets by loading data from S3 based on configuration from data_description.json.
    
    Returns
    -------
    bool
        True if processing was successful, False otherwise
    """
    results_dir = pathlib.Path("/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset name from data_description.json
    try:
        dataset_name = load_data_description()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Failed to load dataset configuration: {e}")
        return False

    # Construct S3 path for zarr tiles
    s3_path = f"s3://aind-open-data/{dataset_name}/image_radial_correction/"
    
    # List zarr tiles from S3
    try:
        zarr_tiles = list_zarr_tiles_from_s3(s3_path)
        if not zarr_tiles:
            logger.error(f"No zarr tiles found at {s3_path}")
            return False
    except RuntimeError as e:
        logger.error(f"Failed to list zarr tiles: {e}")
        return False
    
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"Found {len(zarr_tiles)} zarr tiles to process")
        
    # Get project name to determine if we should skip alignment
    project_name = get_project_name()
    skip_alignment = False #(project_name == "PLACE")
    
    # Set up arguments for the correction function
    args = {
        "dataset_name": dataset_name,
        "bucket": "aind-open-data",
        "s3_bucket": "aind-open-data",
        "z_correct": False,  # Default to 2D correction
        "pipeline": True,  # Always use pipeline mode for S3 data
        "s3_zarr_path": s3_path,  # Pass S3 path for zarr tiles
        "skip_alignment": skip_alignment,  # Skip alignment for PLACE projects
    }
    
    if skip_alignment:
        logger.info(f"Skipping camera alignment for project: {project_name}")
    else:
        logger.info(f"Running camera alignment for project: {project_name}")
        
    # Run camera correction (will handle skipping internally)
    run_2d_camera_correction(args)
    
    # Record successful processing
    with open(results_dir / "processing_complete.txt", "w") as f:
        f.write(f"Successfully processed {dataset_name}\n")
        f.write(f"S3 zarr path: {s3_path}\n")
        f.write(f"Number of tiles processed: {len(zarr_tiles)}\n\n")

    # Emit processing.json for cross-image alignment
    try:
        processing_path = generate_processing_json(
            output_dir=results_dir,
            dataset_name=dataset_name,
            parameters =args,
        )
        logger.info(f"Wrote processing.json to {processing_path}")
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.warning("Failed to write processing.json: %s", exc)
    
    logger.info(f"Successfully completed processing for {dataset_name}")
    return True
        



def run_camera_alignment_qc_only() -> bool:
    """Execute QC plot generation without running alignment."""
    results_root = pathlib.Path("/results")
    results_root.mkdir(parents=True, exist_ok=True)
    logger.info("Generating camera alignment QC plots only (no alignment run).")
    try:
        results = camera_alignment_qc.generate_camera_alignment_qc()

        if results == {}: 
            status_file = results_root / "qc_status.txt"
            with open(status_file, "w") as f:
                f.write("Camera alignment QC was not generated.\n")
                f.write("Reason: Less than 2 channels available after excluding CH_405.\n")
                f.write("This is expected for datasets with only 1-2 channels.\n")
            logger.warning("No QC results generated - insufficient channels")
            return True  # Still return True - this is expected, not a failure

    except Exception as exc:  # noqa: BLE001
        logger.error(f"QC plot generation failed: {exc}")
        logger.debug("QC failure details", exc_info=True)
        status_file = results_root / "qc_status.txt"
        with open(status_file, "w") as f:
            f.write("Camera alignment QC generation failed.\n")
            f.write(f"Error: {exc}\n")
        return False

    logger.info(f"QC plots saved to: {results_root}")
    return True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for capsule execution."""
    parser = argparse.ArgumentParser(description="Run camera alignment capsule targets")
    parser.add_argument(
        "target",
        nargs="?",
        default="camera_alignment",
        choices=("camera_alignment", "camera_alignment_qc"),
        help="Execution target: camera alignment pipeline or QC-only",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()

    try:
        if cli_args.target == "camera_alignment_qc":
            success = run_camera_alignment_qc_only()
        else:
            success = process_zarr_datasets()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error during execution: {exc}")
        logger.debug("Execution failure details", exc_info=True)
