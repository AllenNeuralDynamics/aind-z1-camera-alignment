import pathlib
import logging
from cam_affine_cuda import main as run_2d_camera_correction
import os
from typing import List, Dict, Any
import json
import s3fs
import glob
from utils import (
    load_data_description,
    list_zarr_tiles_from_s3,
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
        
    # Set up arguments for the correction function
    args = {
        "dataset_name": dataset_name,
        "bucket": "aind-open-data",
        "s3_bucket": "aind-open-data",
        "z_correct": False,  # Default to 2D correction
        "pipeline": True,  # Always use pipeline mode for S3 data
        "s3_zarr_path": s3_path,  # Pass S3 path for zarr tiles
    }
    

    # Run 2D camera correction
    run_2d_camera_correction(args)
    
    # Record successful processing
    with open(results_dir / "processing_complete.txt", "w") as f:
        f.write(f"Successfully processed {dataset_name}\n")
        f.write(f"S3 zarr path: {s3_path}\n")
        f.write(f"Number of tiles processed: {len(zarr_tiles)}\n\n")
    
    logger.info(f"Successfully completed processing for {dataset_name}")
    return True
        



if __name__ == "__main__":
    try:
        process_zarr_datasets()
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise