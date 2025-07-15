import pathlib
import logging
from cam_affine_cuda import main as run_2d_camera_correction
import os
from typing import List, Dict, Any
import json
import s3fs


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_zarr_datasets() -> List[pathlib.Path]:
    """
    Find all zarr datasets in the data directory.
    Returns a list of paths to zarr datasets.
    """
    data_dir = pathlib.Path("/data")
    
    # Look for zarr files directly in data directory and one level deep
    zarr_datasets = []
    
    # Direct zarr files
    #zarr_datasets.extend(list(data_dir.glob('*.zarr')))
    zarr_datasets.extend(list(data_dir.glob('*.ome.zarr')))
    
    # Check one level deep
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            #zarr_datasets.extend(list(subdir.glob('*.zarr')))
            zarr_datasets.extend(list(subdir.glob('*.ome.zarr')))
    
    logger.info(f"Found zarr datasets: {zarr_datasets}")
    return zarr_datasets

def load_data_description() -> str:
    """
    Load the data_description.json file from the data directory and extract dataset name.
    
    Returns
    -------
    str
        The dataset name extracted from the 'name' field in data_description.json
        
    Raises
    ------
    FileNotFoundError
        If no data_description.json file is found in /data directory
    RuntimeError
        If error occurs while loading or parsing the JSON configuration
    """
    data_dir = pathlib.Path("/data")
    json_file = data_dir / "data_description.json"
    
    if not json_file.exists():
        raise FileNotFoundError("No data_description.json file found in /data directory")
    
    logger.info(f"Loading configuration from {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            config = json.load(f)
            dataset_name = config.get('name')
            if not dataset_name:
                raise ValueError("'name' field not found in data_description.json")
            logger.info(f"Loaded dataset name: {dataset_name}")
            return dataset_name
    except Exception as e:
        raise RuntimeError(f"Error loading data_description.json: {str(e)}")


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
    
    try:
        # Run 2D camera correction
        run_2d_camera_correction(args)
        
        # Record successful processing
        with open(results_dir / "processing_complete.txt", "w") as f:
            f.write(f"Successfully processed {dataset_name}\n")
            f.write(f"S3 zarr path: {s3_path}\n")
            f.write(f"Number of tiles processed: {len(zarr_tiles)}\n\n")
        
        logger.info(f"Successfully completed processing for {dataset_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error during camera correction processing: {str(e)}")
        return False

def list_zarr_tiles_from_s3(s3_path: str) -> List[str]:
    """
    List all zarr tile files from the specified S3 path.
    
    Parameters
    ----------
    s3_path : str
        The S3 path in format 's3://bucket/prefix/' to search for zarr tiles
        
    Returns
    -------
    List[str]
        List of S3 paths to zarr tile files
        
    Raises
    ------
    RuntimeError
        If unable to connect to S3 or list files
    """
    try:
        # Initialize S3 filesystem
        s3 = s3fs.S3FileSystem(anon=False)
        
        # Remove 's3://' prefix for s3fs
        s3_path_clean = s3_path.replace('s3://', '')
        if not s3_path_clean.endswith('/'):
            s3_path_clean += '/'
            
        logger.info(f"Listing zarr tiles from: {s3_path}")
        
        # List all files with .zarr extension
        zarr_files = []
        try:
            all_files = s3.glob(f"{s3_path_clean}*.zarr")
            zarr_files = [f"s3://{file}" for file in all_files]
        except Exception as e:
            logger.warning(f"No zarr files found at {s3_path}: {e}")
            return []
        
        logger.info(f"Found {len(zarr_files)} zarr tile files")
        return zarr_files
        
    except Exception as e:
        raise RuntimeError(f"Error listing zarr tiles from S3: {str(e)}")

if __name__ == "__main__":
    try:
        process_zarr_datasets()
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise