import pathlib
import logging
from cam_affine_cuda import main as run_2d_camera_correction
import os
from typing import List, Dict, Any
import yaml


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

def load_yaml_config() -> Dict[str, Any]:
    """
    Load the YAML configuration file from the data directory.
    Returns a dictionary containing the YAML configuration.
    """
    data_dir = pathlib.Path("/data")
    yaml_files = list(data_dir.glob("*.yml")) + list(data_dir.glob("*.yaml"))
    
    if not yaml_files:
        raise FileNotFoundError("No YAML configuration file found in /data directory")
    
    if len(yaml_files) > 1:
        logger.warning(f"Multiple YAML files found, using {yaml_files[0]}")
    
    yaml_path = yaml_files[0]
    logger.info(f"Loading configuration from {yaml_path}")
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration for {len(config)} datasets")
            return config
    except Exception as e:
        raise RuntimeError(f"Error loading YAML configuration: {str(e)}")


def process_zarr_datasets():
    """
    Process all zarr datasets found in the data directory.
    """
    results_dir = pathlib.Path("/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    #load yaml to get the path of the dataset_name
    yml_config = load_yaml_config()
    if not yml_config:
        raise RuntimeError("Empty or invalid YAML configuration")
    else:
        dataset_name, dataset_info = list(yml_config.items())[0]

    # Find all zarr datasets
    datasets = find_zarr_datasets()
    print(f'datasets {datasets}')

    
    if not datasets:
        logger.error("No zarr datasets found in input directory")
        return False
    
    # Process each dataset
    success = False
    for dataset_path in datasets:
        dataset_path = dataset_path.as_posix()
        logger.info(f"Processing dataset: {dataset_path}")
            
        # Get the parent directory name as the dataset name
        # This assumes the zarr file is in a directory named after the dataset
        #dataset_name = dataset_path.parent.name
        if dataset_path == "/data/radial_correction_temp.ome.zarr":  # If zarr is directly in /data
            #dataset_name = dataset_path.stem
            pipeline_bool = True
        else:
            pipeline_bool = False
        
        # Set up arguments for the correction function
        args = {
            "dataset_name": dataset_name,
            "bucket": "aind-open-data",
            "s3_bucket": "aind-open-data",
            "z_correct": False,  # Default to 2D correction
            "pipeline": pipeline_bool,
        }
        
        # Run 2D camera correction
        run_2d_camera_correction(args)
        success = True
        
        # Record successful processing
        with open(results_dir / "processing_complete.txt", "a") as f:
            f.write(f"Successfully processed {dataset_name}\n")
            f.write(f"Zarr path: {dataset_path}\n\n")

    
    if not success:
        logger.error("Failed to process any datasets successfully")
    
    return success

if __name__ == "__main__":
    try:
        process_zarr_datasets()
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise