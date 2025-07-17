import s3fs
import os
from typing import List, Dict, Any
import json
import glob
import pathlib
import logging

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
    
    Searches for data_description.json in multiple possible locations using glob patterns:
    1. ../data/output_aind_metadata/
    2. ../data/
    3. ../data/{any_subdirectory}/
    
    Returns
    -------
    str
        The dataset name extracted from the 'name' field in data_description.json
        
    Raises
    ------
    FileNotFoundError
        If no data_description.json file is found in any of the search locations
    RuntimeError
        If error occurs while loading or parsing the JSON configuration
    """
    base_data_dir = pathlib.Path("/data")
    
    # Use glob to search for data_description.json in all possible locations
    search_patterns = [
        base_data_dir / "output_aind_metadata" / "data_description.json",
        base_data_dir / "data_description.json",
        # glob.glob(f"{base_data_dir.as_posix()}/data_description.json")[0],  # Any subdirectory
        glob.glob(f"{base_data_dir.as_posix()}/*/data_description.json")[0], 
        
    ]
    
    # Find the first existing file
    json_file_path = None
    for json_path in search_patterns:
        if pathlib.Path(json_path).exists():
            json_file_path = json_path
            logger.info(f"Found data_description.json at: {json_file_path}")
            break
    
    if json_file_path is None:
        raise FileNotFoundError(
            f"No data_description.json file found in {base_data_dir} or any of its subdirectories"
        )
    
    logger.info(f"Loading configuration from {json_file_path}")
    
    try:
        with open(json_file_path, 'r') as f:
            config = json.load(f)
            dataset_name = config.get('name')
            if not dataset_name:
                raise ValueError("'name' field not found in data_description.json")
            logger.info(f"Loaded dataset name: {dataset_name}")
            return dataset_name
    except Exception as e:
        raise RuntimeError(f"Error loading data_description.json: {str(e)}")

# def list_all_tiles_in_path(SPIM_folder: str) -> list:
#     SPIM_folder = pathlib.Path(SPIM_folder)
#     # assert SPIM_folder.exists()
    
#     return list(SPIM_folder.glob("*.zarr"))

# def list_all_tiles_in_bucket_path(bucket_SPIM_folder: str, bucket_name = "aind-open-data") -> list: 
#     """
#     List all tiles in bucket path in s3
#     """
#     # s3 = boto3.resource('s3')
#     bucket_name, prefix = bucket_SPIM_folder.replace("s3://","").split("/", 1)
#     # my_bucket = s3.Bucket(bucket_name)

#     client = boto3.client('s3')
#     result = client.list_objects(Bucket=bucket_name, Prefix=prefix+"/", Delimiter='/')
#     # print(result)
#     tiles = []
#     for o in result.get('CommonPrefixes'):
#         #print('sub folder : ', o.get('Prefix'))
#         tiles.append(o.get('Prefix')) 
#     return tiles

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
