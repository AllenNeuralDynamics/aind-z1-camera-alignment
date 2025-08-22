# from ome_zarr.io import parse_url
import s3fs
import time
import numpy as np
from typing import Union
import dask.array as da
import logging
import zarr
from pathlib import Path
import json
from xml.etree import ElementTree as ET
from glob import glob
import pathlib


logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def get_resolution_zyx(dataset_path):
    """
    Get the voxel resolution from acquisition metadata files.
    
    Attempts to extract voxel resolution from acquisition.json file using
    multiple search locations, with fallback to XML metadata if needed.
    
    Parameters
    ----------
    dataset_path : str
        Dataset name or path used to locate acquisition metadata files
        
    Returns
    -------
    list[float]
        Voxel resolution in microns as [z, y, x] list
        
    Raises
    ------
    AssertionError
        If no acquisition.json file can be found in any search location
        
    Notes
    -----
    Searches for acquisition.json in the following order:
    1. /data/{dataset_path}/acquisition.json
    2. /data/acquisition.json  
    3. /data/output_aind_metadata/acquisition.json
    
    If JSON parsing fails, falls back to XML metadata parsing.
    Supports both aind-data-schema v2.0.0 format and legacy XML format.
    """
    try:
        acq_json_path = '/data/'+dataset_path + "/acquisition.json"
        if not Path(acq_json_path).exists():
            acq_json_path = '/data/'+ "acquisition.json"
            if not Path(acq_json_path).exists(): 
                acq_json_path = f'/data/output_aind_metadata/acquisition.json'
            assert Path(acq_json_path).exists()
        with open(acq_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        return _get_voxel_resolution_schema_2(data)
    except: 
            #use xml
        xml_list = list(glob(f'/data/{dataset_path}/*.xml'))
        xml_file_loc = xml_list[0]
        tree = ET.parse(xml_file_loc)
        root = tree.getroot()
        for elem in root.iter('voxelSize'):
            xyz_voxelsize= elem.findtext('size')
            break

        zyx_voxelsize_list= xyz_voxelsize.split(' ')
        zyx_voxelsize_list.reverse() # does in place
        voxel_float_zyx = [float(i) for i in zyx_voxelsize_list]
        return voxel_float_zyx

def _get_voxel_resolution_schema_2(
        acquisition_config,
    ) -> list[float]:
    """
    Get the voxel resolution from an acquisition.json file for aind-data-schema==2.0.0.
    
    Extracts voxel resolution from the new acquisition schema format used by
    aind-data-schema version 2.0.0 and later.
    
    Parameters
    ----------
    acquisition_config : dict
        Parsed acquisition.json configuration dictionary
        
    Returns
    -------
    list[float]
        Voxel resolution as [z, y, x] in microns
        
    Raises
    ------
    ValueError
        If acquisition_config structure is invalid or missing required fields
        
    Notes
    -----
    Assumes all tiles in the dataset were acquired with the same resolution.
    Looks for Scale transform in the first data stream configuration.
    """

    # Grabbing a tile with metadata from acquisition - we assume all
    # dataset was acquired with the same resolution
    try:
        data_stream = acquisition_config.get("data_streams", [])[0]
        configuration = data_stream.get("configurations", [])[0]
        image = configuration.get("images", [])[0]
        image_to_acquisition_transform = image[
            "image_to_acquisition_transform"
        ]
    except (IndexError, AttributeError, KeyError) as e:
        raise ValueError(
            "acquisition_config structure is invalid or missing "
            "required fields"
        ) from e

    scale_transform = [
        x["scale"]
        for x in image_to_acquisition_transform
        if x["object_type"] == "Scale"
    ][0]

    x = float(scale_transform[0])
    y = float(scale_transform[1])
    z = float(scale_transform[2])

    return [z, y, x]

def ensure_array_5d(
    arr: Union[np.ndarray, da.Array]
) -> Union[np.ndarray, da.Array]:
    """
    Checks that the array is 5D, adding singleton dimensions to the
    start of the array if less, throwing a DimensionsError if more
    Args:
        arr: the arraylike object
    Returns:
        the 5D array
    Raises:
        DimensionsError: if the array has more than 5 dimensions
    """
    if arr.ndim > 5:
        raise DimensionsError("Only arrays up to 5D are supported")
    while arr.ndim < 5:
        arr = arr[np.newaxis, ...]
    return arr


def copy_file_to_s3(file_path: str, s3_location: str) -> bool:
    """
    Copy a local file to S3.
    
    Parameters
    ----------
    file_path : str
        Path to the local file to copy
    s3_location : str
        S3 destination path. Can be in format:
        - 's3://bucket/path/to/file.xml'
        - 'bucket/path/to/file.xml'
        
    Returns
    -------
    bool
        True if copy was successful, False otherwise
        
    Raises
    ------
    FileNotFoundError
        If the local file doesn't exist
    RuntimeError
        If there's an error uploading to S3
    """
    
    # Check if local file exists
    local_path = pathlib.Path(file_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {file_path}")
    
    # Normalize S3 location - remove s3:// prefix if present
    s3_path_clean = s3_location.replace('s3://', '') if s3_location.startswith('s3://') else s3_location
    
    LOGGER.info(f"Copying {file_path} to s3://{s3_path_clean}")
    num_cpus = 14
    
    try:
        # Initialize S3 filesystem
        s3 = s3fs.S3FileSystem(
            config_kwargs={
                'max_pool_connections': num_cpus,
                'retries': {
                    'total_max_attempts': 1000, 
                    'mode': 'adaptive',
                }
            }, 
            use_ssl=True
        )

        
        # Copy the file
        s3.put(file_path, s3_path_clean)
        
        # Verify the file was uploaded by checking if it exists
        if s3.exists(s3_path_clean):
            file_size = local_path.stat().st_size
            LOGGER.info(f"Successfully uploaded {file_path} ({file_size} bytes) to s3://{s3_path_clean}")
            return True
        else:
            LOGGER.error(f"Upload verification failed for s3://{s3_path_clean}")
            return False
            
    except Exception as e:
        raise RuntimeError(f"Error uploading {file_path} to S3: {str(e)}")
