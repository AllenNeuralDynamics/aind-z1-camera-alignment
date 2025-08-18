import dask
from dask.distributed import Client, LocalCluster, performance_report
# from ome_zarr.io import parse_url
from aind_data_transfer.util.io_utils import BlockedArrayWriter
from aind_data_transfer.util.chunk_utils import ensure_shape_5d, ensure_array_5d
from aind_data_transfer.transformations.ome_zarr import (
    store_array,
    downsample_and_store,
    _get_bytes,
    write_ome_ngff_metadata
)
import s3fs
import time
import numpy as np
from typing import Union
import dask.array as da
from numcodecs import blosc
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

def save_corrected_tiles_to_s3(corrected_scratch_dir, s3_path, resolution_zyx):
    """
    Save camera-corrected tiles from local scratch directory to S3.
    
    Orchestrates the parallel upload of corrected zarr tiles to S3 storage,
    setting up a Dask cluster for concurrent processing.
    
    Parameters
    ----------
    corrected_scratch_dir : str
        Local directory path containing corrected zarr tiles
    s3_path : str
        S3 URI path where tiles should be uploaded (e.g., 's3://bucket/prefix/')
    resolution_zyx : tuple or list
        Voxel resolution in microns as (z, y, x) tuple
        
    Returns
    -------
    None
        Tiles are uploaded to S3
        
    Notes
    -----
    Uses a Dask LocalCluster with 14 workers for parallel processing.
    Each tile is processed independently for efficient scaling.
    """
    num_cpus = 14

    client = Client(LocalCluster(n_workers=num_cpus, threads_per_worker=1, processes=True))


    list_of_tiles = list(glob(f'{corrected_scratch_dir}/*.zarr'))
    LOGGER.info(f'Saving tiles now!')
    for tilename in list_of_tiles:
        LOGGER.info(f'saving {tilename}.... ')
        output_path = s3_path + tilename
        save_tile(tilename, output_path, resolution_zyx, num_cpus)


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


def run_multiscale(full_res_arr: dask.array, 
                   out_group: zarr.group,
                   voxel_sizes_zyx: tuple):
    """
    Generate and save multiscale pyramid for a zarr array.
    
    Creates multiple resolution levels by downsampling and saves them
    with proper OME-NGFF multiscale metadata.
    
    Parameters
    ----------
    full_res_arr : dask.array.Array
        Full resolution input array to create pyramid from
    out_group : zarr.Group
        Output zarr group to store the multiscale data
    voxel_sizes_zyx : tuple[float, float, float]
        Voxel sizes in microns for Z, Y, X dimensions
        
    Notes
    -----
    Creates pyramid levels with 2x downsampling in each spatial dimension.
    Adds proper OME-NGFF metadata including scale and coordinate transformations.
    """ 

    arr = ensure_array_5d(full_res_arr)
    arr = arr.rechunk((1, 1, 128, 256, 256))
    LOGGER.info(f"input array: {arr}")

    LOGGER.info(f"input array size: {arr.nbytes / 2 ** 20} MiB")
    block_shape = ensure_shape_5d(BlockedArrayWriter.get_block_shape(arr))
    LOGGER.info(f"block shape: {block_shape}")
    
    scale_factors = (2, 2, 2) 
    scale_factors = ensure_shape_5d(scale_factors)
    n_levels = 5
    compressor = blosc.Blosc("zstd", 1, shuffle=blosc.SHUFFLE)  #None

    # Actual Processing
    t0 = time.time()

    write_ome_ngff_metadata(
            out_group,
            arr,
            out_group.path,
            n_levels,
            scale_factors[-3:],
            voxel_sizes_zyx[-3:],
            origin=None,
        )
    
    store_array(arr, out_group, '0', block_shape, compressor)
    #out_group.create_dataset("0", data = arr, compressor=compressor, overwrite = True, chunks = (1, 1, 128, 256, 256))

 

    pyramid = downsample_and_store(
        arr, out_group, n_levels, scale_factors, block_shape, compressor
    )
    write_time = time.time() - t0

    LOGGER.info(
        f"Finished writing tile.\n"
        f"Took {write_time}s. {_get_bytes(pyramid) / write_time / (1024 ** 2)} MiB/s"
    )



def save_tile(dataset_loc, output_path, resolution_zyx, num_cpus):
    """
    Save a single camera-corrected tile to S3 as multiscale OME-Zarr.
    
    Converts the corrected tile to multiscale OME-Zarr format with proper
    metadata and uploads to the specified S3 location.
    
    Parameters
    ----------
    dataset_loc : str
        Path to camera corrected zarr file in local scratch directory
    output_path : str
        S3 URI path where the tile will be saved
    resolution_zyx : list[float]
        Voxel resolution in microns (Z, Y, X order) for highest resolution level
    num_cpus : int
        Number of CPU cores to use for parallel processing
        
    Notes
    -----
    Creates multiscale pyramid with downsampling factors and proper OME-NGFF metadata.
    Uses S3FileSystem with connection pooling and retry configuration for reliability.
    """
    # resolution_zyx = (1,0.256, 0.256)
    #print(f'{dataset_loc}')

    #get number of cpus
    


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

    split_out = output_path.split('/')
    ome_path = 's3://'+split_out[2]+'/' + split_out[3] + '/'+split_out[4]
    
    store = s3fs.S3Map(root=ome_path, s3=s3, check=False)
    root_group = zarr.group(store=store, overwrite=False)

    tilename = str(Path(output_path).name)
    out_group = root_group.create_group(tilename, overwrite=True) 



    #start dask client()

 
    #downsample and save
    with performance_report(filename="/results/dask-report.html"):

        #save as local zarr
        
        # zarr_loc = f'/scratch/{tilename}.zarr'

        # zarr.save_array( zarr_loc, corrected_tile)
        #temp_zarr = zarr.load(zarr_loc)
        #corr_arr = da.from_array(corrected_tile, name=tilename)
        corr_arr = da.from_zarr(dataset_loc, chunks = (128,256,256))
        run_multiscale(corr_arr, out_group, resolution_zyx)   
    

    return 


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
