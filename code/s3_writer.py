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


logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def save_corrected_tiles_to_s3(corrected_scratch_dir, s3_path, resolution_zyx):
    num_cpus = 14

    client = Client(LocalCluster(n_workers=num_cpus, threads_per_worker=1, processes=True))


    list_of_tiles = list(glob(f'{corrected_scratch_dir}/*.zarr'))
    LOGGER.info(f'Saving tiles now!')
    for tilename in list_of_tiles:
        output_path = s3_path + tilename
        save_tile(tilename, output_path, resolution_zyx, num_cpus)


def get_resolution_zyx(dataset_path):
    """get the resolution from the first tile in the acquisition.json"""
    try:
        acq_json_path = '/data/'+dataset_path + "/acquisition.json"
        if not Path(acq_json_path).exists():
            acq_json_path = '/data/'+ "acquisition.json"
            assert Path(acq_json_path).exists()
        with open(acq_json_path, 'r') as file:
            data = json.load(file)
        
        # Get the first tile
        first_tile = data['tiles'][0]
        
        # Extract the scale from coordinate_transformations
        for transformation in first_tile['coordinate_transformations']:
            if transformation['type'] == 'scale':
                resolution_xyz =  transformation['scale']
                resolution_zyx = [float(i*1e6) for i in resolution_xyz]
                resolution_zyx.reverse()
                return resolution_zyx
            # If no scale found, return None
        return None
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
    """Save a single tile to an s3 location
    
    Parameters:
    -----------
    dataset_loc: str
        path to camera corrected zarr file in /scratch/

    output_path: str
        s3 uri path to dataset's resting place in s3 bucket

    resolution_zyx: list
        voxel resolution in microns of highest resolution of dataset

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
    ome_path = 's3://'+split_out[2]+'/' + split_out[3] + '/'+split_out[4]+'.ome.zarr'
    
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
