# apply affine transforms for cameras 1-3
# affines are found based on camera and exposure

import sys, os, csv, numpy as np
import time, threading
from glob import glob
import dask.array as da
import cupy
from cupyx.scipy.ndimage import map_coordinates
with cupy.cuda.Device(0): cupy.get_default_memory_pool().set_limit(size = 14 * 1024**3) 
import zarr
from pathlib import Path

from calc_affine import calc_affine, get_channel_wavelength_from_single_channel_digit
from qc_results import make_and_save_qc_plots

import logging
from s3_writer import save_tile, get_resolution_zyx, save_corrected_tiles_to_s3
from co_api import list_data_directory
from typing import List, Dict, Any
import pathlib
from utils import (
    list_zarr_tiles_from_s3,
)




logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
CAMERA_CORRECTED_S3_FOLDER_NAME="image_camera_alignment"

def main(args):
    if args['pipeline']: 
        data_folder = '/data/'
        scratch_root = '/scratch/'
        results_root = '/results/'
        name = args["dataset_name"]
        s3_bucket = args["bucket"]

        print(f'Running pipeline version of Camera Alignment!')
        print(f'name {name}')

        out_dir = scratch_root + name + "/affine.ome.zarr/" #saves temp full res of data here 
        qc_results_dir = results_root + name + "/tile_qc_plots/" #unnecessary 

        #TODO add failsafe code that submits calc_affine() with the path to the folder radial_correction_temp, which contains the zarr files to do camera_alignment on
        s3_path_rc = f"s3://aind-open-data/{name}/image_radial_correction/"
        rc_data_folder_list = list_zarr_tiles_from_s3(s3_path_rc)

        #check that there are tiles to work on
        if len(rc_data_folder_list)!=0:
            calc_affine(s3_path_rc)
            apply_affine_to_tiles(s3_path_rc, scratch_root, out_dir)
            LOGGER.info('*'*50)
            LOGGER.info(f'Saving to S3 now')
            LOGGER.info('*'*50)
            s3_path = f's3://{s3_bucket}/{name}/{CAMERA_CORRECTED_S3_FOLDER_NAME}'
            #list_data_directory('/scratch/')
            resolution_zyx = get_resolution_zyx(name)
            save_corrected_tiles_to_s3(out_dir, s3_path, resolution_zyx)
            LOGGER.info('*'*50)
            LOGGER.info(f'Making QC Figures now ')
            LOGGER.info('*'*50)
            make_and_save_qc_plots(rc_root_path, out_dir)
        else:
            print(f'no radial_correction_temp')
        
        

    else: 
        root = '/data/'
        scratch_root = '/scratch/'
        results_root = '/results/'

        name = args["dataset_name"]
        s3_bucket = args["bucket"]
        root += name+'/radial_correction.ome.zarr/'
        backup_name = "/data/"+ name + '/SPIM.ome.zarr/'
        out_dir = scratch_root + name + "/affine.ome.zarr/" #saves temp full res of data here before saving to s3 and making QC plots
        qc_results_dir = results_root + name + "/tile_qc_plots/" #unnecessary 
        
        #calulate affine between sets of channels 

        LOGGER.info(f'Calculating affine between channels now ! ')
        LOGGER.info('*'*50)

        print(f'name {name}')
        print(f'root {root}')
        print(f'bu name{backup_name}')


        rc_root_list = list(Path('/data/').glob('image_radial_correction'))
        if len(rc_root_list)==0:
            rc_root_list = list(Path('/data/').joinpath(name).glob('image_radial_correction'))
            print(f'running capsule version')
        else: 
            print(f'running pipeline version')
        print(f'rc_root_list {rc_root_list}')

        if len(rc_root_list) >0 :
            rc_root_path = rc_root_list[0].as_posix()
            calc_affine(rc_root_path)
            apply_affine_to_tiles(rc_root_path, scratch_root, out_dir)

            LOGGER.info('*'*50)
            LOGGER.info(f'Saving to S3 now')
            LOGGER.info('*'*50)
            s3_path = f's3://{s3_bucket}/{name}/{CAMERA_CORRECTED_S3_FOLDER_NAME}'
            #list_data_directory('/scratch/')
            resolution_zyx = get_resolution_zyx(name)
            save_corrected_tiles_to_s3(out_dir, s3_path, resolution_zyx)
            LOGGER.info('*'*50)
            LOGGER.info(f'Making QC Figures now ')
            LOGGER.info('*'*50)
            make_and_save_qc_plots(rc_root_path, out_dir)
        
        else:
            calc_affine(backup_name)
            apply_affine_to_tiles(backup_name, scratch_root, out_dir)
            LOGGER.info('*'*50)
            LOGGER.info(f'Making QC Figures now ')
            LOGGER.info('*'*50)
            make_and_save_qc_plots(backup_name, out_dir)
            s3_path = f's3://{s3_bucket}/{name}/{CAMERA_CORRECTED_S3_FOLDER_NAME}'
            
            resolution_zyx = get_resolution_zyx(name)

            save_corrected_tiles_to_s3(out_dir, s3_path, resolution_zyx)
            
    

def debug():
    root = '/data/'
    scratch_root = '/scratch/'
    results_root = '/results/'

    name = "HCR_BL6-001_2023-06-19_00-01-00"
    s3_bucket = 'aind-open-data'


    root += name+'/radial_correction.ome.zarr/'
    backup_name = '/data/' + name + '/SPIM.ome.zarr/'
    out_dir = scratch_root + name + "/affine.ome.zarr/"

    #calulate affine between sets of channels 

    LOGGER.info(f'Calculating affine between channels now ! ')
    LOGGER.info('*'*50)



    if Path(root).exists():
        calc_affine(root)
        apply_affine_to_tiles(root, scratch_root, out_dir)

        LOGGER.info('*'*50)
        LOGGER.info(f'Making QC Figures now ')
        LOGGER.info('*'*50)
        make_and_save_qc_plots(root, out_dir)
    else:
        calc_affine(backup_name)
        apply_affine_to_tiles(backup_name, scratch_root, out_dir)
        LOGGER.info('*'*50)
        LOGGER.info(f'Making QC Figures now ')
        LOGGER.info('*'*50)
        make_and_save_qc_plots(backup_name, out_dir)
    s3_path = f's3://{s3_bucket}/{name}/{CAMERA_CORRECTED_S3_FOLDER_NAME}'
    
    resolution_zyx = get_resolution_zyx(name)

    # save_corrected_tiles_to_s3(out_dir, s3_path, resolution_zyx)
    

    # if Path(root).exists():
    #     calc_affine(root)
    #     apply_affine_to_tiles(root, scratch_root, out_dir)
    #     make_and_save_qc_plots(root, out_dir)
    # elif Path(backup_name).exists():
    #     calc_affine(backup_name)
    #     apply_affine_to_tiles(backup_name, scratch_root, out_dir)
    #     make_and_save_qc_plots(backup_name, out_dir)
    # else:
    #     LOGGER.error("No zarr file at SPIM.ome.zarr or radial_correction.ome.zarr")
    # s3_path = f's3://{s3_bucket}/{name}/corrected/'
     
    # resolution_zyx = get_resolution_zyx(name)

    #save_corrected_tiles_to_s3(out_dir, s3_path, resolution_zyx)


def find_zarr_datasets() -> List[pathlib.Path]:
    """
    Find all zarr datasets in the data directory.
    Returns a list of paths to zarr datasets.
    """
    data_dir = pathlib.Path("/data")
    
    # Look for zarr files directly in data directory and one level deep
    zarr_datasets = []
    
    # Direct zarr files
    zarr_datasets.extend(list(data_dir.glob('*.zarr')))
    
    # Check one level deep
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            zarr_datasets.extend(list(subdir.glob('*.zarr')))
    
    logger.info(f"Found zarr datasets: {zarr_datasets}")
    return zarr_datasets

def out_name(fn, out_dir): return out_dir + os.path.basename(fn)
# def out_name(fn): return Path(out_dir).joinpath(Path(fn).Parent)


def process_chunk(chunk, aff):
    chunk_cupy = cupy.array(chunk)
    pixels = chunk.shape[1]
    z_chunk = chunk.shape[0]
    
    coords = cupy.array(cupy.meshgrid(*[cupy.arange(pixels), cupy.arange(pixels)], indexing='ij'), np.float32).reshape((2, -1)).T
    coords = cupy.vstack((cupy.tile(cupy.arange(z_chunk).astype(np.float32), (pixels, pixels, 1)).swapaxes(2,0)[cupy.newaxis,:,:,:], 
                          cupy.repeat((coords @ aff[:,:2].T + aff[:,2]).T.reshape((2, pixels, pixels)).astype(np.float32)[:, cupy.newaxis, :, :], z_chunk, axis=1)))
    
    transformed = map_coordinates(chunk_cupy, coords, order=1, mode='constant')
    return transformed.get()



def get_channel_from_fn(fn):
    return fn.split('_')[-1].split('.')[0]

def apply_affine_to_tiles(root, scratch_root, out_dir):
    """Reads the affine from file and applies it to each tile in the dataset

    """
        
    affine = scratch_root + 'updated.M.txt'
    with open(affine) as f: Ms = {x[0]: list(map(float, x[1:])) for x in csv.reader(f, dialect='excel-tab')}
    prefix = '0'

    #list of tiles
    list_of_tiles = list(glob(f'{root}/*.zarr'))


    for fn in list_of_tiles:
        LOGGER.info(f"Processing {fn}")
        zarr_array = da.from_zarr(fn, prefix)
        _,_, z, y, x = zarr_array.shape
        z_chunk = min(50, z)  # Ensure z_chunk doesn't exceed total z

        fn_channel = get_channel_from_fn(fn)
        if len(fn_channel)!=3:
            fn_channel = get_channel_wavelength_from_single_channel_digit(fn_channel)
        out_zarr = zarr.open(out_name(fn, out_dir), mode='w', shape=(z, y, x), chunks=(z_chunk, y, x), dtype=np.uint16)

        if fn_channel == "405" or fn_channel not in Ms.keys():
            # For channel 405, just copy the data
            for i in range(0, z, z_chunk):
                chunk = zarr_array[0,0,i:i+z_chunk].compute()
                out_zarr[i:i+z_chunk] = chunk
        else:
            aff = Ms[fn_channel]
            aff = cupy.array([[aff[4], aff[3], aff[5]], [aff[1], aff[0], aff[2]]]).reshape((2,3))

            for i in range(0, z, z_chunk):
                chunk = zarr_array[0,0,i:i+z_chunk].compute()
                transformed_chunk = process_chunk(chunk, aff)
                out_zarr[i:i+z_chunk] = transformed_chunk

        LOGGER.info(f"Finished processing {fn}")

if __name__ == "__main__":
    debug()