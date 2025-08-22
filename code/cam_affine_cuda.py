# apply affine transforms for cameras 1-3
# affines are found based on camera and exposure

import sys, os, csv, numpy as np
import time, threading
from glob import glob
import zarr
from pathlib import Path
from calc_affine import calc_affine, get_channel_wavelength_from_single_channel_digit
from qc_results import make_and_save_qc_plots
import logging
from s3_writer import  get_resolution_zyx,  copy_file_to_s3
from typing import List, Dict, Any
import pathlib
from utils import (
    list_zarr_tiles_from_s3,
    add_affines_to_channel, 
    update_xml_path_to_camera_alignment, 
    copy_file
)

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
TILE_ALIGNMENT_S3_FOLDER_NAME = "image_tile_alignment"

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

        s3_path_rc = f"s3://aind-open-data/{name}/image_radial_correction/"
        rc_data_folder_list = list_zarr_tiles_from_s3(s3_path_rc)
        xml_path = data_folder + 'stitching_rc_spot_channels.xml'
        assert Path(xml_path).exists()

        #check that there are tiles to work on
        if len(rc_data_folder_list)!=0:
            calc_affine(s3_path_rc)
            
            updated_xml_path = apply_affine_to_xml(data_folder, scratch_root, xml_path = xml_path)
            updated_xml_path_forward = apply_affine_to_xml_forward_transform(data_folder, scratch_root, xml_path = xml_path )
            # apply_affine_to_tiles(s3_path_rc, scratch_root, out_dir)
            LOGGER.info('*'*50)
            LOGGER.info(f'Saving to S3 now')
            LOGGER.info('*'*50)
            s3_path = f's3://{s3_bucket}/{name}/{TILE_ALIGNMENT_S3_FOLDER_NAME}/{Path(updated_xml_path).name}'
            copy_file_to_s3(updated_xml_path, s3_path)
            copy_file_to_s3(updated_xml_path_forward, s3_path)
            results_xml_path = results_root + Path(updated_xml_path).name
            results_xml_path_forward = results_root + Path(updated_xml_path_forward).name
            LOGGER.info(f'Copying XML to results directory: {results_xml_path}')
            copy_file(updated_xml_path, results_xml_path)
            copy_file(updated_xml_path_forward, results_xml_path_forward)

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

def out_name(fn, out_dir): 
    """
    Generate output filename from input filename and output directory.
    
    Parameters
    ----------
    fn : str
        Input filename
    out_dir : str
        Output directory path
        
    Returns
    -------
    str
        Full path to output file
    """
    return out_dir + os.path.basename(fn)

def get_channel_from_fn(fn):
    """
    Extract channel identifier from filename.
    
    Parameters
    ----------
    fn : str
        Filename containing channel information
        
    Returns
    -------
    str
        Channel identifier extracted from filename
        
    Notes
    -----
    Expects filename format ending with '_CHANNEL.zarr' where CHANNEL
    is the channel identifier.
    """
    return fn.split('_')[-1].split('.')[0]

def invert_affine_2d(aff_2x3):
    """Invert a 2x3 affine transformation matrix"""
    A = aff_2x3[:, :2]  # 2x2 linear part
    t = aff_2x3[:, 2]   # translation vector
    A_inv = np.linalg.inv(A)
    t_inv = -A_inv @ t
    return np.column_stack([A_inv, t_inv])

    
def apply_affine_to_xml_forward_transform(root, scratch_root, xml_path = None): 
    """
    Read affine transforms from file and append them to the relevant XML without inversion.
    
    Loads precomputed affine transformation matrices and integrates them into
    the XML configuration file for downstream processing steps. Uses the forward
    transformation matrices without inversion.
    
    Parameters
    ----------
    root : str
        Root directory path containing source data
    scratch_root : str
        Directory path containing the affine transformation file (updated.M.txt)
    xml_path : str, optional
        Path to input XML file, by default None
        If None, uses root + 'stitching_rc_spot_channels.xml'
        
    Returns
    -------
    str
        Path to the updated XML file with camera alignment forward transforms
        
    Notes
    -----
    - Reads transformation matrices from scratch_root + 'updated.M.txt'
    - Creates output XML at '/scratch/stitching_cam_alignment_forward_transform_spot_channels.xml'
    - Updates XML path metadata to reflect camera alignment processing
    - Adds forward affine transforms (no inversion) for each channel found in the transform file
    """
    affine_path = scratch_root + 'updated.M.txt'
    with open(affine_path) as f: affine_dict = {x[0]: list(map(float, x[1:])) for x in csv.reader(f, dialect='excel-tab')}

    if xml_path == None: 
        xml_path = root + 'stitching_rc_spot_channels.xml'
    output_xml_path = '/scratch/stitching_cam_alignment_forward_transform_spot_channels.xml'
    update_xml_path_to_camera_alignment(xml_path, output_xml_path)

    for channel in affine_dict.keys(): 
        raw_affine = affine_dict[channel]
        #XYZ for bigstitcher
        raw_affine = np.array([[raw_affine[0], raw_affine[1], raw_affine[2]], 
                           [raw_affine[3], raw_affine[4], raw_affine[5]]])
        
        # Use the forward transform directly (no inversion)
        affine_matrix = raw_affine

        reordered_affine = affine_matrix.flatten()
        # write out as long string

        updated_xml_path = add_affines_to_channel(xml_path, reordered_affine, channel, output_xml_path)
        xml_path = updated_xml_path
        LOGGER.info(f"Finished processing channel {channel}")

    return xml_path

def apply_affine_to_xml(root, scratch_root, xml_path = None): 
    """
    Read affine transforms from file and append them to the relevant XML.
    
    Loads precomputed affine transformation matrices and integrates them into
    the XML configuration file for downstream processing steps.
    
    Parameters
    ----------
    root : str
        Root directory path containing source data
    scratch_root : str
        Directory path containing the affine transformation file (updated.M.txt)
    xml_path : str, optional
        Path to input XML file, by default None
        If None, uses root + 'stitching_rc_spot_channels.xml'
        
    Returns
    -------
    str
        Path to the updated XML file with camera alignment transforms
        
    Notes
    -----
    - Reads transformation matrices from scratch_root + 'updated.M.txt'
    - Creates output XML at '/results/stitching_cam_alignment_spot_channels.xml'
    - Updates XML path metadata to reflect camera alignment processing
    - Adds affine transforms for each channel found in the transform file
    """
    affine_path = scratch_root + 'updated.M.txt'
    with open(affine_path) as f: affine_dict = {x[0]: list(map(float, x[1:])) for x in csv.reader(f, dialect='excel-tab')}

    if xml_path == None: 
        xml_path = root + 'stitching_rc_spot_channels.xml'
    output_xml_path = '/scratch/stitching_cam_alignment_spot_channels.xml'
    update_xml_path_to_camera_alignment(xml_path, output_xml_path)

    for channel in affine_dict.keys(): 
        raw_affine = affine_dict[channel]
        #XYZ for bigstitcher
        raw_affine = np.array([[raw_affine[0], raw_affine[1], raw_affine[2]], 
                           [raw_affine[3], raw_affine[4], raw_affine[5]]])
        affine_matrix = invert_affine_2d(raw_affine)

        reordered_affine = affine_matrix.flatten()
        # write out as long string

        updated_xml_path = add_affines_to_channel(xml_path, reordered_affine, channel, output_xml_path)
        xml_path = updated_xml_path
        LOGGER.info(f"Finished processing channel {channel}")

    return xml_path

if __name__ == "__main__":
    debug()