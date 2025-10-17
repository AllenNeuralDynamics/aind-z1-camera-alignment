# apply affine transforms for cameras 1-3
# affines are found based on camera and exposure

import sys, os, csv, numpy as np
import time, threading
from glob import glob
import zarr
from pathlib import Path
from calc_affine import calc_affine, get_channel_wavelength_from_single_channel_digit
import logging
from s3_writer import get_resolution_zyx, copy_file_to_s3
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

def handle_skip_alignment(args: Dict[str, Any]) -> None:
    """
    Handle projects that should skip camera alignment (like PLACE/proteomics datasets).
    
    For projects that don't need camera alignment, this function copies the input XML
    file and renames it to match the expected output naming convention used for
    camera-aligned datasets. This ensures downstream processes can find the XML
    file in the expected location with the expected name.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Arguments dictionary containing:
        - dataset_name: Name of the dataset
        - bucket: S3 bucket name
        - skip_alignment: Boolean indicating whether to skip alignment
        
    Returns
    -------
    None
        Files are copied and renamed in place
    """
    if not args.get('skip_alignment', False):
        return
        
    data_folder = '/data/'
    results_root = '/results/'
    name = args["dataset_name"]
    s3_bucket = args["bucket"]
    
    # Source XML path (input from previous pipeline step)
    input_xml_path = Path(data_folder) / 'stitching_rc_spot_channels.xml'
    
    if not input_xml_path.exists():
        LOGGER.warning(f"Input XML not found at {input_xml_path}. Skipping XML copy.")
        return
    
    # Generate output XML names matching camera alignment convention
    output_xml_name = "stitching_cam_alignment_spot_channels.xml"
    output_xml_forward_name = "stitching_cam_alignment_forward_transform_spot_channels.xml"
    
    # Copy to results directory
    results_xml_path = Path(results_root) / output_xml_name
    results_xml_forward_path = Path(results_root) / output_xml_forward_name
    
    copy_file(str(input_xml_path), str(results_xml_path))
    copy_file(str(input_xml_path), str(results_xml_forward_path))
    
    # Copy to S3
    s3_path_base = f's3://{s3_bucket}/{name}/{TILE_ALIGNMENT_S3_FOLDER_NAME}/'
    copy_file_to_s3(str(results_xml_path), s3_path_base + output_xml_name)
    copy_file_to_s3(str(results_xml_forward_path), s3_path_base + output_xml_forward_name)
    
    LOGGER.info(f"Skipped camera alignment for project. XML files copied and renamed.")
    LOGGER.info(f"Results: {results_xml_path}")
    LOGGER.info(f"S3: {s3_path_base}")

def run_camera_alignment_pipeline(args: Dict[str, Any]) -> None:
    """
    Run camera alignment for pipeline mode (S3-based data).
    
    Parameters
    ----------
    args : Dict[str, Any]
        Arguments dictionary containing dataset_name, bucket, etc.
    """
    data_folder = '/data/'
    scratch_root = '/scratch/'
    results_root = '/results/'
    name = args["dataset_name"]
    s3_bucket = args["bucket"]

    LOGGER.info(f'Running pipeline version of Camera Alignment for {name}')

    # Check for radial correction data on S3
    s3_path_rc = f"s3://aind-open-data/{name}/image_radial_correction/"
    rc_data_folder_list = list_zarr_tiles_from_s3(s3_path_rc)
    xml_path = data_folder + 'stitching_rc_spot_channels.xml'
    
    if not Path(xml_path).exists():
        LOGGER.error(f"Required XML file not found: {xml_path}")
        return
    
    if len(rc_data_folder_list) == 0:
        LOGGER.warning(f'No radial correction data found at {s3_path_rc}')
        return
        
    # Calculate affine transforms
    calc_affine(s3_path_rc)
    
    # Apply transforms to XML
    updated_xml_path = apply_affine_to_xml(data_folder, scratch_root, xml_path=xml_path)
    updated_xml_path_forward = apply_affine_to_xml_forward_transform(
        data_folder, scratch_root, xml_path=xml_path
    )
    
    # Save results
    _save_xml_results(updated_xml_path, updated_xml_path_forward, results_root, s3_bucket, name)
    

def run_camera_alignment_capsule(args: Dict[str, Any]) -> None:
    """
    Run camera alignment for capsule/local mode.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Arguments dictionary containing dataset_name, bucket, etc.
    """
    data_folder = '/data/'
    scratch_root = '/scratch/'
    results_root = '/results/'
    name = args["dataset_name"]
    s3_bucket = args["bucket"]
    s3_path_rc = f"s3://aind-open-data/{name}/image_radial_correction/"
    
    LOGGER.info(f'Running capsule version of Camera Alignment for {name}')
    
    # Try to find radial correction data locally
    rc_root_list = list(Path(data_folder).glob('image_radial_correction'))
    if len(rc_root_list) == 0:
        rc_root_list = list(Path(data_folder).joinpath(name).glob('image_radial_correction'))
    
    if len(rc_root_list) > 0:
        rc_root_path = rc_root_list[0].as_posix()
        _process_local_data(rc_root_path, data_folder, scratch_root, results_root, s3_bucket, name)
    else:
        # Fallback to backup data location
        backup_name = f"/data/{name}/SPIM.ome.zarr/"
        _process_local_data(backup_name, data_folder, scratch_root, results_root, s3_bucket, name)
    

def _process_local_data(
    data_path: str, 
    data_folder: str, 
    scratch_root: str, 
    results_root: str, 
    s3_bucket: str, 
    name: str
) -> None:
    """
    Process local zarr data for camera alignment.
    
    Parameters
    ----------
    data_path : str
        Path to local zarr data
    data_folder : str
        Data directory path
    scratch_root : str
        Scratch directory path
    results_root : str
        Results directory path  
    s3_bucket : str
        S3 bucket name
    name : str
        Dataset name
    """
    # Calculate affine transforms
    calc_affine(data_path)
    
    # Look for XML file
    xml_path = str(Path(data_path) / 'stitching_rc_spot_channels.xml')
    if Path(xml_path).exists():
        # Apply transforms to XML
        updated_xml_path = apply_affine_to_xml(data_folder, scratch_root, xml_path=xml_path)
        updated_xml_path_forward = apply_affine_to_xml_forward_transform(
            data_folder, scratch_root, xml_path=xml_path
        )
        
        # Save results
        _save_xml_results(updated_xml_path, updated_xml_path_forward, results_root, s3_bucket, name)
        
        # Generate QC plots
        LOGGER.info('*' * 50)
        LOGGER.info(f'Making QC Figures now (Capsule Mode)')
        LOGGER.info('*' * 50)
        try:
            qc_output_dir = results_root + name + "/camera_alignment_qc/"
            make_comprehensive_qc_plots(data_path, scratch_root, output_root=qc_output_dir)
            LOGGER.info(f'QC plots saved to: {qc_output_dir}')
        except Exception as e:
            LOGGER.warning(f"QC plot generation failed in capsule mode: {e}")
            import traceback
            traceback.print_exc()
    else:
        LOGGER.warning(f"No XML file found at {xml_path}. Skipping XML processing.")

def _save_xml_results(
    updated_xml_path: str, 
    updated_xml_path_forward: str, 
    results_root: str, 
    s3_bucket: str, 
    name: str
) -> None:
    """
    Save XML results to local results directory and S3.
    
    Parameters
    ----------
    updated_xml_path : str
        Path to updated XML file
    updated_xml_path_forward : str
        Path to forward transform XML file
    results_root : str
        Results directory path
    s3_bucket : str
        S3 bucket name
    name : str
        Dataset name
    """
    # Copy to results directory
    results_xml_path = results_root + Path(updated_xml_path).name
    results_xml_path_forward = results_root + Path(updated_xml_path_forward).name
    copy_file(updated_xml_path, results_xml_path)
    copy_file(updated_xml_path_forward, results_xml_path_forward)
    
    # Copy to S3
    LOGGER.info('*' * 50)
    LOGGER.info(f'Saving XML files to S3')
    LOGGER.info('*' * 50)
    s3_path_base = f's3://{s3_bucket}/{name}/{TILE_ALIGNMENT_S3_FOLDER_NAME}/'
    copy_file_to_s3(updated_xml_path, s3_path_base + Path(updated_xml_path).name)
    copy_file_to_s3(updated_xml_path_forward, s3_path_base + Path(updated_xml_path_forward).name)
    
    LOGGER.info(f'XML files saved to: {results_root}')
    LOGGER.info(f'XML files uploaded to: {s3_path_base}')
def main(args: Dict[str, Any]) -> None:
    """
    Main entry point for camera alignment processing.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Configuration dictionary containing:
        - pipeline: bool - Whether to use pipeline (S3) or capsule (local) mode
        - skip_alignment: bool - Whether to skip alignment (for certain project types)
        - dataset_name: str - Name of the dataset to process
        - bucket: str - S3 bucket name
        
    Returns
    -------
    None
        Processing results are saved to disk and S3
    """
    # Check if we should skip alignment entirely
    if args.get('skip_alignment', False):
        LOGGER.info("Skipping camera alignment per configuration")
        handle_skip_alignment(args)
        return
    
    # Route to appropriate processing mode
    if args.get('pipeline', True):
        run_camera_alignment_pipeline(args)
    else:
        run_camera_alignment_capsule(args)
            
            
def debug():
    """Debug function for local testing."""
    args = {
        "dataset_name": "HCR_BL6-001_2023-06-19_00-01-00",
        "bucket": "aind-open-data",
        "pipeline": False,  # Use capsule mode for debug
        "skip_alignment": False,  # Run alignment for debug
    }
    main(args)


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
    
    LOGGER.info(f"Found zarr datasets: {zarr_datasets}")
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