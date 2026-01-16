import s3fs
import os
import psutil
from typing import List, Dict, Any
import json
import glob
import pathlib
import logging
import xmltodict
from collections import OrderedDict
import shutil


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def get_code_ocean_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
        cfs_quota_us = int(fp.read())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
        cfs_period_us = int(fp.read())
    container_cpus = cfs_quota_us // cfs_period_us
    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus

def copy_file(file_path: str, out_path: str) -> str:
    """
    Copy a file from source path to destination path.
    
    Copies a file from the source location (typically '/scratch/') to the 
    destination location (typically '/results/'). Creates destination 
    directories if they don't exist.
    
    Parameters
    ----------
    file_path : str
        Source file path to copy from
    out_path : str
        Destination file path to copy to
        
    Returns
    -------
    str
        Path to the copied file
        
    Raises
    ------
    FileNotFoundError
        If the source file does not exist
    PermissionError
        If there are insufficient permissions to read source or write destination
    OSError
        If there are other OS-level errors during the copy operation
        
    Notes
    -----
    This function will:
    - Create destination directories if they don't exist
    - Preserve file metadata (timestamps, permissions) when possible
    - Overwrite destination file if it already exists
    
    Examples
    --------
    >>> copy_file('/scratch/results.json', '/results/final_results.json')
    '/results/final_results.json'
    """
    try:
        # Convert to pathlib.Path objects for easier manipulation
        source_path = pathlib.Path(file_path)
        dest_path = pathlib.Path(out_path)
        
        # Check if source file exists
        if not source_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {file_path}")
        
        # Create destination directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy2(source_path, dest_path)
        
        logger.info(f"Successfully copied file from {file_path} to {out_path}")
        return str(dest_path)
        
    except Exception as e:
        logger.error(f"Error copying file from {file_path} to {out_path}: {str(e)}")
        raise

def find_zarr_datasets() -> List[pathlib.Path]:
    """
    Find all zarr datasets in the data directory.
    
    Searches for .ome.zarr files both directly in the data directory and 
    one level deep in subdirectories.
    
    Parameters
    ----------
    None
        
    Returns
    -------
    List[pathlib.Path]
        List of paths to found zarr datasets
        
    Notes
    -----
    Searches for files with '.ome.zarr' extension in:
    - /data directory directly
    - All subdirectories of /data (one level deep)
    
    This function is legacy and may not be used in the current S3-based workflow.
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

def get_project_name() ->str: 
    """
    Load the data_description.json file from the data directory and extract project_name .
    
    Searches for data_description.json in multiple possible locations using glob patterns:
    1. ../data/output_aind_metadata/
    2. ../data/
    3. ../data/{any_subdirectory}/
    
    Returns
    -------
    str
        The project name extracted from the 'project_name' field in data_description.json
        
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
            project_name = config.get('project_name')
            if not project_name:
                raise ValueError("'project_name' field not found in data_description.json")
            logger.info(f"Loaded project_name : {project_name}")
            return project_name
    except Exception as e:
        raise RuntimeError(f"Error loading data_description.json: {str(e)}")


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

#utility to write a new affine transform to the relevant xml field: 

""" The complete list of affine transforms for a tile is in the form: 
  <ViewRegistration timepoint="0" setup="74">
      <ViewTransform type="affine">
        <Name>Stitching Transform</Name>
        <affine>1.0 0.0 0.0 -7.56666561145903 0.0 1.0 0.0 5.039427221176538 0.0 0.0 1.0 3.836497927664709</affine>
      </ViewTransform>
      <ViewTransform type="affine">
        <Name>Translation to Nominal Grid</Name>
        <affine>1.0 0.0 0.0 4560.0 0.0 1.0 0.0 -3648.0 0.0 0.0 1.0 0.0</affine>
      </ViewTransform>
    </ViewRegistration>

Therefore we need the following: 

1. A utility that gets setup_id from "tilename" 
    DONE: get_tile_id_from_name()
2. A utility that finds the correct ViewRegistration given the setup_id
    DONE? get_tile_transform_given_tilename()
3. A utility that adds the new affine transform to the top of the stack, 
    including writing/saving the xml. 
    DONE: add_affine_to_xml()

4. A utility that does this for all tiles (all channels).
    DONE: add_affines_to_channel()

5. modify the XML dataset path to point to 'image_camera_alignment'
    DONE: update_xml_path_to_camera_alignment()

6. A utility to convert a 6x1 2D affine array to a 12x1 3D affine array 
    Done: convert_2D_affine_to_3D_affine

"""


def add_affines_to_channel(xml_path: str, channel_affine: list, channel: str, output_xml_path: str): 
    """
    Add camera alignment affine transform to all tiles that belong to a channel.
    
    Parses the XML to find all tiles belonging to the specified channel and
    adds the same affine transform to each tile.
    
    Parameters
    ----------
    xml_path : str
        Path to the input XML file
    channel_affine : list
        2D affine transform as a list of 6 values [A, B, dX, D, E, dY]
    channel : str
        Channel wavelength to add transforms to (used to find matching tiles)
    output_xml_path : str
        Path where the updated XML file will be saved
        
    Returns
    -------
    str or None
        Path to the updated XML file, or None if no tiles found for the channel
        
    Notes
    -----
    The affine transform is converted from 2D to 3D format before insertion.
    Transform is inserted at the beginning of the transform stack (highest priority).
    """
    # First, parse the XML to find all tiles that belong to this channel
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())
    
    try:
        viewsetups = data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"]
    except KeyError as e:
        logger.error(f"Missing expected key in XML structure: {e}")
        logger.error("Ensure the XML file has the correct structure: 'SpimData -> SequenceDescription -> ViewSetups -> ViewSetup'")
        return None
    
    # Handle case where ViewSetup could be a single dict or list of dicts
    if not isinstance(viewsetups, list):
        viewsetups = [viewsetups]
    
    # Find all tile names that match the specified channel
    matching_tilenames = []
    for viewsetup in viewsetups:
        if str(viewsetup['attributes']['channel']) == str(channel):
            matching_tilenames.append(viewsetup['name'])
    
    if not matching_tilenames:
        logger.warning(f"No tiles found for channel {channel}")
        return None
    
    logger.info(f"Found {len(matching_tilenames)} tiles for channel {channel}: {matching_tilenames}")
    
    # Add the affine transform to each tile in this channel
    updated_xml_path = None
    for tilename in matching_tilenames:
        updated_xml_path = add_affine_to_xml(xml_path, channel_affine, tilename, output_xml_path)
        # Use the updated XML as input for the next iteration
        xml_path = updated_xml_path
    
    return updated_xml_path


def add_affine_to_xml(xml_path: str, channel_affine: list, tilename: str, output_xml_path: str = None): 
    """
    Add a camera alignment affine transform to a specific tile in the XML.
    
    Parses the XML to find the specified tile and adds the affine transform
    at the beginning of its transformation stack.
    
    Parameters
    ----------
    xml_path : str
        Path to the input XML file
    channel_affine : list
        2D affine transform as a list of 6 values [A, B, dX, D, E, dY]
    tilename : str
        Name of the tile to add the transform to
    output_xml_path : str, optional
        Path where the updated XML file will be saved. 
        If None, appends '_cam_align' to input filename
        
    Returns
    -------
    str
        Path to the updated XML file
        
    Notes
    -----
    The 2D affine is converted to 3D format before insertion.
    Transform is named "Camera Alignment Affine" and inserted with highest priority.
    """ 
    """
    Add a camera alignment affine transform to a specific tile in the XML file.
    
    Parameters
    ----------
    xml_path : str
        Path to the input XML file
    channel_affine : list
        2D affine transform as a list of 6 values
    tilename : str
        Name of the tile to add the transform to
    """
    
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())

    affine_3D = convert_2D_affine_to_3D_affine(channel_affine)
    # Convert affine_3D to str
    affine_3D_str = " ".join([str(i) for i in affine_3D])

    tile_number = get_tile_id_from_name(data, tilename)

    # Get the view registration for this tile
    view_registration = data["SpimData"]["ViewRegistrations"]["ViewRegistration"][tile_number]
   
    # Create the new ViewTransform object
    new_view_transform = OrderedDict([
        ('@type', 'affine'),
        ('Name', 'Camera Alignment Affine'),
        ('affine', affine_3D_str)
    ])
    
    # Handle the ViewTransform structure - it can be a single dict or a list
    current_transforms = view_registration.get("ViewTransform", [])
    
    # Ensure we have a list
    if not isinstance(current_transforms, list):
        current_transforms = [current_transforms]
    
    # Insert the new transform at the beginning (highest priority)
    current_transforms.insert(0, new_view_transform)
    
    # Update the view registration
    data["SpimData"]["ViewRegistrations"]["ViewRegistration"][tile_number]["ViewTransform"] = current_transforms

    # Generate output path
    if output_xml_path == None: 
        output_xml_path = xml_path.replace('.xml', '_cam_align.xml')
       
    
    # Write the updated XML back to file
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        # Write XML declaration
        # Convert back to XML and write
        xmltodict.unparse(data, f, pretty=True)
    
    logger.info(f"Updated XML saved to: {output_xml_path}")
    return output_xml_path


def update_xml_path_to_camera_alignment(xml_path: str, output_xml_path = None): 
    """
    Update dataset path in XML to point to camera alignment directory.
    
    Modifies the XML file to change the dataset path from the original location
    to the camera alignment output directory.
    
    Parameters
    ----------
    xml_path : str
        Path to the input XML file to modify
    output_xml_path : str, optional
        Path where the updated XML file will be saved.
        If None, appends '_cam_align' to input filename
        
    Returns
    -------
    str
        Path to the updated XML file
        
    Notes
    -----
    Changes the dataset path to point to 'image_camera_alignment' subdirectory.
    This ensures the XML references the corrected tiles rather than originals.
    """ 
    """
    Update the data pointer to 'image_camera_alignment'
    
    Parameters
    ----------
    xml_path : str
        Path to the input XML file
    output_xml_path : str, optional
        Path for the output XML file. If None, will append '_camera_alignment' to input filename
        
    Returns
    -------
    str
        Path to the updated XML file
    
    """
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())

    dataset_path = extract_dataset_path(xml_path)

    logger.info(f'Dataset path {dataset_path}')

    if 'SPIM' in dataset_path:
        updated_path = dataset_path.replace('SPIM', 'image_camera_alignment')
    elif 'image_radial_correction' in dataset_path: 
        updated_path = dataset_path.replace('image_radial_correction', 'image_camera_alignment') 
    else: 
        updated_path = dataset_path + '/image_camera_alignment/'
        logger.warning(f"No SPIM found in path, appending: {dataset_path} -> {updated_path}")
        
    
    # Update the XML data
    data["SpimData"]["SequenceDescription"]["ImageLoader"]["zarr"]["#text"] = updated_path
    
    logger.info(f"Updated dataset path: {updated_path}")
    
    # Generate output path if not provided
    if output_xml_path is None:
        output_xml_path = xml_path.replace('.xml', '_camera_alignment.xml')
    
    # Write the updated XML
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        xmltodict.unparse(data, f, pretty=True)
    
    logger.info(f"Updated XML with camera alignment path saved to: {output_xml_path}")
    return output_xml_path  


def convert_2D_affine_to_3D_affine(affine:list, dZ = 0)-> list: 
    """
    Converts a 2D affine to a 3D affine transform in
    list format. 
    2D affine has format 
    1 0 dX
    0 1 dY
    0 0 1 (<- we ignore this line, it 
            is assumed to be there)

    3D affine has the format 
    1 0 0 dX
    0 1 0 dY
    0 0 1 dZ
    0 0 0 1 (<- we ignore this line too)

    Parameters: 
    -----------
    affine: list
        the 2D affine to be converted
    
    Returns: 
    --------
    affine_3D: list
        3D affine to be saved to xml

    """
    (A, B, dX, D, E, dY) = affine

    affine_3D = [A, B , 0, dX, D, E, 0, dY, 0, 0, 1, dZ]
    return affine_3D


def extract_dataset_path(xml_path: str) -> dict[int, str]:
    """
    Parses BDV xml and outputs map of setup_id -> tile path.

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted from BigStitcher.

    Returns
    ------------------------
    dict[int, str]:
        Dictionary of tile ids to tile paths.

    """

    view_paths: dict[int, str] = {}
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())

    dataset_path = data["SpimData"]["SequenceDescription"]["ImageLoader"]["zarr"]
        

    return dataset_path["#text"]

def extract_tile_paths(xml_path: str) -> dict[int, str]:
    """
    Parses BDV xml and outputs map of setup_id -> tile path.

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted from BigStitcher.

    Returns
    ------------------------
    dict[int, str]:
        Dictionary of tile ids to tile paths.

    """

    view_paths: dict[int, str] = {}
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())

    if not isinstance(data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"][
            "zgroup"
        ], list):
        view_paths = data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"][
            "zgroup"
        ]['path']
    else:
        for id, zgroup in enumerate(
            data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"][
                "zgroup"
            ]
        ):
            view_paths[int(id)] = zgroup["path"]

    return view_paths


def extract_tile_vox_size(xml_path: str) -> tuple[float, float, float]:
    """
    Parses BDV xml and output 3-ple of voxel sizes: (x, y, z)

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted by BigStitcher.

    Returns
    ------------------------
    tuple[float, float, float]:
        Tuple containing voxel sizes.

    """

    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())

    if isinstance(data["SpimData"]["SequenceDescription"][
        "ViewSetups"
    ]["ViewSetup"], list):
        first_tile_metadata = data["SpimData"]["SequenceDescription"][
            "ViewSetups"
        ]["ViewSetup"][0]
    else:
        first_tile_metadata = data["SpimData"]["SequenceDescription"][
        "ViewSetups"
    ]["ViewSetup"]
    vox_sizes: str = first_tile_metadata["voxelSize"]["size"]
    return tuple(float(val) for val in vox_sizes.split(" "))


def extract_tile_transforms(xml_path: str) -> dict[int, list[dict]]:
    """
    Parses BDV xml and outputs map of setup_id -> list of transformations
    Output dictionary maps view number to list of {'@type', 'Name', 'affine'}
    where 'affine' contains the transform as string of 12 floats.

    Matrices are listed in the order of forward execution.

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted by BigStitcher.

    Returns
    ------------------------
    dict[int, list[dict]]
        Dictionary of tile ids to transform list. List entries described above.

    """

    view_transforms: dict[int, list[dict]] = {}
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())
    
    view_registration = data["SpimData"]["ViewRegistrations"]["ViewRegistration"]
    if not isinstance(view_registration, list):
        tfm_stack = view_registration["ViewTransform"]
        
        if type(tfm_stack) is not list:
            tfm_stack = [tfm_stack]
        view_transforms[int(view_registration["@setup"])] = tfm_stack
    else:
        for view_reg in view_registration:
            tfm_stack = view_reg["ViewTransform"]
            if type(tfm_stack) is not list:
                tfm_stack = [tfm_stack]
            view_transforms[int(view_reg["@setup"])] = tfm_stack

    view_transforms = {
        view: tfs[::-1] for view, tfs in view_transforms.items()
    }

    return view_transforms

def get_tile_id_from_name(data:dict, tilename):
    """
    Extract tile ID from tilename using parsed XML data.
    
    Finds the viewsetup with matching tilename and returns its tile number.
    
    Parameters
    ----------
    data : dict
        Parsed XML data as dictionary
    tilename : str
        Name of the tile to find ID for
        
    Returns
    -------
    int
        Tile number/ID corresponding to the tilename
        
    Notes
    -----
    Searches through ViewSetups to find matching tile name, then extracts
    the tile attribute which serves as the unique tile identifier.
    """
    
    #find viewsetup with matching tilename
    viewsetups = data["SpimData"]["SequenceDescription"][
            "ViewSetups"
        ]["ViewSetup"]
    matching_viewsetup=[v for v in viewsetups if v['name']==tilename]

    #get tile_number from this viewsetup
    matching_tile_number = matching_viewsetup[0]['attributes']['tile']

    return int(matching_tile_number)
    


def get_tile_transform_given_tilename(xml_path: str, tilename:str):
    """
    Extract transformation matrix for a specific tile from XML file.
    
    Finds the tile by name and returns its current transformation matrix
    as a list of 12 float values.
    
    Parameters
    ----------
    xml_path : str
        Path to the XML file containing tile transformations
    tilename : str
        Name of the tile to extract transform for
        
    Returns
    -------
    list[float]
        List of 12 float values representing the 3D affine transformation matrix
        
    Notes
    -----
    Uses tile ID to index into ViewRegistrations and extracts the 'affine' field
    from the ViewTransform, which contains space-separated float values.
    """
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())
    
    tile_number = get_tile_id_from_name(data, tilename)

    #use tile_number to index view_registrations
    view_registrations = data["SpimData"]["ViewRegistrations"]["ViewRegistration"]

    view_registration = view_registrations[tile_number]

    transform = view_registration['ViewTransform']['affine']

    nums = [float(val) for val in transform.split(" ")]

    return nums

def get_channel_for_tilename(xml_path:str, tilename:str) -> int:
    """
    Get the channel number for a specific tile from XML metadata.
    
    Parses the XML file to find the viewsetup matching the tilename and
    extracts its channel attribute.
    
    Parameters
    ----------
    xml_path : str
        Path to the XML file containing viewsetup information
    tilename : str
        Name of the tile to get channel for
        
    Returns
    -------
    int
        Channel number associated with the tile
        
    Notes
    -----
    Searches through ViewSetups to find matching tile name, then extracts
    the channel attribute from the tile's attributes.
    """
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())
    
    viewsetups = data["SpimData"]["SequenceDescription"][
            "ViewSetups"
        ]["ViewSetup"]
    matching_viewsetup=[v for v in viewsetups if v['name']==tilename]
    channel = matching_viewsetup[0]['attributes']['channel']
    return int(channel)

def read_channels_from_xml(xml_path:str) -> list[int]:
    """
    Read the XML attribute named "channel" and return
     a list of the names of the channels there. 
    
    Parameters
    ---------
    xml_path: str

    Returns: 
    channels: list[int]
        Unique channels in the xml
    """
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())
    
    channel_attributes = data['SpimData']['SequenceDescription']['ViewSetups']['Attributes'][1]
    if channel_attributes['@name']=='channel':
        channels = list(set([int(i['name']) for i in channel_attributes['Channel']]))
        return channels
    else: 
        return None


def get_channel_wavelength_from_single_channel_digit(single_channel_digit):
    """
    Convert single channel digit to wavelength string.
    
    Parameters
    ----------
    single_channel_digit : int or str
        Single digit or string representing the channel (0-4)
        
    Returns
    -------
    str or int
        Wavelength string corresponding to the channel digit, or -1 if invalid
        
    Examples
    --------
    >>> get_channel_wavelength_from_single_channel_digit('1')
    '561'
    >>> get_channel_wavelength_from_single_channel_digit(2)
    '488'
    """
    if single_channel_digit==  0 or single_channel_digit== '0': 
        return '405'
    elif single_channel_digit== 1 or single_channel_digit== '1': 
        return '561'
    elif single_channel_digit== 2 or single_channel_digit== '2': 
        return '488'
    elif single_channel_digit== 3 or single_channel_digit== '3': 
        return '647'
    elif single_channel_digit== 4 or single_channel_digit== '4': 
        return '515'
    else:  return -1

def get_digit_from_channel_wavelength(wavelength):
    """
    Convert wavelength string to single channel digit.
    
    Parameters
    ----------
    wavelength : str
        Wavelength string (e.g., '405', '561', '488', '647', '638', '515')
        
    Returns
    -------
    str or int
        Single digit string corresponding to the wavelength, or -1 if not found
        
    Examples
    --------
    >>> get_digit_from_channel_wavelength('561')
    '1'
    >>> get_digit_from_channel_wavelength('488') 
    '2'
    """
    lookup = {
        '405':'0', 
        '561':'1', 
        '488':'2',
        '647':'3',
        '638':'3',
        '515':'4'
    }
    try:
        digit = lookup[wavelength]
    except:
        digit = -1
    return digit

def get_list_of_channels(data_loc):
    """
    Get list of unique channels from zarr tiles in local or S3 location.
    
    Parameters
    ----------
    data_loc : str
        Path to directory containing zarr tiles, or S3 path
        
    Returns
    -------
    list[str]
        List of unique channel identifiers found in tile filenames
        
    Notes
    -----
    Automatically detects S3 paths and delegates to get_list_of_channels_s3.
    Expects tile filenames in format: 'tile_x_####_y_####_z_####_ch_XXX.zarr'
    """
    list_of_tiles = list(glob.glob(f'{data_loc}/*.zarr'))

    if "s3" in data_loc:
        return get_list_of_channels_s3(data_loc)
    
    # find unique channels in the list of tiles
    channels = []
    for tile in list_of_tiles:
        #form is tile_x_0000_y_0000_z_0000_ch_405.zarr
        channel = tile.split('_')[-1].split('.')[0]
        if channel not in channels:
            channels.append(channel)    
    return channels

def get_list_of_channels_s3(data_loc):
    """
    Get list of unique channels from zarr tiles in S3 location.
    
    Parameters
    ----------
    data_loc : str
        S3 path to directory containing zarr tiles
        
    Returns
    -------
    list[str]
        List of unique channel identifiers found in S3 tile filenames
        
    Notes
    -----
    Expects tile filenames in format: 'tile_x_####_y_####_z_####_ch_XXX.zarr'
    """
    list_of_tiles = list_zarr_tiles_from_s3(data_loc)
    # find unique channels in the list of tiles
    channels = []
    for tile in list_of_tiles:
        #form is tile_x_0000_y_0000_z_0000_ch_405.zarr
        channel = tile.split('_')[-1].split('.')[0]
        if channel not in channels:
            channels.append(channel)    
    return channels

def make_pairs_of_channels(channels):
    """
    Choose pairs of channels to align based on spectral overlap.
    
    Creates sequential pairs of channels that are close in wavelength to each other,
    which improves alignment quality due to spectral similarity. Skips pairs where
    the expected bleedthrough between channels is <= 1.
    
    Parameters
    ----------
    channels : list[str]
        List of channel wavelengths in the dataset
        
    Returns
    -------
    list[list[str]]
        List of pairs of channels to align, where each pair is [channel1, channel2]
        Only includes pairs with significant spectral bleedthrough (> 1)
        
    Examples
    --------
    >>> make_pairs_of_channels(['405', '488', '561', '647'])
    [['405', '488'], ['488', '561']]  # May skip some pairs based on bleedthrough
        
    Notes
    -----
    Channels are first sorted to ensure consistent pairing order.
    Each adjacent pair in the sorted list is evaluated for bleedthrough significance.
    Pairs with bleedthrough values <= 1 in both directions are skipped.
    """
    #simplify to be just this list of channels: [[488,514],[514,561],[561,594],[594,638]]
    approved_list_of_channel_pairs = [['488','514'],['514','561'],['561','594'],['594','638']]
    #sort the channels
    channels.sort()
    
    #initialize list of pairs
    pairs = []
    
    #iterate through the channels
    for i in range(len(channels)-1):
        #get the current channel
        channel = channels[i]
        
        #get the next channel
        next_channel = channels[i+1]
        
        #append the pair to the list of pairs
        proposed_pair = [channel, next_channel]
        if proposed_pair in approved_list_of_channel_pairs:
            pairs.append(proposed_pair)

        
    return pairs

def make_pairs_of_channels_without_bleedthrough_checking(channels):
    """
    Choose pairs of channels to align based on spectral overlap.
    
    Creates sequential pairs of channels that are close in wavelength to each other,
    which improves alignment quality due to spectral similarity.
    
    Parameters
    ----------
    channels : list[str]
        List of channel wavelengths in the dataset
        
    Returns
    -------
    list[list[str]]
        List of pairs of channels to align, where each pair is [channel1, channel2]
        
    Examples
    --------
    >>> make_pairs_of_channels(['405', '488', '561', '647'])
    [['405', '488'], ['488', '561'], ['561', '647']]
        
    Notes
    -----
    Channels are first sorted to ensure consistent pairing order.
    Each adjacent pair in the sorted list becomes an alignment pair.
    """

    #sort the channels
    channels.sort()
    
    #initialize list of pairs
    pairs = []
    
    #iterate through the channels
    for i in range(len(channels)-1):
        #get the current channel
        channel = channels[i]
        
        #get the next channel
        next_channel = channels[i+1]
        
        #append the pair to the list of pairs
        pairs.append([channel, next_channel])
        
    return pairs

def make_pairs_of_channels_with_reference(channels, reference_channel):
    """
    Create pairs of channels to align with a reference channel that never gets transformed.
    
    Creates a spanning tree with the reference channel as the root, where each 
    non-reference channel is paired with a channel that's already aligned.
    This ensures the reference channel maintains its original coordinates.
    
    Parameters
    ----------
    channels : list[str]
        List of channel wavelengths in the dataset
    reference_channel : str
        Channel to use as reference (no transforms applied to this channel)
        
    Returns
    -------
    list[list[str]]
        List of pairs where first element is channel to transform, 
        second element is the reference/already-aligned channel
        
    Raises
    ------
    ValueError
        If reference_channel is not found in the channels list
        
    Examples
    --------
    >>> make_pairs_of_channels_with_reference(['405', '488', '561', '647'], '647')
    [['561', '647'], ['488', '561'], ['405', '488']]
        
    Notes
    -----
    Uses wavelength distance as a heuristic for spectral similarity when choosing
    which already-aligned channel to pair with each unaligned channel.
    """
    if reference_channel not in channels:
        raise ValueError(f"Reference channel '{reference_channel}' not found in channels: {channels}")
    
    # Sort channels for consistent behavior
    sorted_channels = sorted(channels)
    
    # Remove reference channel from the list of channels that need transforms
    other_channels = [ch for ch in sorted_channels if ch != reference_channel]
    
    if len(other_channels) == 0:
        return []  # Only one channel, no pairs needed
    
    pairs = []
    aligned_channels = {reference_channel}  # Start with reference channel as "aligned"
    
    # For each channel that needs alignment, pair it with the closest aligned channel
    while other_channels:
        best_pair = None
        best_distance = float('inf')
        
        # Find the best channel to align next (closest to an already aligned channel)
        for unaligned_ch in other_channels:
            for aligned_ch in aligned_channels:
                # Use wavelength distance as a heuristic for spectral similarity
                distance = abs(int(unaligned_ch) - int(aligned_ch))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (unaligned_ch, aligned_ch)
        
        if best_pair:
            unaligned_ch, aligned_ch = best_pair
            pairs.append([unaligned_ch, aligned_ch])  # Order: [channel_to_transform, reference_channel]
            aligned_channels.add(unaligned_ch)
            other_channels.remove(unaligned_ch)
        else:
            # Fallback: just pair with reference channel
            ch = other_channels.pop(0)
            pairs.append([ch, reference_channel])
            aligned_channels.add(ch)
    
    return pairs

def create_tile_number_dict(tilenames: list) -> dict:
    """
    Create a dictionary mapping tilenames to tile numbers based on raster scanning order.
    
    Assigns sequential numbers to tiles based on their X, Y, Z coordinates in 
    raster scanning order (X fastest, then Y, then Z).
    
    Parameters
    ----------
    tilenames : list[str]
        List of unsorted tilenames containing X, Y, Z coordinates
        Expected format: 'tile_X_####_Y_####_Z_####_ch_XXX.zarr'
        
    Returns
    -------
    dict[str, int]
        Dictionary mapping tilenames to tile numbers (0-indexed)
        
    Notes
    -----
    Only processes tiles from the first channel found to avoid duplicates.
    Tiles are sorted by (X, Y, Z) coordinates before numbering.
    """
    # Extract coordinates and create a list of (tilename, x, y, z) tuples
    tile_info = []
    #first channel will be the reference channel 
    first_channel = tilenames[0].split('_')[-1].split('.')[0]

    for tilename in tilenames:

        channel = tilename.split('_')[-1].split('.')[0]
        if channel == first_channel:
            parts = tilename.split("_")
            x, y, z = parts[2], parts[4], parts[6]
            tile_info.append((tilename, x, y, z))
    
    # Find the maximum dimensions
    # max_x = max(tile[1] for tile in tile_info)
    # max_y = max(tile[2] for tile in tile_info)
    # max_z = max(tile[3] for tile in tile_info)
    
    # Sort tiles based on raster scanning order
    sorted_tiles = sorted(tile_info, key=lambda t: (t[1], t[2], t[3]))
    
    # Create the dictionary mapping tilenames to tile numbers
    tile_number_dict = {tilename: i for i, (tilename, _, _, _) in enumerate(sorted_tiles)}
    
    return tile_number_dict