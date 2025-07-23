import s3fs
import os
from typing import List, Dict, Any
import json
import glob
import pathlib
import logging
import xmltodict
import xml.etree.ElementTree as ET
from collections import OrderedDict

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
    Add a camera alignment affine transform to all tiles that belong 
    to a channel group. 
    
    Parameters
    ----------
    xml_path : str
        Path to the input XML file
    channel_affine : list
        2D affine transform as a list of 6 values
    channel : str
        Channel wavelength to add transforms to (will be used to find tiles)
    
    Returns
    -------
    str
        Path to the updated XML file
    """
    
    # First, parse the XML to find all tiles that belong to this channel
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())
    
    viewsetups = data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"]
    
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
    else: 
        
    
    # Write the updated XML back to file
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        # Write XML declaration
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        # Convert back to XML and write
        xmltodict.unparse(data, f, pretty=True)
    
    logger.info(f"Updated XML saved to: {output_xml_path}")
    return output_xml_path


def update_xml_path_to_camera_alignment(xml_path: str, output_xml_path = None): 
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