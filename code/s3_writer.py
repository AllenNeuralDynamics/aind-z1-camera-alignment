# from ome_zarr.io import parse_url
import json
from typing import Any, Optional

import s3fs
import logging
import pathlib
from utils import get_code_ocean_cpu_limit


logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


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
    num_cpus = int(get_code_ocean_cpu_limit() or 1)
    
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


def _get_s3_filesystem() -> s3fs.S3FileSystem:
    """Create an ``s3fs`` filesystem configured for Code Ocean usage.

    Returns
    -------
    s3fs.S3FileSystem
        Filesystem instance with connection pooling sized to the available CPUs.
    """
    num_cpus = int(get_code_ocean_cpu_limit() or 1)
    return s3fs.S3FileSystem(
        config_kwargs={
            "max_pool_connections": num_cpus,
            "retries": {
                "total_max_attempts": 1000,
                "mode": "adaptive",
            },
        },
        use_ssl=True,
    )


def read_json_from_s3(s3_location: str) -> Optional[Any]:
    """Read and parse a JSON object stored in S3.

    Parameters
    ----------
    s3_location : str
        S3 path to the JSON file. Accepts ``'s3://bucket/key.json'`` or
        ``'bucket/key.json'``.

    Returns
    -------
    Any or None
        Parsed JSON content (typically a ``dict``), or ``None`` if the object
        does not exist.

    Raises
    ------
    RuntimeError
        If the object exists but cannot be read or parsed.
    """
    s3_path_clean = (
        s3_location.replace("s3://", "") if s3_location.startswith("s3://") else s3_location
    )

    try:
        s3 = _get_s3_filesystem()
        if not s3.exists(s3_path_clean):
            LOGGER.info(f"No existing JSON found at s3://{s3_path_clean}")
            return None

        LOGGER.info(f"Reading JSON from s3://{s3_path_clean}")
        with s3.open(s3_path_clean, "r") as handle:
            return json.load(handle)
    except Exception as e:
        raise RuntimeError(f"Error reading JSON from s3://{s3_path_clean}: {str(e)}")


def write_json_to_s3(data: Any, s3_location: str) -> bool:
    """Serialize a JSON-compatible object and upload it to S3.

    Parameters
    ----------
    data : Any
        JSON-serializable object to upload.
    s3_location : str
        S3 destination path. Accepts ``'s3://bucket/key.json'`` or
        ``'bucket/key.json'``.

    Returns
    -------
    bool
        True if the upload was verified successfully.

    Raises
    ------
    RuntimeError
        If there is an error serializing or uploading the object.
    """
    s3_path_clean = (
        s3_location.replace("s3://", "") if s3_location.startswith("s3://") else s3_location
    )

    LOGGER.info(f"Writing JSON to s3://{s3_path_clean}")

    try:
        s3 = _get_s3_filesystem()
        with s3.open(s3_path_clean, "w") as handle:
            json.dump(data, handle, indent=2)

        if s3.exists(s3_path_clean):
            LOGGER.info(f"Successfully wrote JSON to s3://{s3_path_clean}")
            return True
        LOGGER.error(f"Write verification failed for s3://{s3_path_clean}")
        return False
    except Exception as e:
        raise RuntimeError(f"Error writing JSON to s3://{s3_path_clean}: {str(e)}")

