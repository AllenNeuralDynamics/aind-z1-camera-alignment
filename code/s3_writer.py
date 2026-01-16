# from ome_zarr.io import parse_url
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
    num_cpus = get_code_ocean_cpu_limit()
    
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
