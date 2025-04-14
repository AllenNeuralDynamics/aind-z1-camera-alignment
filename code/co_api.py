from codeocean.client import CodeOcean as CodeOceanClient
from codeocean.capsule import Capsules
from typing import List
from codeocean.data_asset import (
    DataAssetAttachParams,
    DataAssetAttachResults,
    DataAssetSearchParams
)
import os
from typing import List, Optional
import time

def attach_data_asset(dataset_ids: List[str], dataset_names: List[str], capsule_id: str = "bdc1c4ea-1d44-46f0-8770-a2dd57f99186"):
    client = CodeOceanClient(
        domain='https://codeocean.allenneuraldynamics.org',
        token=os.environ['CUSTOM_KEY']
    )
    
    #dataset_ids = ["4ca12867-1547-4b1b-8900-595b8ba0a0f2"]  # Replace with actual dataset IDs
    
    # Try the attachment with debug info
    success = safe_attach_datasets(client, capsule_id, dataset_ids, dataset_names)


def attach_datasets_to_capsule(
    client: CodeOceanClient,
    capsule_id: str,
    dataset_ids: List[str],
    custom_mounts: dict = None
) -> List[DataAssetAttachResults]:
    """
    Attaches multiple datasets to a capsule with optional custom mount points.
    
    Parameters
    ----------
    client : CodeOceanClient
        Authenticated CodeOcean client
    capsule_id : str
        ID of the target capsule
    dataset_ids : List[str]
        List of dataset IDs to attach
    custom_mounts : dict, optional
        Dictionary mapping dataset IDs to custom mount points
        
    Returns
    -------
    List[DataAssetAttachResults]
        Results of the attachment operations
    """
    # Prepare attachment parameters
    attach_params = []
    for dataset_id in dataset_ids:
        # Create attachment parameters for each dataset
        params = DataAssetAttachParams(
            id=dataset_id,
            # Use custom mount point if provided, otherwise None (default mount)
            mount=custom_mounts.get(dataset_id) if custom_mounts else None
        )
        attach_params.append(params)
    
    try:
        # Perform the attachment
        results = client.capsules.attach_data_assets(
            capsule_id=capsule_id,
            attach_params=attach_params
        )
        
        # Print results
        for result in results:
            print(f"Attached dataset {result.id} at mount point '{result.mount}'")
            
        return results
        
    except Exception as e:
        print(f"Error attaching datasets to capsule {capsule_id}: {str(e)}")
        raise


def get_codeocean_client() -> CodeOceanClient:
    """Creates and returns an authenticated CodeOcean client"""
    return CodeOceanClient(
        domain='https://codeocean.allenneuraldynamics.org',
        token=os.environ['CUSTOM_KEY']
    )

########################################


def list_data_directory(base_path: str = "/data", max_depth: Optional[int] = 2) -> None:
    """
    Lists all files and folders in the specified directory using os.walk.
    
    Parameters
    ----------
    base_path : str
        Base directory to start listing from (default: "/data")
    max_depth : Optional[int]
        Maximum depth to traverse (None for unlimited)
    """
    try:
        # Ensure base path exists
        if not os.path.exists(base_path):
            print(f"Directory not found: {base_path}")
            return

        print(f"\n📁 {base_path}")
        
        # Walk through directory
        for root, dirs, files in os.walk(base_path):
            # Calculate current depth
            depth = root[len(base_path):].count(os.sep)
            
            # Check if we've exceeded max_depth
            if max_depth is not None and depth > max_depth:
                dirs.clear()  # Clear dirs list to prevent deeper recursion
                continue
            
            # Create indentation based on depth
            indent = "  " * depth
            
            # Get relative path for display
            rel_path = os.path.relpath(root, base_path)
            if rel_path != ".":
                print(f"{indent}📁 {os.path.basename(root)}/")
            
            # Print all files in current directory
            for file in sorted(files):
                print(f"{indent}  📄 {file}")
                
            # Sort directories for consistent output
            dirs.sort()
            
    except Exception as e:
        print(f"Error listing directory {base_path}: {str(e)}")


def verify_capsule_access(client: CodeOceanClient, capsule_id: str) -> bool:
    """
    Verifies that the capsule exists and is accessible.
    
    Returns
    -------
    bool
        True if capsule is accessible, False otherwise
    """
    try:
        # Try to get the capsule details
        capsule = client.capsules.get_capsule(capsule_id)
        print(f"Successfully accessed capsule: {capsule_id}")
        return True
    except Exception as e:
        print(f"Error accessing capsule {capsule_id}: {str(e)}")
        return False


def safe_attach_datasets(
    client: CodeOceanClient,
    capsule_id: str,
    dataset_ids: List[str],
    dataset_names: List[str],
    max_retries: int = 3,
    retry_delay: int = 5
) -> bool:
    """
    Safely attaches datasets with retries and detailed error reporting.
    
    Parameters
    ----------
    client : CodeOceanClient
        Authenticated CodeOcean client
    capsule_id : str
        ID of the target capsule
    dataset_ids : List[str]
        List of dataset IDs to attach
    max_retries : int
        Maximum number of retry attempts
    retry_delay : int
        Delay between retries in seconds
        
    Returns
    -------
    bool
        True if attachment was successful, False otherwise
    """
    # First verify capsule access
    if not verify_capsule_access(client, capsule_id):
        print("Cannot proceed - capsule verification failed")
        return False
        
    # Verify datasets exist
    valid_datasets = []
    for dataset_id in dataset_ids:
        try:
            dataset = client.data_assets.get_data_asset(dataset_id)
            valid_datasets.append(dataset_id)
            print(f"Verified dataset access: {dataset_id}")
        except Exception as e:
            print(f"Error verifying dataset {dataset_id}: {str(e)}")
    
    if not valid_datasets:
        print("No valid datasets to attach")
        return False
    
    # Prepare attachment parameters
    attach_params = []
    for i, dataset_id in enumerate(valid_datasets):
        params = DataAssetAttachParams(
            id=dataset_id,
            mount=f"{dataset_names[i]}"  # Or your preferred mounting scheme
        )
        attach_params.append(params)
    
    # Try to attach with retries
    for attempt in range(max_retries):
        try:
            print(f"\nAttachment attempt {attempt + 1} of {max_retries}")
            
            # Print request details for debugging
            print(f"Making request to: capsules/{capsule_id}/data_assets")
            print(f"With parameters: {[p.to_dict() for p in attach_params]}")
            
            results = client.capsules.attach_data_assets(
                capsule_id=capsule_id,
                attach_params=attach_params
            )
            
            # Verify results
            all_ready = True
            for result in results:
                print(f"\nAttachment result for {result.id}:")
                print(f"  Mount point: {result.mount}")
                print(f"  Ready: {result.ready}")
                #print(f"  Mount state: {result.mount_state}")
                if not result.ready:
                    all_ready = False
            
            if all_ready:
                print("\nAll datasets attached successfully")
                # List the data directory structure after successful attachment
                #print("\nListing /data directory structure:")
                #list_data_directory()
                return True
            else:
                print("\nSome attachments not ready, will retry...")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            continue
    
    print("Failed to attach datasets after all retries")
    return False


# Example usage
def debug_attachment():
    client = CodeOceanClient(
        domain='https://codeocean.allenneuraldynamics.org',
        token=os.environ['CUSTOM_KEY']
    )
    
    capsule_id = "bdc1c4ea-1d44-46f0-8770-a2dd57f99186"
    dataset_ids = ["4ca12867-1547-4b1b-8900-595b8ba0a0f2"]  # Replace with actual dataset IDs
    
    # Try the attachment with debug info
    success = safe_attach_datasets(client, capsule_id, dataset_ids)
    
    if not success:
        print("\nDebugging steps to try:")
        print("1. Verify capsule ID is correct")
        print("2. Check your API token has necessary permissions")
        print("3. Verify all dataset IDs exist and are accessible")
        print("4. Check if the capsule is in a state that allows attachments")
        print("5. Consider any rate limits or timing issues")

