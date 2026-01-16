# determine out affine transforms based on image data using ransac
from glob import glob
from skimage.feature import blob_dog, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform
from multiprocessing import Pool, cpu_count
from os.path import basename as bn
import dask.array as da
import json
import numpy as np
from numpy.linalg import norm
import numpy.linalg
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import (
    list_zarr_tiles_from_s3,
    get_list_of_channels,
    get_channel_wavelength_from_single_channel_digit,
    make_pairs_of_channels,
    create_tile_number_dict,
)

nodes = cpu_count()-1
dot_num, dot_threshold, min_sample = 10000, 10, 5
FRACTION_OF_DOTS_PER_PLANE_THRESHOLD = 0.4
THICKNESS, SPACING = 40, 4
max_tiles = 20
pyramid_level = '0'

def save_tile_metrics(metrics_dict, results_root):
    """
    Save the metrics dictionary to a JSON file.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing tile metrics data
    results_root : str
        Root directory path where the JSON file will be saved
        
    Returns
    -------
    None
    """
    with open(results_root + '/tile_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, cls=NumpyArrayEncoder)

class NumpyArrayEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def calc_affine(root: str, results_root: str = '/scratch/', qc_root = '/results/tile_affine_qc', reference_channel = None):
    """
    Calculate affine transformations between camera channels for alignment.
    
    Performs interest point detection on individual tiles and finds pairs of matched 
    points between spectrally neighboring channels. Calculates 2D affine transforms 
    using RANSAC and computes global weighted averages based on inlier counts.
    
    Parameters
    ----------
    root : str
        Path to the root directory containing zarr tiles, or S3 path
    results_root : str, optional
        Root directory path for saving results, by default '/scratch/'
    qc_root : str, optional  
        Root directory path for saving QC plots and metrics, by default '/results/tile_affine_qc'
    reference_channel : str, optional
        Channel to use as reference (no transforms applied), by default None
        If None, uses the longest wavelength channel
        
    Returns
    -------
    None
        Results are saved to files in results_root and qc_root directories
        
    Raises
    ------
    ValueError
        If reference_channel is specified but not found in available channels
        
    Notes
    -----
    This function:
    1. Detects interest points in tiles using DoG blob detection
    2. Matches descriptors between neighboring channels  
    3. Uses RANSAC to fit affine transforms
    4. Computes weighted averages of transforms across tiles
    5. Saves transform matrices and QC metrics
    """
    Path(qc_root).mkdir(exist_ok=True)
    list_of_channels = get_list_of_channels(root)
    #print(f'list of channels {list_of_channels}')
    
    if len(list_of_channels[0])!=3:
        list_of_channels = [get_channel_wavelength_from_single_channel_digit(cam) for cam in list_of_channels] 

    if '405' in list_of_channels:
        list_of_channels.remove('405')

    # Determine reference channel
    if reference_channel is None:
        # Default behavior: use longest wavelength (current behavior)
        sorted_channels = sorted(list_of_channels)
        reference_channel = sorted_channels[-1]
        print(f"No reference channel specified. Using default: {reference_channel}")
    elif reference_channel not in list_of_channels:
        raise ValueError(f"Reference channel '{reference_channel}' not found in available channels: {list_of_channels}")
    
    # print(f"Using reference channel: {reference_channel}")

    # Create pairs of channels with reference channel consideration
    # new_pairs_of_channels = make_pairs_of_channels_with_reference(list_of_channels, reference_channel)
    # print(f' new pairs of channels {new_pairs_of_channels}')
    pairs_of_channels = make_pairs_of_channels(list_of_channels)
    # print(f' old pairs of channels {pairs_of_channels}')
    # assert pairs_of_channels == bu_pairs_of_channels

    keep_cam = {x:'' for x in sum(pairs_of_channels, [])}
    
    pairs_of_channels.reverse()
    print(f'pairs of channels = {pairs_of_channels}')

    list_of_tiles = list(glob(f'{root}/*.zarr'))
    if len(list_of_tiles)==0: 
        list_of_tiles = list_zarr_tiles_from_s3(root)
        
        if len(list_of_tiles) == 0:
            raise ValueError(f"No zarr tiles found in root directory: {root}")
    tile_number_dict = create_tile_number_dict(list_of_tiles)
    max_pos = max(tile_number_dict.values()) 

    # Dictionary to store metrics for each tile and channel pair
    tile_metrics = {}
    
    # Dictionary to store image data and corresponding tile info
    Is = {}
    tile_info = {}  # To keep track of which points came from which tiles
    finalM = {}

    # Load image data and track tile information
    for fn in list_of_tiles:
        #change the logic for tilename paths
        cam = bn(fn.split(".")[0]).split('_')[-1]
        if len(cam)!= 3:
            cam = get_channel_wavelength_from_single_channel_digit(cam)
        if cam not in keep_cam: continue

        z = da.from_zarr(fn, pyramid_level).shape[2]
        pixels = da.from_zarr(fn, pyramid_level).shape[3]

        if cam not in Is: 
            Is[cam] = []
            tile_info[cam] = []
            
        # Extract coordinates from filename
        coords = Path(fn).stem[0:25]
        
        # Store image data and corresponding tile info
        Is[cam].append(da.from_zarr(fn, pyramid_level)[0,0,list(range((z-THICKNESS)//2, (z+THICKNESS)//2, SPACING)),...])
        tile_info[cam].append(coords)

    # Process images to get points while maintaining tile association
    points_by_tile = {}
    for k in Is.keys():
        print('finding spots for', k)
        points_by_tile[k] = {}
        
        # Process each tile's worth of planes
        for tile_idx, tile_planes in enumerate(Is[k]):
            tile_coord = tile_info[k][tile_idx]
            # Get points for all planes in this tile
            points = Pool(nodes).map(getTop, np.array(tile_planes).reshape((-1, pixels, pixels)))
            points_by_tile[k][tile_coord] = points

    # Process each channel pair
    for c1, c2 in pairs_of_channels:
        tile_metrics[f"{c1}_{c2}"] = {}
        
        # Process each tile
        common_tiles = set(points_by_tile[c1].keys()) & set(points_by_tile[c2].keys())
        
        all_affs = []  # Store all affines for weighted average calculation
        all_inliers = []  # Store all inlier counts
        
        for tile_coord in common_tiles:
            points_c1 = points_by_tile[c1][tile_coord]
            points_c2 = points_by_tile[c2][tile_coord]
            
            # Initialize metrics for this tile
            tile_metrics[f"{c1}_{c2}"][tile_coord] = {
                f"points_{c1}": sum(len(p) for p in points_c1),
                f"points_{c2}": sum(len(p) for p in points_c2)
            }
            
            # Convert points to integer type
            points_c1 = [p.astype(int) for p in points_c1]
            points_c2 = [p.astype(int) for p in points_c2]
            
            # Apply existing transform if necessary
            if c2 in finalM.keys():
                aff_to_apply = finalM[c2]
                points_c2 = [apply_affine_transform_to_spots(p, np.linalg.inv(aff_to_apply)) for p in points_c2]
            
            # Find model for this tile
            # use plane with the highest number of inliers 
            # for i, _plane in enumerate(list(range((z-thickness)//2, (z+thickness)//2, spacing))):
            #     model_result = find_model((points_c2[i], points_c1[i]))  

            model_result = find_model((points_c2[0], points_c1[0]))  # Using first plane for now
            
            if model_result[0] is not None:
                aff, num_matches, num_inliers = model_result
                if num_inliers is None or num_inliers==0:
                    aff = np.eye(2,3)

                all_affs.append(aff)
                all_inliers.append(num_inliers)
                
                # Store the metrics for this tile
                tile_metrics[f"{c1}_{c2}"][tile_coord].update({
                    "matched_points": num_matches,
                    "num_inliers": num_inliers,
                    "affine_transform": np.append(aff,[0,0,1]).reshape((3,3))
                })
        
        if all_affs:
            # Calculate weighted average affine
            inliers = np.array(all_inliers)
            if inliers.sum() > 0: 
                weights = inliers / inliers.sum()
                weighted_affine = np.append(np.array([aff*w for aff, w in zip(all_affs, weights)]).sum(0),[0,0,1]).reshape((3,3))
            else: 
                weighted_affine = np.array(np.eye(3))
            finalM[c1] = weighted_affine
            
            # Calculate difference from average for each tile
            for tile_coord in common_tiles:
                if "affine_transform" in tile_metrics[f"{c1}_{c2}"][tile_coord]:
                    tile_affine = tile_metrics[f"{c1}_{c2}"][tile_coord]["affine_transform"]
                    diff_from_average = numpy.linalg.norm(tile_affine - weighted_affine)
                    tile_metrics[f"{c1}_{c2}"][tile_coord]["diff_from_average"] = diff_from_average

    # add default condition for no pairs of channels 
    if len(pairs_of_channels) == 0: 
        for ch in list_of_channels: 
            finalM[ch] = np.array([[1,0,0], [0,1,0],[0,0,1]])
    # Save the original affine transforms
    with open(results_root+'updated.M.txt', 'w') as f:
        for e in sorted(finalM.keys()):
            print(e+'\t'+'\t'.join(list(map(str,np.round(finalM[e][:-1].flatten(),6)))), file=f)
            print(e+'\t'+'\t'.join(list(map(str,np.round(finalM[e][:-1].flatten(),6)))))
    
    # Save the detailed metrics
    save_tile_metrics(tile_metrics, qc_root)
    metrics_file = "tile_metrics.json"
    metrics_loc = Path(qc_root).joinpath(metrics_file)
    visualize_tile_metrics(metrics_loc, qc_root)

def extract_coordinates(tile_name):
    """
    Extract X, Y coordinates from tile name.
    
    Parameters
    ----------
    tile_name : str
        Tile name in format 'Tile_X_####_Y_####_...'
        
    Returns
    -------
    tuple[int, int]
        X and Y coordinates as integers
        
    Examples
    --------
    >>> extract_coordinates('Tile_X_0007_Y_0003_Z_0000')
    (7, 3)
    """
    parts = tile_name.split('_')
    x = int(parts[2])
    y = int(parts[4])
    return x, y

def get_grid_dimensions(tile_name):
    """
    Extract grid dimensions from a tile name.
    
    Parameters
    ----------
    tile_name : str
        Tile name in format 'Tile_X_####_Y_####_Z_####'
        
    Returns
    -------
    tuple[int, int]
        Grid dimensions as (width, height) where width is max_x+1 and height is max_y+1
        
    Examples
    --------
    >>> get_grid_dimensions('Tile_X_0007_Y_0003_Z_0000')
    (8, 4)
    """
    parts = tile_name.split('_')
    max_x = int(max([parts[i+1] for i, part in enumerate(parts) if part == 'X']))
    max_y = int(max([parts[i+1] for i, part in enumerate(parts) if part == 'Y']))
    return max_x + 1, max_y + 1

def create_metric_grid(metrics_data, channel_pair, metric_name):
    """
    Create a grid of metric values using maximum dimensions from tile names.
    
    Parameters
    ----------
    metrics_data : dict
        Dictionary containing metrics data for all channel pairs and tiles
    channel_pair : str
        String identifying the channel pair (e.g., '488_561')
    metric_name : str
        Name of the metric to extract and grid
        
    Returns
    -------
    numpy.ndarray
        2D array with metric values positioned according to tile coordinates
        Missing values are filled with np.nan
        
    Notes
    -----
    The grid is indexed as [y, x] to match image conventions.
    """
    # Get all tile names across all channel pairs
    all_tile_names = []
    for channel_data in metrics_data.values():
        all_tile_names.extend(channel_data.keys())
    
    # Get maximum grid dimensions from all tile names
    max_dims = (0, 0)
    for tile_name in all_tile_names:
        dims = get_grid_dimensions(tile_name)
        max_dims = (max(max_dims[0], dims[0]), max(max_dims[1], dims[1]))
    
    grid = np.full(max_dims[::-1], np.nan)  # Reverse dims for [y,x] indexing
    
    for tile_name, tile_data in metrics_data[channel_pair].items():
        if metric_name in tile_data:
            x, y = extract_coordinates(tile_name)
            grid[y, x] = tile_data[metric_name]
    
    return grid

def plot_metric_heatmap(grid, title, filename, results_folder, vmin=None, vmax=None, cmap='viridis'):
    """
    Create and save a heatmap plot for a metric.
    
    Parameters
    ----------
    grid : numpy.ndarray
        2D array containing metric values to plot
    title : str
        Title for the plot
    filename : str
        Name of the file to save (including extension)
    results_folder : str or pathlib.Path
        Directory path where the plot will be saved
    vmin : float, optional
        Minimum value for color scale, by default None
    vmax : float, optional  
        Maximum value for color scale, by default None
    cmap : str, optional
        Colormap name, by default 'viridis'
        
    Returns
    -------
    None
        Plot is saved to file
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(grid, 
                cmap=cmap,
                annot=True, 
                fmt='.2f',
                mask=np.isnan(grid),
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': title})
    
    plt.title(title)
    plt.xlabel('Tile X Position')
    plt.ylabel('Tile Y Position')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(str(Path(results_folder) / filename))
    plt.close()

def visualize_tile_metrics(metrics_file, results_folder):
    """
    Create visualizations for tile metrics.
    
    Generates heatmap plots for various metrics across the tile grid including
    matched points, RANSAC inliers, differences from average transforms, and
    point counts per channel.
    
    Parameters
    ----------
    metrics_file : str or pathlib.Path
        Path to the JSON file containing tile metrics
    results_folder : str or pathlib.Path
        Directory where visualization plots will be saved
        
    Returns
    -------
    None
        Plots are saved to files in results_folder
        
    Notes
    -----
    Creates the following plots for each channel pair:
    - Number of matched points heatmap
    - Number of RANSAC inliers heatmap  
    - Difference from average transform heatmap
    - Point counts per channel heatmaps
    """
    # Load metrics data
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
    
    
    # Create plots for each channel pair and metric
    for channel_pair in metrics_data.keys():
        print(f"Processing channel pair: {channel_pair}")
        
        # Plot number of matched points
        matched_points_grid = create_metric_grid(metrics_data, channel_pair, 'matched_points')
        plot_metric_heatmap(matched_points_grid,
                          f'Number of Matched Points\n{channel_pair}',
                          f'matched_points_{channel_pair}.png',
                          results_folder,
                          vmin=0)
        
        # Plot number of inliers
        inliers_grid = create_metric_grid(metrics_data, channel_pair, 'num_inliers')
        plot_metric_heatmap(inliers_grid,
                          f'Number of RANSAC Inliers\n{channel_pair}',
                          f'num_inliers_{channel_pair}.png',
                          results_folder,
                          vmin=0)
        
        # Plot difference from average
        diff_grid = create_metric_grid(metrics_data, channel_pair, 'diff_from_average')
        plot_metric_heatmap(diff_grid,
                          f'Difference from Average Transform\n{channel_pair}',
                          f'diff_from_average_{channel_pair}.png',
                          results_folder,
                          vmin=0,
                          cmap='YlOrRd')
        
        # Plot points in each channel
        for ch in channel_pair.split('_'):
            points_grid = create_metric_grid(metrics_data, channel_pair, f'points_{ch}')
            plot_metric_heatmap(points_grid,
                              f'Number of Points in Channel {ch}\n',
                              f'points_{ch}.png',
                              results_folder,
                              vmin=0)

def main():
    root = '/root/capsule/data/HCR_759560_2024-10-17_16-00-00/SPIM.ome.zarr/'
    results_root = '/root/capsule/scratch/HCR_759560_2024-10-17_16-00-00/'
    # calc_affine(root, results_root)
    metrics_file = "tile_metrics.json"
    metrics_loc = Path(results_root).joinpath(metrics_file)

    visualize_tile_metrics(metrics_loc, results_root)






def find_model(input):
    """
    Find affine transformation model between two sets of points using RANSAC.
    
    Matches descriptors between two point sets and uses RANSAC to fit a robust
    affine transformation model, filtering out outliers.
    
    Parameters
    ----------
    input : tuple[numpy.ndarray, numpy.ndarray]
        Tuple containing (points_A, points_B) where each is an array of (y, x) coordinates
        
    Returns
    -------
    tuple[numpy.ndarray or None, int or None, int or None]
        - Affine transformation parameters (2x3 matrix flattened to exclude last row)
        - Number of initial matched correspondences  
        - Number of inlier correspondences after RANSAC
        Returns (None, None, None) if insufficient matches or RANSAC fails
        
    Notes
    -----
    Uses descriptor matching with max_distance=6 and max_ratio=0.8.
    Requires at least min_sample correspondences to proceed with RANSAC.
    RANSAC uses residual_threshold=1 and max_trials=5000.
    """
    A, B = input

    #verify that both A, B have spots in them
    if len(A) == 0 or len(B) == 0: 
        return None, None, None

    # print(f'shape of A {np.shape(A)} shape of B {np.shape(B)}')
    correspond = match_descriptors( A, B, max_distance=6, max_ratio=0.8 )
    if type(correspond) == type(None) or len(correspond)<=min_sample: return None, None, None
    try:
        model, inliers = ransac((A[correspond[:,0]], B[correspond[:,1]]), AffineTransform, min_samples = min_sample,  residual_threshold=1, max_trials=5000)
        if model: return model.params[:-1], len(correspond), inliers.sum()
    except: pass
    return None, None, None

def getTop(A):
    """
    Detect blobs in image using Difference of Gaussians (DoG) and return top intensity blobs.
    
    Parameters
    ----------
    A : numpy.ndarray
        2D image array for blob detection
        
    Returns
    -------
    numpy.ndarray
        Array of (y, x) coordinates of detected blobs, sorted by intensity (highest first)
        Shape: (n_blobs, 2) where n_blobs <= dot_num
        
    Notes
    -----
    Uses DoG blob detection with sigma range [1, 2] and threshold defined by dot_threshold.
    Returns at most dot_num blobs, sorted by intensity in descending order.
    """
    blobs = blob_dog(A.T.astype(np.float32), min_sigma=1, max_sigma=2,threshold = dot_threshold)
    intensities = A.T[blobs[:,0].astype(np.uint16),blobs[:,1].astype(np.uint16)]
    return blobs[np.flip(np.argsort(intensities))[:dot_num],:-1]


def apply_affine_transform_to_spots(spots, transform_matrix):
    """
    Apply affine transformation to a set of spot coordinates.
    
    Converts 2D coordinates to homogeneous coordinates, applies the transformation
    matrix, and converts back to Cartesian coordinates.
    
    Parameters
    ----------
    spots : numpy.ndarray
        Array of 2D coordinates with shape (n_spots, 2) containing (x, y) positions
    transform_matrix : numpy.ndarray
        3x3 affine transformation matrix in homogeneous coordinates
        
    Returns
    -------
    numpy.ndarray
        Array of transformed 2D coordinates with shape (n_spots, 2)
        
    Notes
    -----
    The transformation is applied as: new_coords = spots @ transform_matrix.T
    Uses homogeneous coordinates to handle affine transformations properly.
    """
    # Convert spots to homogeneous coordinates
    homogeneous_spots = np.hstack([spots, np.ones((spots.shape[0], 1))])
    
    # Apply transformation
    transformed_spots = np.dot(homogeneous_spots, transform_matrix.T)
    
    # Convert back to Cartesian coordinates
    return transformed_spots[:, :2] / transformed_spots[:, 2:]

if __name__ == "__main__": 
    main()
