# determine out affine transforms based on image data using ransac

import tifffile as tf, numpy as np, os, sys
import pickle
from glob import glob
from skimage.feature import blob_dog, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform
from multiprocessing import Pool, cpu_count
# from common_func import getConfigs, root
from os.path import basename as bn
from ast import literal_eval
import dask.array as da
import json
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from co_api import list_data_directory
from utils import (
    list_zarr_tiles_from_s3,
)

nodes = cpu_count()-1
dot_num, dot_threshold, min_sample = 10000, 10, 5
FRACTION_OF_DOTS_PER_PLANE_THRESHOLD = 0.4
thickness, spacing = 20, 4
max_tiles = 20
pyramid_level = '0'

def save_tile_metrics(metrics_dict, results_root):
    """Save the metrics dictionary to a JSON file"""
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


def calc_affine(root: str, results_root: str = '/scratch/', qc_root = '/results/tile_affine_qc'):
    Path(qc_root).mkdir(exist_ok=True)
    list_of_channels = get_list_of_channels(root)
    #print(f'list of channels {list_of_channels}')
    
    if len(list_of_channels[0])!=3:
        list_of_channels = [get_channel_wavelength_from_single_channel_digit(cam) for cam in list_of_channels] 

    if '405' in list_of_channels:
        list_of_channels.remove('405')

    pairs_of_channels = make_pairs_of_channels(list_of_channels)
    keep_cam = {x:'' for x in sum(pairs_of_channels, [])}
    
    pairs_of_channels.reverse()
    print(f'pairs of channels = {pairs_of_channels}')

    list_of_tiles = list(glob(f'{root}/*.zarr'))
    if len(list_of_tiles)==0: 
        list_of_tiles = list_zarr_tiles_from_s3(root)
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
        Is[cam].append(da.from_zarr(fn, pyramid_level)[0,0,list(range((z-thickness)//2, (z+thickness)//2, spacing)),...])
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
            model_result = find_model((points_c2[0], points_c1[0]))  # Using first plane for now
            
            if model_result[0] is not None:
                aff, num_matches, num_inliers = model_result
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
            weights = inliers / inliers.sum()
            weighted_affine = np.append(np.array([aff*w for aff, w in zip(all_affs, weights)]).sum(0),[0,0,1]).reshape((3,3))
            finalM[c1] = weighted_affine
            
            # Calculate difference from average for each tile
            for tile_coord in common_tiles:
                if "affine_transform" in tile_metrics[f"{c1}_{c2}"][tile_coord]:
                    tile_affine = tile_metrics[f"{c1}_{c2}"][tile_coord]["affine_transform"]
                    diff_from_average = norm(tile_affine - weighted_affine)
                    tile_metrics[f"{c1}_{c2}"][tile_coord]["diff_from_average"] = diff_from_average

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

def calc_affine_bu(root: str, results_root: str = '/scratch/'):

    list_of_channels = get_list_of_channels(root)

    if len(list_of_channels[0])!=3:
        list_of_channels = [get_channel_wavelength_from_single_channel_digit(cam) for cam in list_of_channels] 

    #remove 405 from list of channels 
    if '405' in list_of_channels:
        list_of_channels.remove('405')



    pairs_of_channels = make_pairs_of_channels(list_of_channels)
    keep_cam = {x:'' for x in sum(pairs_of_channels, [])}
    
    #reverse pairs of channels so we can apply affines successively to the dot locations
    # ie the first
    pairs_of_channels.reverse()
    print(f'pairs of channels = {pairs_of_channels}')

    list_of_tiles = list(glob(f'{root}/*.zarr'))


    tile_number_dict = create_tile_number_dict(list_of_tiles)
    max_pos = max(tile_number_dict.values()) 

    #extract a subset of the image based on the thickness and spacing parameters 
    Is, finalM = {}, {}
    #if not True:
    

    for fn in list_of_tiles:

        cam = bn(fn.split(".")[2]).split('_')[-1]
        if len(cam)!= 3: #get pairs of channels from other logic 
            cam = get_channel_wavelength_from_single_channel_digit(cam)
        # cam = bn(fn.split(".")[2])[-3:]
        # pos  = tile_number_dict[fn]
        if cam not in keep_cam: continue

        #rewrite for zarr
        # z, pixels = tf.TiffFile(fn).series[0].shape[:2]
        z = da.from_zarr(fn, pyramid_level).shape[0]
        pixels = da.from_zarr(fn, pyramid_level).shape[1]

        if cam not in Is: Is[cam] = []
        #rewrite for zarr
        # Is[cam].append(tf.imread(fn, key = list(range((z-thickness)//2, (z+thickness)//2, spacing))))
        Is[cam].append(da.from_zarr(fn, pyramid_level)[list(range((z-thickness)//2, (z+thickness)//2, spacing)),...])


    
    for k, v in Is.items():
        print('finding spots for', k)
        Is[k] = Pool(nodes).map(getTop, np.array(v).reshape((-1, pixels, pixels))) 
    #seeing 200 here, because it is 40 tiles, getting 5 planes per tile


    #find affine transform that best matches the blobs in the images from the two cameras:
    # calculate a werighted average of the model parameters based on the number of inliers. 
    chans = list(Is.keys())


    for c1, c2 in pairs_of_channels:

        dots = np.array([min(len(x), len(y)) for x, y in zip(Is[c1], Is[c2])]) #a list of the mininum number of blobs in each plane for this pair of images
        mininum_number_of_dots = max(dots)*FRACTION_OF_DOTS_PER_PLANE_THRESHOLD

        Is_c1 = [i.astype(int) for i in Is[c1]]
        outm = []
        for i, arr in enumerate(Is_c1):
            if dots[i] > mininum_number_of_dots:
                outm.append(arr)
        
        Is_c2 = [i.astype(int) for i in Is[c2]]
        outf = []
        for i, arr in enumerate(Is_c2):
            if dots[i] > mininum_number_of_dots:
                outf.append(arr)

        if c2 in finalM.keys():
            aff_to_apply_to_c2_spots = finalM[c2]
            outf = [apply_affine_transform_to_spots(c2_spots,  np.linalg.inv(aff_to_apply_to_c2_spots)) for c2_spots in Is_c2]

        
        affs = list(filter(lambda x: x[1], Pool(nodes).map(find_model, zip(outf, outm))))
        inliers = np.array([x[2] for x in affs])
        inliers = inliers/inliers.sum()
        for x in affs: print(c1, c2, x[1:], x[0].flatten().round(4).tolist())
        finalM[c1] = np.append(np.array([x[0]*y for x,y in zip(affs, inliers)]).sum(0),[0,0,1]).reshape((3,3))


#save out the first 6 items out of the affine transformation matrix, as the last 3 are always 0,0,1.

    with open(results_root+'updated.M.txt', 'w') as f:
        for e in sorted(finalM.keys()):
            print(e+'\t'+'\t'.join(list(map(str,np.round(finalM[e][:-1].flatten(),6)))), file=f)
            print(e+'\t'+'\t'.join(list(map(str,np.round(finalM[e][:-1].flatten(),6)))))



def extract_coordinates(tile_name):
    """Extract X, Y coordinates from tile name"""
    parts = tile_name.split('_')
    x = int(parts[2])
    y = int(parts[4])
    return x, y

def get_grid_dimensions(tile_name):
    """Extract grid dimensions from a tile name like 'Tile_X_0007_Y_0003_Z_0000'"""
    parts = tile_name.split('_')
    max_x = int(max([parts[i+1] for i, part in enumerate(parts) if part == 'X']))
    max_y = int(max([parts[i+1] for i, part in enumerate(parts) if part == 'Y']))
    return max_x + 1, max_y + 1

def create_metric_grid(metrics_data, channel_pair, metric_name):
    """Create a grid of metric values using maximum dimensions from tile names"""
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
    """Create and save a heatmap plot for a metric"""
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
    """Create visualizations for tile metrics"""
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


def get_channel_wavelength_from_single_channel_digit(single_channel_digit):
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
    list_of_tiles = list(glob(f'{data_loc}/*.zarr'))

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
    list_of_tiles = list_zarr_tiles_from_s3(data_loc)
    # find unique channels in the list of tiles
    channels = []
    for tile in list_of_tiles:
        #form is tile_x_0000_y_0000_z_0000_ch_405.zarr
        channel = tile.split('_')[-1].split('.')[0]
        if channel not in channels:
            channels.append(channel)    
    return channels

#make pairs of moving channels 
def make_pairs_of_channels(channels):
    """Chooses pairs of channels to align, based on spectral overlap between. 
     This essentially means that we will align channels that are close in wavelength to each other.
     Args:
         channels (list): list of channels in the dataset
         
     Returns:
         list: list of pairs of channels to align
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


def create_tile_number_dict(tilenames: list) -> dict:
    """
    Create a dictionary mapping tilenames to tile numbers based on raster scanning order.
    
    :param tilenames: List of unsorted tilenames containing X, Y, Z coordinates
    :return: Dictionary mapping tilenames to tile numbers
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




def find_model(input):
    A, B = input
    # print(f'shape of A {np.shape(A)} shape of B {np.shape(B)}')
    correspond = match_descriptors( A, B, max_distance=6, max_ratio=0.8 )
    if type(correspond) == type(None) or len(correspond)<min_sample: return None, None, None
    try:
        model, inliers = ransac((A[correspond[:,0]], B[correspond[:,1]]), AffineTransform, min_samples = min_sample,  residual_threshold=1, max_trials=5000)
        if model: return model.params[:-1], len(correspond), inliers.sum()
    except: pass
    return None, None, None

def getTop(A):
    """
    Detects blobs in image A, using Difference of Gaussians (DoG)
    Sort them by intensity and 
    return coordinates of the top intensity blobs
    """
    blobs = blob_dog(A.T.astype(np.float32), min_sigma=1, max_sigma=1.5,threshold = dot_threshold)
    intensities = A.T[blobs[:,0].astype(np.uint16),blobs[:,1].astype(np.uint16)]
    return blobs[np.flip(np.argsort(intensities))[:dot_num],:-1]

# tiffs = glob(root+'TILE*tif')


def apply_affine_transform_to_spots(spots, transform_matrix):
    # Convert spots to homogeneous coordinates
    homogeneous_spots = np.hstack([spots, np.ones((spots.shape[0], 1))])
    
    # Apply transformation
    transformed_spots = np.dot(homogeneous_spots, transform_matrix.T)
    
    # Convert back to Cartesian coordinates
    return transformed_spots[:, :2] / transformed_spots[:, 2:]

if __name__ == "__main__": 
    main()
