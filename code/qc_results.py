import dask.array as da
import numpy as np 
import numpy.linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from glob import glob
import zarr
from pathlib import Path
import json
import os
from tqdm import tqdm
from multiprocessing import cpu_count

# Handle optional imports for QC functionality
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.registration import phase_cross_correlation
    from skimage import exposure
    from skimage.feature import blob_dog, match_descriptors
    from skimage import transform as tf
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    from matplotlib.backends.backend_pdf import PdfPages
    from PyPDF2 import PdfMerger
    ADVANCED_QC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced QC features not available. Missing packages: {e}")
    ADVANCED_QC_AVAILABLE = False

try:
    from PyPDF2 import PdfMerger
except ImportError:
    try:
        from PyPDF2 import PdfFileMerger as PdfMerger
    except ImportError:
        print("Warning: PDF merging not available. Install PyPDF2 for full functionality.")
        PdfMerger = None

from calc_affine import get_list_of_channels, get_channel_wavelength_from_single_channel_digit, make_pairs_of_channels

# QC Constants and Configuration
QC_CONFIG = {
    'dot_num': 10000,
    'dot_threshold': 10,
    'top_n_points': 4,
    'n_bins': 50,
    'width': 30,
    'tiles_to_check': np.arange(0, 8, 2),  # Check tiles 0, 2, 4, 6
    'pyramid_level': '0'
}

def get_top_points(image, dot_num=10000, dot_threshold=10):
    """
    Detect blobs in image using Difference of Gaussians (DoG).
    Sort them by intensity and return coordinates of the top intensity blobs.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array for blob detection
    dot_num : int, default=10000
        Maximum number of points to return
    dot_threshold : float, default=10
        Threshold for blob detection
        
    Returns
    -------
    np.ndarray
        Array of (y, x) coordinates of top intensity blobs
    """
    if not ADVANCED_QC_AVAILABLE:
        print("Advanced QC features not available. Please install required packages.")
        return np.array([])
        
    try:
        blobs = blob_dog(
            image.T.astype(np.float32), 
            min_sigma=1, 
            max_sigma=1,  # Changed from 1.5 to int 
            threshold=dot_threshold
        )
        if len(blobs) == 0:
            return np.array([])
            
        intensities = image.T[blobs[:, 0].astype(np.uint16), blobs[:, 1].astype(np.uint16)]
        return blobs[np.flip(np.argsort(intensities))[:dot_num], :-1]
    except Exception as e:
        print(f"Error in blob detection: {e}")
        return np.array([])

def merge_pdfs(pdf_paths, output_path):
    """
    Merge multiple PDF files into a single PDF.
    
    Parameters
    ----------
    pdf_paths : list[str]
        List of paths to PDF files to merge
    output_path : str
        Path for the merged output PDF
    """
    if PdfMerger is None:
        print("PDF merging not available. Please install PyPDF2.")
        return
        
    try:
        merger = PdfMerger()
        for pdf in pdf_paths:
            if os.path.exists(pdf):
                merger.append(pdf)
        merger.write(output_path)
        merger.close()
        print(f"Merged {len(pdf_paths)} PDFs into {output_path}")
    except Exception as e:
        print(f"Error merging PDFs: {e}")

def load_affine_transforms(affine_file_path):
    """
    Load affine transformation matrices from the updated.M.txt file.
    
    Parameters
    ----------
    affine_file_path : str
        Path to the affine transforms file
        
    Returns
    -------
    dict
        Dictionary mapping channel names to 3x3 affine transformation matrices
    """
    affine_dict = {}
    
    if not os.path.exists(affine_file_path):
        print(f"Warning: Affine file not found at {affine_file_path}")
        return affine_dict
    
    try:
        with open(affine_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 7:  # channel + 6 affine parameters
                    channel = parts[0]
                    params = [float(x) for x in parts[1:7]]
                    
                    # Convert to 3x3 affine matrix
                    # params are [A, B, dX, D, E, dY] for 2D affine transform
                    affine_matrix = np.array([
                        [params[0], params[1], params[2]],
                        [params[3], params[4], params[5]],
                        [0, 0, 1]
                    ])
                    affine_dict[channel] = affine_matrix
                    
    except Exception as e:
        print(f"Error loading affine transforms: {e}")
    
    return affine_dict

def apply_affine_to_image(image, affine_matrix):
    """
    Apply affine transformation to an image.
    
    Parameters
    ----------
    image : np.ndarray
        2D image to transform
    affine_matrix : np.ndarray
        3x3 affine transformation matrix
        
    Returns
    -------
    np.ndarray
        Transformed image
    """
    try:
        # Extract 2x3 transformation matrix for skimage
        transform_matrix = affine_matrix[:2, :]
        return tf.warp(image, transform_matrix, output_shape=image.shape)
    except Exception as e:
        print(f"Error applying affine transform: {e}")
        return image

def find_corresponding_points(points1, points2, max_distance=6, max_ratio=0.8):
    """
    Find corresponding points between two sets using feature matching.
    
    Parameters
    ----------
    points1 : np.ndarray
        First set of points (N x 2)
    points2 : np.ndarray
        Second set of points (M x 2)
    max_distance : float, default=6
        Maximum distance for matching
    max_ratio : float, default=0.8
        Maximum ratio for matching
        
    Returns
    -------
    tuple
        (corresponding_points1, corresponding_points2, match_indices)
    """
    if len(points1) == 0 or len(points2) == 0:
        return np.array([]), np.array([]), np.array([])
    
    try:
        match_indices = match_descriptors(
            points1, points2, 
            max_distance=max_distance, 
            max_ratio=max_ratio
        )
        
        if len(match_indices) == 0:
            return np.array([]), np.array([]), np.array([])
            
        corresponding_points1 = points1[match_indices[:, 0], :]
        corresponding_points2 = points2[match_indices[:, 1], :]
        
        return corresponding_points1, corresponding_points2, match_indices
        
    except Exception as e:
        print(f"Error finding corresponding points: {e}")
        return np.array([]), np.array([]), np.array([])

def find_peak_regions(points1, points2, tile_shape, n_bins=50, top_n_points=4):
    """
    Find peak regions where point correspondences are concentrated.
    
    Parameters
    ----------
    points1, points2 : np.ndarray
        Corresponding point sets
    tile_shape : tuple
        (height, width) of the tile
    n_bins : int, default=50
        Number of bins for histogram
    top_n_points : int, default=4
        Number of peak regions to return
        
    Returns
    -------
    np.ndarray
        Peak region coordinates (x, y)
    """
    if len(points1) == 0 or len(points2) == 0:
        return np.array([])
    
    if not ADVANCED_QC_AVAILABLE:
        print("Advanced QC features not available for peak region analysis.")
        return np.array([])
    
    try:
        # Create 2D histogram of point distributions
        points_hist_1 = gaussian_filter(
            np.histogram2d(
                points1[:, 0], points1[:, 1], 
                bins=n_bins, 
                range=[[0, tile_shape[0]], [0, tile_shape[1]]]
            )[0], 
            sigma=1
        )
        
        points_hist_2 = gaussian_filter(
            np.histogram2d(
                points2[:, 0], points2[:, 1], 
                bins=n_bins, 
                range=[[0, tile_shape[0]], [0, tile_shape[1]]]
            )[0], 
            sigma=1
        )
        
        # Find peaks in combined histogram
        combined_hist = points_hist_1 + points_hist_2
        
        # Use a simple peak finding approach instead of peak_local_maxima
        # Find top N points with highest values
        flat_indices = np.argpartition(combined_hist.ravel(), -top_n_points)[-top_n_points:]
        peak_coords = np.unravel_index(flat_indices, combined_hist.shape)
        
        # Convert bin indices to coordinates
        x_locs = (np.linspace(0, tile_shape[1], n_bins + 1)[1:] + 
                  np.linspace(0, tile_shape[1], n_bins + 1)[:-1]) // 2
        y_locs = (np.linspace(0, tile_shape[0], n_bins + 1)[1:] + 
                  np.linspace(0, tile_shape[0], n_bins + 1)[:-1]) // 2
        
        peaks = np.array([(x_locs[peak_coords[1][i]], y_locs[peak_coords[0][i]]) 
                         for i in range(len(peak_coords[0]))])
        return peaks
        
    except Exception as e:
        print(f"Error finding peak regions: {e}")
        return np.array([])

def make_and_save_qc_plots(dataset_path, corrected_path):
    """
    Generate and save quality control plots comparing raw and corrected image data.
    
    Creates comparative visualizations between original and camera-aligned datasets
    to assess the quality of the alignment correction process.
    
    Parameters
    ----------
    dataset_path : str
        Path to the original/raw dataset directory containing zarr tiles
    corrected_path : str
        Path to the camera-corrected dataset directory containing processed zarr tiles
        
    Returns
    -------
    None
        QC plots are saved to disk
        
    Notes
    -----
    - Generates plots for each pair of channels that were aligned
    - Compares raw vs corrected data using metrics like SSIM and cross-correlation
    - Handles both single-digit channel identifiers and full wavelength strings
    - Excludes channel 405 from analysis as it serves as the reference channel
    """
    # tile_list = get_list_of_tiles(dataset_path)

    list_of_channels = get_list_of_channels(dataset_path)
    if len(list_of_channels[0])!=3:
        if '0' in list_of_channels:
            list_of_channels.remove('0')
        list_of_wavelengths = [get_channel_wavelength_from_single_channel_digit(cam) for cam in list_of_channels] 
        pairs_of_channels = make_pairs_of_channels(list_of_wavelengths)

        #pairs_of_channels = [[488,515], [515, 561]...]
        #conver to channels = [[2,4], [4, 3], [3,1]]
        new_pairs_of_channels = []
        for c1, c2 in pairs_of_channels:
            #get index of list_of_wavelengths
            c1_index = next((i for i, x in enumerate(list_of_channels) if hasattr(x, 'value') and x.value == c1), None)
            c2_index = next((i for i, x in enumerate(list_of_channels) if hasattr(x, 'value') and x.value == c2), None)
            if c1_index is not None and c2_index is not None:
                new_pairs_of_channels.append([list_of_channels[c1_index], list_of_channels[c2_index]])
        pairs_of_channels = new_pairs_of_channels
    else:
    #remove 405 from list of channels 
        if '405' in list_of_channels:
            list_of_channels.remove('405')

        #make dict of list_of_wavelengths...
        list_of_wavelengths = {}
        for ch in list_of_channels: 
            list_of_wavelengths[int(ch)] = ch

        pairs_of_channels = make_pairs_of_channels(list_of_channels)

    z = 50 #replace with logic
    
    for c1, c2 in pairs_of_channels:
        raw_tiles_c1 = get_tiles_of_channel(dataset_path, c1)
        raw_tiles_c2 = get_tiles_of_channel(dataset_path, c2)

        corrected_tile_c1 = get_tiles_of_channel(corrected_path, c1)
        corrected_tile_c2 = get_tiles_of_channel(corrected_path, c2)

        if len(raw_tiles_c1) == len(raw_tiles_c2) and len(corrected_tile_c1) == len(corrected_tile_c2) and len(raw_tiles_c1) == len(corrected_tile_c1):
            for i in range(len(corrected_tile_c1)):

                #z = get_z_plane(raw_tiles_c1[i])
                z_planes = get_z_planes(raw_tiles_c1[i], level = 0, thickness = 20, spacing = 20)
                for z in z_planes:

                    raw_c1 = load_raw_zarr_slice(raw_tiles_c1[i], z)
                    raw_c2 = load_raw_zarr_slice(raw_tiles_c2[i], z)

                    corrected_c1 = load_scratch_zarr_slice(corrected_tile_c1[i], z)
                    corrected_c2 = load_scratch_zarr_slice(corrected_tile_c2[i], z)

                    tilename = raw_tiles_c1[i].split('/')[-1].split('.')[0].split('_ch')[0]
                    results_dir ="/results/tile_qc_plots"
                    Path(results_dir).mkdir(parents=False, exist_ok=True)
                    wavelength_c1 = list_of_wavelengths[int(c1)]
                    wavelength_c2 = list_of_wavelengths[int(c2)]
                    filename = f"{results_dir}/{tilename}_z-{z}_{wavelength_c1}-{wavelength_c2}_channel_overlay"


                    overlay_images_rgb_zoom( #test
                        raw_c1, raw_c2, 
                        corrected_c1, corrected_c2, 
                        filename
                    )


                
                # plot_two_sets_of_spots_on_image_zoomed(corrected_c1, transformed_spots, spots2=spots_561, zoom_factor=0.2, spot_size=1, 
                #                spot_color1='red', spot_color2='cyan', alpha=0.6, 
                #                vmin_percentile=1, vmax_percentile=99)


    
def get_z_plane(tile_loc, level = 0):
    """
    Get midpoint Z index from a zarr tile at specified pyramid level.
    
    Parameters
    ----------
    tile_loc : str
        Path to the zarr tile file
    level : int, default=0
        Pyramid level to extract shape information from
        
    Returns
    -------
    int
        Z index corresponding to the midpoint of the tile
    """
    tile = da.from_zarr(tile_loc, level)

    midpoint = int(tile.shape[2]/2)
    return midpoint

def get_z_planes(tile_loc, level = 0, thickness = 20, spacing = 4):
    """
    Get list of Z plane indices from a zarr tile with specified parameters.
    
    Parameters
    ----------
    tile_loc : str
        Path to the zarr tile file
    level : int, default=0
        Pyramid level to extract shape information from
    thickness : int, default=20
        Thickness of the Z sampling region
    spacing : int, default=4
        Spacing between sampled Z planes
        
    Returns
    -------
    list[int]
        List of Z indices within the sampling region
    """
    tile = da.from_zarr(tile_loc, level)
    z = tile.shape[2]
    planes = list(range((z-thickness)//2, (z+thickness)//2, spacing))
    return planes 

def get_tiles_of_channel(dataset_path, channel):
    """
    Get list of zarr tile files for a specific channel.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory
    channel : str
        Channel identifier to filter tiles by
        
    Returns
    -------
    list[str]
        List of file paths to zarr tiles for the specified channel
    """
    list_of_tiles = list(glob(f'{dataset_path}/*{channel}.zarr'))

    return list_of_tiles

def get_list_of_tiles(dataset_path):
    """
    Get list of all zarr tile files in a dataset directory.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory
        
    Returns
    -------
    list[str]
        List of file paths to all zarr tiles in the dataset
    """
    list_of_tiles = list(glob(f'{dataset_path}/*.zarr'))
    return list_of_tiles

def load_raw_zarr_slice(zarr_path, z_index, level = '0'):
    """
    Load a specific Z slice from a raw zarr tile at specified pyramid level.
    
    Parameters
    ----------
    zarr_path : str
        Path to the zarr tile file
    z_index : int
        Z index of the slice to load
    level : str, default='0'
        Pyramid level as string identifier
        
    Returns
    -------
    dask.array.Array
        2D dask array representing the Z slice
    """
    z = da.from_zarr(zarr_path, level)
    return z[0,0,z_index]

def load_scratch_zarr_slice(zarr_path, z_index):
    """
    Load a specific Z slice from a processed zarr tile in scratch directory.
    
    Parameters
    ----------
    zarr_path : str
        Path to the zarr tile file in scratch directory
    z_index : int
        Z index of the slice to load
        
    Returns
    -------
    dask.array.Array
        2D dask array representing the Z slice
    """
    z = da.from_zarr(zarr_path)
    return z[z_index]

# def load_raw_zarr_slice(zarr_path, z_index):
#     z = zarr.open(zarr_path, mode='r')['0']
#     return z[0,0,z_index]

def overlay_images_rgb(image1_raw, image2_raw, image1_corrected, image2_corrected, title):
    """
    Create RGB overlay comparison of raw and corrected image pairs.
    
    Generates side-by-side RGB overlays where the first image appears in red
    channel and second image in green channel for visual comparison.
    
    Parameters
    ----------
    image1_raw : np.ndarray
        First raw image (red channel in overlay)
    image2_raw : np.ndarray  
        Second raw image (green channel in overlay)
    image1_corrected : np.ndarray
        First corrected image (red channel in overlay)
    image2_corrected : np.ndarray
        Second corrected image (green channel in overlay)
    title : str
        Title for the plot and output filename
        
    Returns
    -------
    None
        Saves plot to disk as PNG file
    """
    def create_overlay(image1, image2):
        # Get sizes of the images
        height1, width1 = image1.shape
        height2, width2 = image2.shape
        
        # Calculate the center positions
        center1 = (width1 // 2, height1 // 2)
        center2 = (width2 // 2, height2 // 2)
        
        # Calculate the top-left corner for placing image2 onto image1
        top_left_corner = (center1[0] - center2[0], center1[1] - center2[1])
        
        # Create a new blank canvas with dimensions that can hold both images
        new_height = max(height1, top_left_corner[1] + height2)
        new_width = max(width1, top_left_corner[0] + width2)
        
        # Separate RGB channels for both images
        red_channel = np.zeros((new_height, new_width), dtype=np.uint16)
        green_channel = np.zeros((new_height, new_width), dtype=np.uint16)
        
        # Paste image1 onto the red channel
        red_channel[0:height1, 0:width1] = image1  # Red channel of image1
        
        # Paste image2 onto the green channel
        green_channel[top_left_corner[1]:top_left_corner[1] + height2, 
                      top_left_corner[0]:top_left_corner[0] + width2] = image2  # Green channel of image2
        
        # Combine red and green channels to form the overlay image
        overlay = np.zeros((new_height, new_width, 3), dtype=np.uint16)
        overlay[:,:,0] = red_channel   # Red channel
        overlay[:,:,1] = green_channel # Green channel
        
        return overlay

    # Create raw and corrected overlays
    raw_overlay = create_overlay(image1_raw, image2_raw)
    corrected_overlay = create_overlay(image1_corrected, image2_corrected)

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=400)
    fig.suptitle(title, fontsize=16)


    # Plot raw overlay
    ax1.imshow(raw_overlay)
    ax1.set_title("Raw Overlay")
    ax1.axis('off')

    # Plot corrected overlay
    ax2.imshow(corrected_overlay)
    ax2.set_title("Corrected Overlay")
    ax2.axis('off')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    #plt.show()
    plt.close()


def create_overlay(image1, image2, vmin=None, vmax=None, clip_percentile=(5, 98)):
    # Ensure input images are uint16
    image1 = np.asarray(image1, dtype=np.uint16)
    image2 = np.asarray(image2, dtype=np.uint16)
    
    # Get sizes of the images
    height1, width1 = image1.shape
    height2, width2 = image2.shape
    
    # Calculate the center positions
    center1 = (width1 // 2, height1 // 2)
    center2 = (width2 // 2, height2 // 2)
    
    # Calculate the top-left corner for placing image2 onto image1
    top_left_corner = (center1[0] - center2[0], center1[1] - center2[1])
    
    # Create a new blank canvas with dimensions that can hold both images
    new_height = max(height1, top_left_corner[1] + height2)
    new_width = max(width1, top_left_corner[0] + width2)
    
    # Separate RGB channels for both images
    red_channel = np.zeros((new_height, new_width), dtype=np.uint8)
    green_channel = np.zeros((new_height, new_width), dtype=np.uint8)
    
    # Normalize and paste image1 onto the red channel
    norm_image1 = normalize_and_scale(image1, vmin, vmax, clip_percentile)
    red_channel[0:height1, 0:width1] = norm_image1
    
    # Normalize and paste image2 onto the green channel
    norm_image2 = normalize_and_scale(image2, vmin, vmax, clip_percentile)
    green_channel[top_left_corner[1]:top_left_corner[1] + height2, 
                    top_left_corner[0]:top_left_corner[0] + width2] = norm_image2
    
    # Combine red and green channels to form the overlay image
    overlay = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    overlay[:,:,0] = red_channel   # Red channel
    overlay[:,:,1] = green_channel  # Green channel
    
    return overlay

def normalize_and_scale(image, vmin=None, vmax=None, clip_percentile=(5, 99.9)):
    """
    Normalize and scale image values to uint8 range with optional clipping.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array
    vmin : float, optional
        Minimum value for clipping. If None, uses percentile values
    vmax : float, optional
        Maximum value for clipping. If None, uses percentile values
    clip_percentile : tuple[float, float], default=(5, 99.9)
        Lower and upper percentiles for clipping when vmin/vmax not specified
        
    Returns
    -------
    np.ndarray
        Normalized and scaled image as uint8 array
    """
    # Compute percentile values for clipping
    low, high = np.percentile(image, clip_percentile)
    
    # Apply vmin and vmax if specified, otherwise use percentile values
    vmin = vmin if vmin is not None else low
    vmax = vmax if vmax is not None else high
    
    # Clip the image values
    image_clipped = np.clip(image, vmin, vmax)
    
    # Normalize to [0, 1] range
    image_normalized = (image_clipped - vmin) / (vmax - vmin)
    
    # Scale to uint8 range and convert
    image_scaled = (image_normalized * 255).astype(np.uint8)
    
    return image_scaled
    
    return img_norm
def overlay_images_rgb_zoom(image1_raw, image2_raw, image1_corrected, image2_corrected, title):
    # def create_overlay(image1, image2):
    #     # Get sizes of the images
    #     height1, width1 = image1.shape
    #     height2, width2 = image2.shape
        
    #     # Calculate the center positions
    #     center1 = (width1 // 2, height1 // 2)
    #     center2 = (width2 // 2, height2 // 2)
        
    #     # Calculate the top-left corner for placing image2 onto image1
    #     top_left_corner = (center1[0] - center2[0], center1[1] - center2[1])
        
    #     # Create a new blank canvas with dimensions that can hold both images
    #     new_height = max(height1, top_left_corner[1] + height2)
    #     new_width = max(width1, top_left_corner[0] + width2)
        
    #     # Separate RGB channels for both images
    #     red_channel = np.zeros((new_height, new_width), dtype=np.uint16)
    #     green_channel = np.zeros((new_height, new_width), dtype=np.uint16)
        
    #     # Paste image1 onto the red channel
    #     red_channel[0:height1, 0:width1] = image1  # Red channel of image1
        
    #     # Paste image2 onto the green channel
    #     green_channel[top_left_corner[1]:top_left_corner[1] + height2, 
    #                     top_left_corner[0]:top_left_corner[0] + width2] = image2  # Green channel of image2
        
    #     # Combine red and green channels to form the overlay image
    #     overlay = np.zeros((new_height, new_width, 3), dtype=np.uint16)
    #     overlay[:,:,0] = red_channel   # Red channel
    #     overlay[:,:,1] = green_channel  # Green channel
        
    #     return overlay
    def plot_zoomed_corner(ax, image, corner, zoom_size):
        h, w, _ = image.shape
        if corner == 'top_left':
            region = image[:zoom_size, :zoom_size]
        elif corner == 'top_right':
            region = image[:zoom_size, -zoom_size:]
        elif corner == 'bottom_left':
            region = image[-zoom_size:, :zoom_size]
        elif corner == 'bottom_right':
            region = image[-zoom_size:, -zoom_size:]
        
        ax.imshow(region)
        ax.set_title(f"{corner.replace('_', ' ').title()}")
        ax.axis('off')

    raw_overlay = create_overlay(image1_raw, image2_raw)
    corrected_overlay = create_overlay(image1_corrected, image2_corrected)

    fig = plt.figure(figsize=(20, 20), dpi=400)
    fig.suptitle(title, fontsize=16)

    gs = fig.add_gridspec(3, 4)

    ax_raw = fig.add_subplot(gs[0, :2])
    ax_raw.imshow(raw_overlay)
    ax_raw.set_title("Raw Overlay")
    ax_raw.axis('off')

    ax_corrected = fig.add_subplot(gs[0, 2:])
    ax_corrected.imshow(corrected_overlay)
    ax_corrected.set_title("Corrected Overlay")
    ax_corrected.axis('off')

    zoom_size = min(raw_overlay.shape[0], raw_overlay.shape[1]) // 8

    corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

     # Add labels for raw and corrected zoomed images
    fig.text(0.02, 0.62, 'Raw Zoomed', fontsize=14, fontweight='bold', rotation=00, va='center')
    fig.text(0.02, 0.32, 'Corrected Zoomed', fontsize=14, fontweight='bold', rotation=00, va='center')
    for i, corner in enumerate(corners):
        ax_raw_zoom = fig.add_subplot(gs[1, i])
        plot_zoomed_corner(ax_raw_zoom, raw_overlay, corner, zoom_size)
        
        ax_corrected_zoom = fig.add_subplot(gs[2, i])
        plot_zoomed_corner(ax_corrected_zoom, corrected_overlay, corner, zoom_size)

    plt.tight_layout()
    plt.savefig(f'{title}.png')
    #plt.show()
    plt.close()

def overlay_two_images_rgb(image1, image2, title):
    # Load images and convert them to RGB format

    # Get sizes of the images
    height1, width1 = image1.shape
    height2, width2 = image2.shape
    
    # Calculate the center positions
    center1 = (width1 // 2, height1 // 2)
    center2 = (width2 // 2, height2 // 2)
    
    # Calculate the top-left corner for placing image2 onto image1
    top_left_corner = (center1[0] - center2[0], center1[1] - center2[1])
    
    # Create a new blank canvas with dimensions that can hold both images
    new_height = max(height1, top_left_corner[1] + height2)
    new_width = max(width1, top_left_corner[0] + width2)
    
    # Separate RGB channels for both images
    red_channel = np.zeros((new_height, new_width), dtype=np.uint16)
    green_channel = np.zeros((new_height, new_width), dtype=np.uint16)
    
    # Paste image1 onto the red channel
    red_channel[0:height1, 0:width1] = image1  # Red channel of image1
    
    # Paste image2 onto the green channel
    green_channel[top_left_corner[1]:top_left_corner[1] + height2, 
                  top_left_corner[0]:top_left_corner[0] + width2] = image2  # Green channel of image2
    
    # Combine red and green channels to form the overlay image
    overlay = np.zeros((new_height, new_width, 3), dtype=np.uint16)
    overlay[:,:,0] = red_channel*0.5  # Red channel
    overlay[:,:,1] = green_channel*0.5  # Green channel
    
    # Display the overlay
    plt.suptitle(title)
    plt.figure(dpi=400)
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.savefig(f'{title}.png')
    #plt.show()
    plt.close()

#plot the spots over the dataset
def plot_spots_on_image(image, spots, spot_size=20, spot_color='red', alpha=0.7, vmin_percentile = 5, vmax_percentile = 99):
    """
    Plot spots on top of an image.
    
    Parameters:
    - image: 2D or 3D numpy array representing the image
    - spots: Nx2 numpy array of (x, y) coordinates of spots
    - spot_size: Size of the spots (default: 20)
    - spot_color: Color of the spots (default: 'red')
    - alpha: Transparency of the spots (default: 0.7)
    """
    # Create a new figure
    plt.figure(figsize=(10, 10))
    
    #normalize the image
    vmin = np.percentile(image, vmin_percentile)
    vmax = np.percentile(image, vmax_percentile)
    # Display the image
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None, vmin = vmin, vmax = vmax)
    
    # Plot the spots
    plt.scatter(spots[:, 0], spots[:, 1], s=spot_size, c=spot_color, alpha=alpha)
    
    # Set the axis limits to match the image dimensions
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)  # Invert y-axis to match image coordinates
    
    plt.title("Image with Spots")
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    #plt.show()
    plt.close()

def plot_spots_on_image_zoomed(image, spots, zoom_factor=0.25, spot_size=20, spot_color='red', alpha=0.7, vmin_percentile=1, vmax_percentile=99):
    """
    Plot spots on top of an image with normalized intensity bounds, showing a zoomed-in portion of the right corner.
    
    Parameters:
    - image: 2D or 3D numpy array representing the image
    - spots: Nx2 numpy array of (x, y) coordinates of spots
    - zoom_factor: Factor determining the size of the zoomed area (default: 0.25, showing the right 25% of the image)
    - spot_size: Size of the spots (default: 20)
    - spot_color: Color of the spots (default: 'red')
    - alpha: Transparency of the spots (default: 0.7)
    - vmin_percentile: Percentile for lower bound of image intensity (default: 1)
    - vmax_percentile: Percentile for upper bound of image intensity (default: 99)
    """
    # Create a new figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Compute vmin and vmax based on percentiles
    vmin = np.percentile(image, vmin_percentile)
    vmax = np.percentile(image, vmax_percentile)
    
    # Display the full image with normalized intensity bounds
    im1 = ax1.imshow(image, cmap='gray' if image.ndim == 2 else None, vmin=vmin, vmax=vmax)
    ax1.scatter(spots[:, 0], spots[:, 1], s=spot_size, c=spot_color, alpha=alpha)
    ax1.set_title("Full Image with Spots")
    ax1.axis('off')
    
    # Calculate the zoomed area
    x_start = int(image.shape[1] * (1 - zoom_factor))
    y_start = 0
    x_end = image.shape[1]
    y_end = int(image.shape[0] * zoom_factor)
    
    # Display the zoomed portion
    im2 = ax2.imshow(image[y_start:y_end, x_start:x_end], cmap='gray' if image.ndim == 2 else None, vmin=vmin, vmax=vmax)
    
    # Filter spots within the zoomed area
    zoomed_spots = spots[(spots[:, 0] >= x_start) & (spots[:, 0] < x_end) & 
                         (spots[:, 1] >= y_start) & (spots[:, 1] < y_end)]
    
    # Adjust spot coordinates for the zoomed view
    zoomed_spots_adjusted = zoomed_spots.copy()
    zoomed_spots_adjusted[:, 0] -= x_start
    zoomed_spots_adjusted[:, 1] -= y_start
    
    ax2.scatter(zoomed_spots_adjusted[:, 0], zoomed_spots_adjusted[:, 1], s=spot_size*2, c=spot_color, alpha=alpha)
    ax2.set_title("Zoomed Right Corner with Spots")
    ax2.axis('off')
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='Intensity')
    plt.colorbar(im2, ax=ax2, label='Intensity')
    
    # Show the zoomed area on the full image
    rect = Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, 
                         fill=False, ec='yellow', lw=2)
    ax1.add_patch(rect)
    
    plt.tight_layout()
    #plt.show()
    plt.close()


def plot_two_sets_of_spots_on_image_zoomed(image, spots1, spots2=None, zoom_factor=0.25, spot_size=20, 
                               spot_color1='red', spot_color2='blue', alpha=0.7, 
                               vmin_percentile=1, vmax_percentile=99):
    """
    Plot one or two sets of spots on top of an image with normalized intensity bounds, 
    showing a zoomed-in portion of the right corner.
    
    Parameters:
    - image: 2D or 3D numpy array representing the image
    - spots1: Nx2 numpy array of (x, y) coordinates of first set of spots
    - spots2: Nx2 numpy array of (x, y) coordinates of second set of spots (optional)
    - zoom_factor: Factor determining the size of the zoomed area (default: 0.25)
    - spot_size: Size of the spots (default: 20)
    - spot_color1: Color of the first set of spots (default: 'red')
    - spot_color2: Color of the second set of spots (default: 'blue')
    - alpha: Transparency of the spots (default: 0.7)
    - vmin_percentile: Percentile for lower bound of image intensity (default: 1)
    - vmax_percentile: Percentile for upper bound of image intensity (default: 99)
    """
    # Create a new figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Compute vmin and vmax based on percentiles
    vmin = np.percentile(image, vmin_percentile)
    vmax = np.percentile(image, vmax_percentile)
    
    # Display the full image with normalized intensity bounds
    im1 = ax1.imshow(image, cmap='gray' if image.ndim == 2 else None, vmin=vmin, vmax=vmax)
    ax1.scatter(spots1[:, 0], spots1[:, 1], s=spot_size, c=spot_color1, alpha=alpha, label='Spots 1')
    if spots2 is not None:
        ax1.scatter(spots2[:, 0], spots2[:, 1], s=spot_size, c=spot_color2, alpha=alpha, label='Spots 2')
    ax1.set_title("Full Image with Spots")
    ax1.axis('off')
    ax1.legend()
    
    # Calculate the zoomed area
    x_start = int(image.shape[1] * (1 - zoom_factor))
    y_start = 0
    x_end = image.shape[1]
    y_end = int(image.shape[0] * zoom_factor)
    
    # Display the zoomed portion
    im2 = ax2.imshow(image[y_start:y_end, x_start:x_end], cmap='gray' if image.ndim == 2 else None, vmin=vmin, vmax=vmax)
    
    # Filter spots within the zoomed area
    def filter_and_adjust_spots(spots):
        zoomed_spots = spots[(spots[:, 0] >= x_start) & (spots[:, 0] < x_end) & 
                             (spots[:, 1] >= y_start) & (spots[:, 1] < y_end)]
        zoomed_spots_adjusted = zoomed_spots.copy()
        zoomed_spots_adjusted[:, 0] -= x_start
        zoomed_spots_adjusted[:, 1] -= y_start
        return zoomed_spots_adjusted
    
    zoomed_spots1_adjusted = filter_and_adjust_spots(spots1)
    ax2.scatter(zoomed_spots1_adjusted[:, 0], zoomed_spots1_adjusted[:, 1], s=spot_size*2, c=spot_color1, alpha=alpha, label='Spots 1')
    
    if spots2 is not None:
        zoomed_spots2_adjusted = filter_and_adjust_spots(spots2)
        ax2.scatter(zoomed_spots2_adjusted[:, 0], zoomed_spots2_adjusted[:, 1], s=spot_size*2, c=spot_color2, alpha=alpha, label='Spots 2')
    
    ax2.set_title("Zoomed Right Corner with Spots")
    ax2.axis('off')
    ax2.legend()
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='Intensity')
    plt.colorbar(im2, ax=ax2, label='Intensity')
    
    # Show the zoomed area on the full image
    rect = Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, 
                         fill=False, ec='yellow', lw=2)
    ax1.add_patch(rect)
    
    plt.tight_layout()
    #plt.show()
    plt.close()

def create_distance_plots(points1, points2, affine1, affine2, ax_hist, ax_scatter, title):
    """
    Create distance comparison plots (histogram and scatter) for point correspondences.
    
    Parameters
    ----------
    points1, points2 : np.ndarray
        Corresponding point sets
    affine1, affine2 : np.ndarray
        Affine transformation matrices
    ax_hist, ax_scatter : matplotlib.axes.Axes
        Axes for histogram and scatter plots
    title : str
        Plot title
    """
    if len(points1) == 0 or len(points2) == 0:
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])
        return
    
    try:
        # Transform points using affine matrices
        homogeneous_points1 = np.vstack([points1.T, np.ones(points1.shape[0])])
        transformed_points1 = (affine1 @ homogeneous_points1)[:2, :].T.astype(int)
        
        homogeneous_points2 = np.vstack([points2.T, np.ones(points2.shape[0])])
        transformed_points2 = (affine2 @ homogeneous_points2)[:2, :].T.astype(int)
        
        # Calculate distances
        distance_pre = numpy.linalg.norm(points1 - points2, axis=1)
        distance_post = numpy.linalg.norm(transformed_points1 - transformed_points2, axis=1)
        
        # Create histogram
        ax_hist.hist(distance_pre, 10, alpha=0.5, label='pre-correction', color='r')
        ax_hist.hist(distance_post, 10, alpha=0.5, label='post-correction', color='g')
        ax_hist.legend()
        ax_hist.set_xlabel('distance (pixels)', fontsize=14)
        ax_hist.set_ylabel('number of points', fontsize=14)
        ax_hist.set_title(title, fontsize=12)
        
        # Create scatter plot
        max_dist = np.max(np.hstack([distance_pre, distance_post]))
        ax_scatter.scatter(distance_pre, distance_post)
        ax_scatter.plot([0, max_dist], [0, max_dist], 'k--', alpha=0.5)
        ax_scatter.set_xlabel('distance pre-correction (pixels)', fontsize=12)
        ax_scatter.set_ylabel('distance post-correction (pixels)', fontsize=12)
        ax_scatter.set_title(title, fontsize=12)
        
    except Exception as e:
        print(f"Error creating distance plots: {e}")

def create_comparison_subplot(tile1_clip, tile2_clip, tile1_transformed_clip, tile2_transformed_clip, 
                            width, c1, c2, x_loc, y_loc, z_loc, tilename, output_dir_png, output_dir_pdf):
    """
    Create before/after comparison subplot for a specific region.
    
    Parameters
    ----------
    tile1_clip, tile2_clip : np.ndarray
        Clipped regions from original tiles
    tile1_transformed_clip, tile2_transformed_clip : np.ndarray
        Clipped regions from transformed tiles
    width : int
        Width of the clipped region
    c1, c2 : str
        Channel names
    x_loc, y_loc, z_loc : int
        Location coordinates
    tilename : str
        Tile identifier
    output_dir_png, output_dir_pdf : str
        Output directories for PNG and PDF files
    """
    try:
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(1, 2, wspace=0.1, hspace=0.1)
        
        # Calculate intensity ranges
        vmin_1 = int(np.percentile(tile1_clip, 10))
        vmax_1 = int(np.percentile(tile1_clip, 99.99))
        vmin_2 = int(np.percentile(tile2_clip, 10))
        vmax_2 = int(np.percentile(tile2_clip, 99.99))
        
        # Pre-correction subplot
        ax = fig.add_subplot(gs[0])
        img = np.zeros((tile1_clip.shape[0], tile1_clip.shape[1], 3), dtype=np.float32)
        img[:, :, 0] = np.clip((tile1_clip.astype(np.float32) - vmin_1) / (vmax_1 - vmin_1), 0, 1)
        img[:, :, 1] = np.clip((tile2_clip.astype(np.float32) - vmin_2) / (vmax_2 - vmin_2), 0, 1)
        ax.imshow(img, aspect='auto')
        ax.text(0, 0, 'Pre-correction', color='w', fontsize=20, 
                horizontalalignment='left', verticalalignment='top')
        ax.axis('off')
        
        # Post-correction subplot
        ax = fig.add_subplot(gs[1])
        vmin_1_trans = np.percentile(tile1_transformed_clip, 10)
        vmax_1_trans = np.percentile(tile1_transformed_clip, 99.99)
        vmin_2_trans = np.percentile(tile2_transformed_clip, 10)
        vmax_2_trans = np.percentile(tile2_transformed_clip, 99.99)
        
        img = np.zeros((tile1_transformed_clip.shape[0], tile1_transformed_clip.shape[1], 3), dtype=np.float32)
        img[:, :, 0] = np.clip((tile1_transformed_clip.astype(np.float32) - vmin_1_trans) / (vmax_1_trans - vmin_1_trans), 0, 1)
        img[:, :, 1] = np.clip((tile2_transformed_clip.astype(np.float32) - vmin_2_trans) / (vmax_2_trans - vmin_2_trans), 0, 1)
        ax.imshow(img, aspect='auto')
        ax.text(0, 0, 'Post-correction', color='w', fontsize=20, 
                horizontalalignment='left', verticalalignment='top')
        ax.axis('off')
        
        # Title and save
        tile_id = '_'.join(tilename.split('/')[-1].split('_')[1:5]).replace('_Y', '-Y').replace('_0', '')
        plt.suptitle(f'{c1.replace("CH_", "")} vs. {c2.replace("CH_", "")}, tile = {tile_id} - x={x_loc}, y={y_loc}, z={z_loc}', 
                    y=0.95, fontsize=18)
        
        # Save PNG
        png_filename = f"{c1.replace('CH_', '')}vs{c2.replace('CH_', '')}_{tilename.split('/')[-1][:18]}_x{x_loc}_y{y_loc}_z{z_loc}.png"
        fig.savefig(os.path.join(output_dir_png, png_filename), dpi=200, bbox_inches='tight')
        
        # Save PDF
        pdf_filename = f"{c1.replace('CH_', '')}vs{c2.replace('CH_', '')}_{tilename.split('/')[-1][:18]}_x{x_loc}_y{y_loc}_z{z_loc}.pdf"
        with PdfPages(os.path.join(output_dir_pdf, pdf_filename)) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        
        plt.close()
        
    except Exception as e:
        print(f"Error creating comparison subplot: {e}")

def make_comprehensive_qc_plots(data_path, scratch_root, output_root="/results/comprehensive_qc"):
    """
    Create comprehensive QC plots comparing pre- and post-correction alignment.
    
    This function implements the full QC analysis from the research notebook,
    including:
    - Distance histograms and scatter plots
    - Before/after image comparisons
    - Peak region analysis
    - PDF report generation
    
    Parameters
    ----------
    data_path : str
        Path to the dataset directory
    scratch_root : str
        Path to scratch directory containing affine transforms
    output_root : str, default="/results/comprehensive_qc"
        Root directory for QC output files
        
    Returns
    -------
    None
        QC plots and reports are saved to disk
    """
    try:
        # Create output directories
        output_dir = output_root
        output_dir_png = os.path.join(output_dir, "png_files")
        output_dir_pdf = os.path.join(output_dir, "pdf_files")
        output_dir_json = os.path.join(output_dir, "json_files")
        
        for dir_path in [output_dir, output_dir_png, output_dir_pdf, output_dir_json]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Load affine transforms
        affine_file = os.path.join(scratch_root, 'updated.M.txt')
        affine_transforms = load_affine_transforms(affine_file)
        
        if not affine_transforms:
            print(f"No affine transforms found in {affine_file}")
            return
        
        # Get channel information
        channels = list(affine_transforms.keys())
        if '405' in channels:
            channels.remove('405')  # Remove reference channel
        
        channel_pairs = [(channels[i], channels[i+1]) for i in range(len(channels)-1)]
        
        print(f"Processing {len(channel_pairs)} channel pairs: {channel_pairs}")
        print(f"Found affine transforms for channels: {channels}")
        
        # Get tiles from one channel to determine structure
        sample_tiles = get_tiles_of_channel(data_path, channels[0])
        if not sample_tiles:
            print(f"No tiles found for channel {channels[0]} in {data_path}")
            return
        
        # Process subset of tiles for QC
        tiles_to_check = QC_CONFIG['tiles_to_check']
        tiles_to_process = min(len(sample_tiles), max(tiles_to_check) + 1)
        tiles_to_check = tiles_to_check[tiles_to_check < tiles_to_process]
        
        print(f"Processing {len(tiles_to_check)} tiles: {tiles_to_check}")
        
        # Get tile dimensions
        sample_tile_path = sample_tiles[0]
        tile_zarr = da.from_zarr(sample_tile_path, QC_CONFIG['pyramid_level'])
        tile_shape = tile_zarr.shape[2:]  # Assuming 5D zarr: (C, T, Z, Y, X)
        
        # Configure Z planes to sample
        thickness = tile_shape[0] // 3
        spacing = tile_shape[0] // 5
        num_planes = int(np.ceil(thickness / spacing))
        planes = np.arange((tile_shape[0] - thickness) // 2, (tile_shape[0] + thickness) // 2, spacing)
        
        print(f"Tile shape: {tile_shape}, Processing {len(planes)} Z planes: {planes}")
        
        # Create distance plot figures
        fig_dist, axs_dist = plt.subplots(
            num_planes * len(tiles_to_check), len(channel_pairs),
            figsize=(5 * len(channel_pairs), 5 * len(tiles_to_check) * num_planes)
        )
        fig_dist_scatter, axs_dist_scatter = plt.subplots(
            num_planes * len(tiles_to_check), len(channel_pairs),
            figsize=(5 * len(channel_pairs), 5 * len(tiles_to_check) * num_planes)
        )
        
        # Ensure axs are 2D arrays
        if len(channel_pairs) == 1:
            axs_dist = axs_dist.reshape(-1, 1)
            axs_dist_scatter = axs_dist_scatter.reshape(-1, 1)
        if num_planes * len(tiles_to_check) == 1:
            axs_dist = axs_dist.reshape(1, -1)
            axs_dist_scatter = axs_dist_scatter.reshape(1, -1)
        
        # Process each tile
        for i_tile, tile_idx in enumerate(tiles_to_check):
            print(f"Processing tile {tile_idx + 1}/{len(sample_tiles)}")
            
            # Process each channel pair
            for i_pair, (c1, c2) in enumerate(channel_pairs):
                print(f"  Processing pair {c1} vs {c2}")
                
                # Load tiles for both channels
                tiles_c1 = get_tiles_of_channel(data_path, c1)
                tiles_c2 = get_tiles_of_channel(data_path, c2)
                
                if tile_idx >= len(tiles_c1) or tile_idx >= len(tiles_c2):
                    print(f"  Skipping - insufficient tiles for channels {c1}, {c2}")
                    continue
                
                tilename_1 = tiles_c1[tile_idx]
                tilename_2 = tiles_c2[tile_idx]
                
                # Load tile data
                tile_1_zarr = da.from_zarr(tilename_1, QC_CONFIG['pyramid_level'])[0, 0, ...]
                tile_2_zarr = da.from_zarr(tilename_2, QC_CONFIG['pyramid_level'])[0, 0, ...]
                
                tile_1 = tile_1_zarr[planes, ...].compute()
                tile_2 = tile_2_zarr[planes, ...].compute()
                
                # Apply affine transforms
                affine_1 = affine_transforms[c1]
                affine_2 = affine_transforms[c2]
                
                tile_1_transformed = np.zeros_like(tile_1, dtype=np.float32)
                tile_2_transformed = np.zeros_like(tile_2, dtype=np.float32)
                
                for i_plane, plane in enumerate(planes):
                    tile_1_transformed[i_plane, ...] = apply_affine_to_image(tile_1[i_plane, ...], affine_1)
                    tile_2_transformed[i_plane, ...] = apply_affine_to_image(tile_2[i_plane, ...], affine_2)
                
                # Process each Z plane
                peaks = {}
                for i_plane, plane in enumerate(planes):
                    # Detect points
                    points_1 = get_top_points(tile_1[i_plane, ...], QC_CONFIG['dot_num'], QC_CONFIG['dot_threshold'])
                    points_2 = get_top_points(tile_2[i_plane, ...], QC_CONFIG['dot_num'], QC_CONFIG['dot_threshold'])
                    
                    # Find correspondences
                    correspond_points_1, correspond_points_2, _ = find_corresponding_points(points_1, points_2)
                    
                    # Create distance plots
                    ax_row = i_plane + num_planes * i_tile
                    title = f'{c1.replace("CH_", "")} vs. {c2.replace("CH_", "")}, tile {tile_idx} - z={plane}'
                    
                    create_distance_plots(
                        correspond_points_1, correspond_points_2, affine_1, affine_2,
                        axs_dist[ax_row, i_pair], axs_dist_scatter[ax_row, i_pair], title
                    )
                    
                    # Find peak regions for detailed analysis
                    if len(correspond_points_1) > 0:
                        peaks[plane] = find_peak_regions(
                            correspond_points_1, correspond_points_2, tile_shape[1:],
                            QC_CONFIG['n_bins'], QC_CONFIG['top_n_points']
                        )
                    else:
                        peaks[plane] = np.array([])
                
                # Create detailed comparison plots for peak regions
                for plane, peak_regions in peaks.items():
                    i_plane = list(peaks.keys()).index(plane)
                    
                    for i_point in range(len(peak_regions)):
                        x_loc = int(peak_regions[i_point, 0])
                        y_loc = int(peak_regions[i_point, 1])
                        z_loc = int(plane)
                        
                        # Extract regions around peak
                        width = QC_CONFIG['width']
                        y_start, y_end = max(0, y_loc - width), min(tile_shape[1], y_loc + width)
                        x_start, x_end = max(0, x_loc - width), min(tile_shape[2], x_loc + width)
                        
                        tile_1_clip = tile_1[i_plane, y_start:y_end, x_start:x_end]
                        tile_2_clip = tile_2[i_plane, y_start:y_end, x_start:x_end]
                        tile_1_transformed_clip = tile_1_transformed[i_plane, y_start:y_end, x_start:x_end]
                        tile_2_transformed_clip = tile_2_transformed[i_plane, y_start:y_end, x_start:x_end]
                        
                        # Create comparison subplot
                        create_comparison_subplot(
                            tile_1_clip, tile_2_clip, tile_1_transformed_clip, tile_2_transformed_clip,
                            width, c1, c2, x_loc, y_loc, z_loc, tilename_1, output_dir_png, output_dir_pdf
                        )
        
        # Save distance plot figures
        fig_dist.suptitle('Distance Histograms - Camera Alignment QC', fontsize=16)
        fig_dist.savefig(os.path.join(output_dir, 'distance_histograms.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_dist)
        
        fig_dist_scatter.suptitle('Distance Scatter Plots - Camera Alignment QC', fontsize=16)
        fig_dist_scatter.savefig(os.path.join(output_dir, 'distance_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_dist_scatter)
        
        # Merge PDFs for each channel pair
        for c1, c2 in channel_pairs:
            pdf_pattern = f"{c1.replace('CH_', '')}vs{c2.replace('CH_', '')}"
            pdf_list = [
                os.path.join(output_dir_pdf, f) for f in os.listdir(output_dir_pdf)
                if f.endswith('.pdf') and pdf_pattern in f
            ]
            
            if pdf_list:
                merged_pdf_path = os.path.join(output_dir, f"{pdf_pattern}_merged.pdf")
                merge_pdfs(pdf_list, merged_pdf_path)
        
        print(f"QC analysis complete. Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error in comprehensive QC analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":

    raw_dataset_path = '/root/capsule/data/HCR_BL6-001_2023-06-19_00-01-00/SPIM.ome.zarr/'
    corrected_path = '/root/capsule/scratch/HCR_BL6-001_2023-06-19_00-01-00/affine.ome.zarr'
    make_and_save_qc_plots(raw_dataset_path, corrected_path)