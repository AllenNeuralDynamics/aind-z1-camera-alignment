import dask.array as da
import numpy as np 
import matplotlib.pyplot as plt
from glob import glob
import zarr
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
from skimage import exposure

from calc_affine import get_list_of_channels, get_channel_wavelength_from_single_channel_digit, make_pairs_of_channels



def make_and_save_qc_plots(dataset_path, corrected_path):
    # tile_list = get_list_of_tiles(dataset_path)

    list_of_channels = get_list_of_channels(dataset_path)
    if len(list_of_channels[0])!=3:
        if '0' in list_of_channels:
            list_of_channels.remove('0')
        list_of_wavelengths = [get_channel_wavelength_from_single_channel_digit(cam) for cam in list_of_channels] 
        pairs_of_channels = make_pairs_of_channels(list_of_wavelengths)

        #pairs_of_channels = [[488,515], [515, 561]...]
        #conver to channels = [[2,4], [4, 3], [3,1]]
        pairs_of_channels = []
        for c1, c2 in pairs_of_channels:
            #get index of list_of_wavelengths
            c1_index = next((i for i, x in enumerate(list_of_channels) if x.value == c1), None)
            c2_index = next((i for i, x in enumerate(list_of_channels) if x.value == c2), None)
            pairs_of_channels.append([list_of_channels[c1_index], list_of_channels[c2_index]])
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
    tile = da.from_zarr(tile_loc, level)

    midpoint = int(tile.shape[2]/2)
    return midpoint

def get_z_planes(tile_loc, level = 0, thickness = 20, spacing = 4):
    tile = da.from_zarr(tile_loc, level)
    z = tile.shape[2]
    planes = list(range((z-thickness)//2, (z+thickness)//2, spacing))
    return planes 

def get_tiles_of_channel(dataset_path, channel):
    list_of_tiles = list(glob(f'{dataset_path}/*{channel}.zarr'))

    return list_of_tiles

def get_list_of_tiles(dataset_path):
    list_of_tiles = list(glob(f'{dataset_path}/*.zarr'))
    return list_of_tiles

def load_raw_zarr_slice(zarr_path, z_index, level = '0'):
    z = da.from_zarr(zarr_path, level)
    return z[0,0,z_index]

def load_scratch_zarr_slice(zarr_path, z_index):
    z = da.from_zarr(zarr_path)
    return z[z_index]

# def load_raw_zarr_slice(zarr_path, z_index):
#     z = zarr.open(zarr_path, mode='r')['0']
#     return z[0,0,z_index]

def overlay_images_rgb(image1_raw, image2_raw, image1_corrected, image2_corrected, title):
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
    rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, 
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
    rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, 
                         fill=False, ec='yellow', lw=2)
    ax1.add_patch(rect)
    
    plt.tight_layout()
    #plt.show()
    plt.close()

if __name__ == "__main__":

    raw_dataset_path = '/root/capsule/data/HCR_BL6-001_2023-06-19_00-01-00/SPIM.ome.zarr/'
    corrected_path = '/root/capsule/scratch/HCR_BL6-001_2023-06-19_00-01-00/affine.ome.zarr'
    make_and_save_qc_plots(raw_dataset_path, corrected_path)