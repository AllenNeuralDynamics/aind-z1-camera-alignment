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
from s3_writer import get_resolution_zyx, copy_file_to_s3


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



nodes = cpu_count()-1
dot_num, dot_threshold = 10000, 10 # Matt's suggestions 500, 100
top_n_points = 4
n_bins = 50
width = 30
dot_threshold = 10
tiles2check = np.arange(0,8,2)

def getTop(A, dot_num = 10000, dot_threshold = 10):
    """
    Detects blobs in image A, using Difference of Gaussians (DoG)
    Sort them by intensity and 
    return coordinates of the top intensity blobs
    """
    blobs = blob_dog(A.T.astype(np.float32), min_sigma=1, max_sigma=1.5,threshold = dot_threshold)
    intensities = A.T[blobs[:,0].astype(np.uint16),blobs[:,1].astype(np.uint16)]
    return blobs[np.flip(np.argsort(intensities))[:dot_num],:-1]


def merge_pdfs(pdf_paths, output_path):
    merger = PdfMerger()
    for pdf in pdf_paths:
        merger.append(pdf)
    merger.write(output_path)
    merger.close()

def make_files_and_ng_links(): 
    data_dir = "/root/capsule/data"
data_assets = os.listdir(data_dir)
data_assets.sort()

data_assets_data = [d for d in data_assets if 'image_radial_correction' in os.listdir(os.path.join(data_dir,d))]
for data_asset in data_assets_data:
    print(f"Processing {data_asset}")
    output_dir = f"/root/capsule/scratch/{data_asset}_camera_alignment_QC"
    os.makedirs(output_dir, exist_ok=True)
    output_dir_png = os.path.join(output_dir, "png_files")
    output_dir_pdf = os.path.join(output_dir, "pdf_files")
    output_dir_json = os.path.join(output_dir, "json_files")
    os.makedirs(output_dir_png, exist_ok=True)
    os.makedirs(output_dir_pdf, exist_ok=True)
    os.makedirs(output_dir_json, exist_ok=True)

    data_path = os.path.join(data_dir, data_asset)
    cc_ng_json = os.path.join(data_path, "camera_aligned_neuroglancer.json")
    rc_ng_json = os.path.join(data_path, "radially_corrected_neuroglancer.json")
    template_ng_json = "/root/capsule/code/ng_template.json"
    with open(cc_ng_json, "r") as f:
        cc_data = json.load(f)


    with open(rc_ng_json, "r") as f:
        rc_data = json.load(f)
    channel_names = [layer['name'] for layer in cc_data['layers']]
    channel_names_sorted = sorted(channel_names)
    channel_names_sorted.remove('CH_405')
    channel_names_paired = [channel_names_sorted[i:i + 2] for i in range(len(channel_names_sorted)-1)]


    layers = {}
    tilenames = {}
    affine_mat = {}
    for c in channel_names_sorted:
        layers[c] = [layer for layer in cc_data['layers'] if layer['name'] == c][0]

    first_tile_name = layers[channel_names_sorted[-1]]['source'][0]['url'].split('/')[-1][:18]
    
    offset = np.array(layers[channel_names_sorted[-1]]['source'][0]['transform']['matrix'])[-2:,5:].T
    for c in channel_names_sorted:
        layer = layers[c]

        tilenames[c] = [s['url'] for s in layer['source']]
        source = [s for s in layer['source'] if first_tile_name in s['url']][0]
        M = np.array(source['transform']['matrix'])[2:,2:]
        M[:,3] += M[:,0]
        M = M[:,1:]
        M[1:,-1] = M[1:,-1] - offset
        M = M[::-1,:]
        M[:,:2]= M[:,1::-1]
        affine_mat[c] = M
        # affine_mat[c] = source['transform']['matrix']
        # print(f"Channel: {c}, Affine matrix:\n{M}\n")

    c_last = channel_names_sorted[-1]
    layer = layers[c_last]
    source = layer['source']
    dist2center = []
    for s in source:
        matrix = np.array(s['transform']['matrix'])
        points = matrix[3:, -1:]
        dist2center.append(np.linalg.norm(points))
    dist2center = np.array(dist2center)
    ind_closest2center = np.argsort(dist2center)
    tilenames[c_last] = [tilenames[c_last][i] for i in ind_closest2center]

    for c in channel_names_sorted[:-1]:
        layer = layers[c]
        c = layer['name']
        tilenames[c] = [t.replace(c_last.split('_')[1],c.split('_')[1]) for t in tilenames[c_last]]

    for i in range(len(cc_data['layers'])):
        cc_data['layers'][i]['name'] = cc_data['layers'][i]['name']+"_cc"  
    for i in range(len(rc_data['layers'])):
        rc_data['layers'][i]['name'] = rc_data['layers'][i]['name']+"_rc"
        
    for channel_pair in channel_names_paired:
        
        with open(template_ng_json, "r") as f:
            template_data = json.load(f)
        layer_cc_0 = [layer for layer in cc_data['layers'] if layer['name'] == channel_pair[0]+"_cc"][0]
        layer_cc_0['shader'] = "#uicontrol vec3 color color(default=\"#ff0000\")\n#uicontrol invlerp normalized\nvoid main() {\nemitRGB(color * normalized());\n}"

        layer_cc_1 = [layer for layer in cc_data['layers'] if layer['name'] == channel_pair[1]+"_cc"][0]
        layer_cc_1['shader'] = "#uicontrol vec3 color color(default=\"#00ff00\")\n#uicontrol invlerp normalized\nvoid main() {\nemitRGB(color * normalized());\n}"

        layer_rc_0 = [layer for layer in rc_data['layers'] if layer['name'] == channel_pair[0]+"_rc"][0]
        layer_rc_0['shader'] = "#uicontrol vec3 color color(default=\"#00ff00\")\n#uicontrol invlerp normalized\nvoid main() {\nemitRGB(color * normalized());\n}"

        layer_rc_1 = [layer for layer in rc_data['layers'] if layer['name'] == channel_pair[1]+"_rc"][0]
        layer_rc_1['shader'] = "#uicontrol vec3 color color(default=\"#ff0000\")\n#uicontrol invlerp normalized\nvoid main() {\nemitRGB(color * normalized());\n}"


        template_data['layers'] = [layer_cc_0, layer_cc_1, layer_rc_0, layer_rc_1]
        template_data['dimensions'] = cc_data['dimensions']
        output_filename = f"cc_ng_{channel_pair[0]}_{channel_pair[1]}.json"
        output_path = os.path.join(output_dir,output_filename)
        with open(output_path, "w") as f:
            json.dump(template_data, f, indent=2)

    first_channel = channel_names_sorted[0]
    tilename = tilenames[first_channel][0].replace('s3://aind-open-data','/root/capsule/data')
    pyramid_level = '0'
    tile_shape = da.from_zarr(tilename, pyramid_level).shape[2:]     #---------------> if 5D z should be shape[2] instead of shape[0]
    img = np.zeros((tile_shape[1],tile_shape[2],3),dtype=np.uint8)
    thickness, spacing = tile_shape[0]//3, tile_shape[0]//5 #-------> from 20 middle planes select 1 from every 4
    num_planes = int(np.ceil(thickness/spacing))
    planes = np.arange((tile_shape[0]-thickness)//2, (tile_shape[0]+thickness)//2, spacing)

    x_locs = (np.linspace(0,tile_shape[2],n_bins+1)[1:]+np.linspace(0,tile_shape[2],n_bins+1)[:-1])//2
    y_locs = (np.linspace(0,tile_shape[1],n_bins+1)[1:]+np.linspace(0,tile_shape[1],n_bins+1)[:-1])//2

    fig_dist, axs_dist = plt.subplots(num_planes*len(tiles2check),len(channel_names_paired),figsize=(5*len(channel_names_paired),5*len(tiles2check)*num_planes))
    fig_dist_scatter, axs_dist_scatter = plt.subplots(num_planes*len(tiles2check),len(channel_names_paired),figsize=(5*len(channel_names_paired),5*len(tiles2check)*num_planes))


    for i_tile, tile_ind in enumerate(tiles2check): # 4 tiles in the center region
        c1 = ''
        c2 = ''
        print("--------------------------------------------------")
        print(f"Processing tile {tile_ind+1}")
        for i_pair, pair in enumerate(channel_names_paired):
            print(f"Processing pair {pair[0]} vs {pair[1]}")

            c1 = pair[0]
            
            print(f'loading channel {c1}')
            if c2 == c1:
                tilename_1 = tilename_2
                affine_1 = affine_2.copy()
                tile_1 = tile_2.copy()
                tile_1_transformed = tile_2_transformed.copy()
            else:
                tilename_1 = tilenames[c1][tile_ind].replace('s3://aind-open-data','/root/capsule/data')
                affine_1 = affine_mat[c1].copy()
                # Use Dask's percentile to avoid loading the whole array into memory
                tile_1_zarr = da.from_zarr(tilename_1, pyramid_level)[0,0,...]

                tile_1 = tile_1_zarr[planes,...].compute()
                # np.zeros((len(planes), tile_shape[1], tile_shape[2]), dtype=np.uint8)
                tile_1_transformed = np.zeros_like(tile_1, dtype=np.float32)
                for i_plane, plane in tqdm(enumerate(planes),desc=f'Processing {c1}'):
                    im = tile_1[i_plane,...]
                    tile_1_transformed[i_plane,...] = tf.warp(im, affine_1, output_shape=im.shape)

            c2 = pair[1]
            print(f'loading channel {c2}')
            tilename_2 = tilenames[c2][tile_ind].replace('s3://aind-open-data','/root/capsule/data')
            affine_2 = affine_mat[c2].copy()
            tile_2_zarr = da.from_zarr(tilename_2, pyramid_level)[0,0,...]

            tile_2 = tile_2_zarr[planes,...].compute()
            tile_2_transformed = np.zeros_like(tile_2, dtype=np.float32)
            for i_plane, plane in tqdm(enumerate(planes),desc=f'Processing {c2}'):
                im = tile_2[i_plane,...]
                tile_2_transformed[i_plane,...] = tf.warp(im, affine_2, output_shape=im.shape)

            peaks = {}
            for i_plane, plane in enumerate(planes):
                
                points_1 = getTop(tile_1[i_plane,...], dot_num=dot_num, dot_threshold=dot_threshold)
                points_2 = getTop(tile_2[i_plane,...], dot_num=dot_num, dot_threshold=dot_threshold)
                
                
                ax = axs_dist[i_plane+num_planes*i_tile,i_pair]
                ax.set_title(f'{c1.replace("CH_","")} vs. {c2.replace("CH_","")}, tile = {'_'.join(tilename_1.split('/')[-1].split('_')[1:5]).replace('_Y','-Y').replace('_0','')} - z={plane}',fontsize = 10)

                ax_scatter = axs_dist_scatter[i_plane+num_planes*i_tile,i_pair]
                ax_scatter.set_title(f'{c1.replace("CH_","")} vs. {c2.replace("CH_","")}, tile = {'_'.join(tilename_1.split('/')[-1].split('_')[1:5]).replace('_Y','-Y').replace('_0','')} - z={plane}',fontsize = 10)

                if i_pair == 0:
                    ax.set_ylabel('number of points', fontsize = 18)

                
                if len(points_1)==0 or len(points_2)==0:
                    ax.set_xticks([])
                    ax.set_yticks([])

                    ax_scatter.set_xticks([])
                    ax_scatter.set_yticks([])
                    continue

                correspond_points_ind = match_descriptors(points_1, points_2, max_distance=6, max_ratio=0.8)
                if len(correspond_points_ind)==0:
                    peaks[plane] = []
                    ax.set_xticks([])
                    ax.set_yticks([])

                    ax_scatter.set_xticks([])
                    ax_scatter.set_yticks([])
                    continue
                correspond_points_1 = points_1[correspond_points_ind[:,0],:]
                correspond_points_2 = points_2[correspond_points_ind[:,1],:]
                points_hist_1 = gaussian_filter(np.histogram2d(correspond_points_1[:,0], correspond_points_1[:,1], bins=n_bins, range=[[0, tile_shape[1]], [0, tile_shape[2]]])[0], sigma=1)
                points_hist_2 = gaussian_filter(np.histogram2d(correspond_points_2[:,0], correspond_points_2[:,1], bins=n_bins, range=[[0, tile_shape[1]], [0, tile_shape[2]]])[0], sigma=1)

                peaks_ind = peak_local_max(points_hist_1 + points_hist_2, min_distance=2, threshold_rel=0.2, num_peaks=top_n_points)
                peaks[plane] = np.array([(x_locs[p[0]],y_locs[p[1]]) for p in peaks_ind])

                
                homogeneous_points_1 = np.vstack([
                    correspond_points_1.T,
                    np.ones(correspond_points_1.shape[0])
                ])
                transformed_homogeneous_points_1 = affine_1 @ homogeneous_points_1
                transformed_points_1 = transformed_homogeneous_points_1[:2, :].T.astype(int)

                homogeneous_points_2 = np.vstack([
                    correspond_points_2.T,
                    np.ones(correspond_points_2.shape[0])
                ])
                transformed_homogeneous_points_2 = affine_2 @ homogeneous_points_2
                transformed_points_2 = transformed_homogeneous_points_2[:2, :].T.astype(int)

                distance_pre = np.linalg.norm(correspond_points_1 - correspond_points_2, axis=1)
                distance_post = np.linalg.norm(transformed_points_1 - transformed_points_2, axis=1)
                


                ax.hist(distance_pre,10,alpha=0.5, label='pre-correction',color='r');
                ax.hist(distance_post,10,alpha=0.5, label='post-correction',color ='g');
                ax.legend()
                ax.set_xlabel('distance (pixels)', fontsize = 18)

                ax_scatter.scatter(distance_pre,distance_post)
                ax_scatter.plot([0,np.max(np.hstack([distance_pre,distance_post]))],[0,np.max(np.hstack([distance_pre,distance_post]))])
                ax_scatter.set_xlabel('distance pre-correction (pixels)', fontsize = 16)
                ax_scatter.set_ylabel('distance post-correction (pixels)', fontsize = 15)

            planes = list(peaks.keys())

            for i_plane, plane in enumerate(planes):
                print(f"Processing plane {plane}")

                for i_point in range(len(peaks[plane])):
                    fig = plt.figure(figsize = (10,5))
                    gs = fig.add_gridspec(1, 2, wspace=0.1, hspace=0.1)    
                    x_loc = int(peaks[plane][i_point,0])
                    y_loc = int(peaks[plane][i_point,1])
                    z_loc = int(plane)
                    print(f"Making figure for point x = {x_loc}, y = {y_loc}, in plane {plane}")
                    tile_1_clip = tile_1[i_plane,y_loc-width:y_loc+width,x_loc-width:x_loc+width]
                    tile_2_clip = tile_2[i_plane,y_loc-width:y_loc+width,x_loc-width:x_loc+width]
                    tile_1_transformed_clip = tile_1_transformed[i_plane,y_loc-width:y_loc+width,x_loc-width:x_loc+width]
                    tile_2_transformed_clip = tile_2_transformed[i_plane,y_loc-width:y_loc+width,x_loc-width:x_loc+width]

                    vmin_1 = int(np.percentile(tile_1_clip, 10))
                    vmax_1 = int(np.percentile(tile_1_clip, 99.99))
                    vmin_2 = int(np.percentile(tile_2_clip, 10))
                    vmax_2 = int(np.percentile(tile_2_clip, 99.99))

                    output_filename = f"cc_ng_{pair[0]}_{pair[1]}.json"
                    output_path = os.path.join(output_dir,output_filename)
                    with open(output_path, "r") as f:
                        template_data = json.load(f)

                    json_data = template_data.copy()
                    for i_c in range(2):
                        c = pair[i_c]
                        tilename = tilenames[pair[i_c]][tile_ind]
                        for i in range(len(json_data['layers'])):
                            if json_data['layers'][i]['name'].startswith(c):
                                json_data['layers'][i]['source'] = [s for s in json_data['layers'][i]['source'] if s['url']==tilename]

                                json_data['layers'][i]['shaderControls']['normalized']['range'] = [vmin_1, vmax_1]
                                json_data['layers'][i]['shaderControls']['normalized']['range'] = [vmin_2, vmax_2]
                                
                                if json_data['layers'][i]['name'].endswith('_rc'):
                                    json_data['layers'][i]['visible'] = False
                    json_data['position'] = [x_loc, y_loc, z_loc,0]
                    json_data['crossSectionScale'] = 0.2
                    json_data['projectionScale'] = 80

                    json_data['layers'][0]['source'][0]['transform']['matrix'][-1][-1] -= json_data['layers'][2]['source'][0]['transform']['matrix'][-1][-1] 
                    json_data['layers'][0]['source'][0]['transform']['matrix'][-2][-1] -= json_data['layers'][2]['source'][0]['transform']['matrix'][-2][-1]

                    json_data['layers'][1]['source'][0]['transform']['matrix'][-1][-1] -= json_data['layers'][3]['source'][0]['transform']['matrix'][-1][-1] 
                    json_data['layers'][1]['source'][0]['transform']['matrix'][-2][-1] -= json_data['layers'][3]['source'][0]['transform']['matrix'][-2][-1]

                    json_data['layers'][2]['source'][0]['transform']['matrix'][-1][-1] = 0
                    json_data['layers'][2]['source'][0]['transform']['matrix'][-2][-1] = 0
                    json_data['layers'][3]['source'][0]['transform']['matrix'][-1][-1] = 0
                    json_data['layers'][3]['source'][0]['transform']['matrix'][-2][-1] = 0

                    output_filename = f"cc_ng_{pair[0]}_{pair[1]}_{tilename.split('/')[-1][:18]}_x{x_loc}_y{y_loc}_z{z_loc}_zoomed.json"
                    output_path = os.path.join(output_dir_json,output_filename)
                    with open(output_path, "w") as f:
                        json.dump(json_data, f, indent=2)
                    ng_link = f"https://neuroglancer-demo.appspot.com/#!s3://aind-open-data/{data_asset}/image_cross_image_alignment/{output_filename}"
        
                    ax = fig.add_subplot(gs[0])
                    img = np.zeros((tile_1_clip.shape[0], tile_1_clip.shape[1],3), dtype=np.float32)
                    img[:,:,0] = np.clip((tile_1_clip.astype(np.float32)- vmin_1) / (vmax_1 - vmin_1),0,1)
                    img[:,:,1] = np.clip((tile_2_clip.astype(np.float32)- vmin_2) / (vmax_2 - vmin_2),0,1)
                    ax.imshow(img, aspect='auto')
                    ax.text(0, 0, 'Pre-correction', color='w', fontsize=20,  horizontalalignment='left',verticalalignment='top')

                    ax.axis('off')

                    ax = fig.add_subplot(gs[1])

                    vmin_1 = np.percentile(tile_1_transformed_clip, 10)
                    vmax_1 = np.percentile(tile_1_transformed_clip, 99.99)
                    vmin_2 = np.percentile(tile_2_transformed_clip, 10)
                    vmax_2 = np.percentile(tile_2_transformed_clip, 99.99)

                    img = np.zeros((tile_1_transformed_clip.shape[0], tile_1_transformed_clip.shape[1],3), dtype=np.float32)
                    img[:,:,0] = np.clip((tile_1_transformed_clip.astype(np.float32)- vmin_1) / (vmax_1 - vmin_1),0,1)
                    img[:,:,1] = np.clip((tile_2_transformed_clip.astype(np.float32)- vmin_2) / (vmax_2 - vmin_2),0,1)
                    ax.imshow(img, aspect='auto')
                    ax.text(0, 0, 'Post-correction', color='w', fontsize=20,  horizontalalignment='left',verticalalignment='top')

                    ax.axis('off')

                    plt.suptitle(f'{c1.replace("CH_","")} vs. {c2.replace("CH_","")}, tile = {'_'.join(tilename_1.split('/')[-1].split('_')[1:5]).replace('_Y','-Y').replace('_0','')} - x={x_loc}, y={y_loc}, z={z_loc}',y=.95, fontsize = 18)
                    fig.text(0.9, 0.08, 'neuroglancer link', 
                    ha='right', # Horizontal alignment
                    color='blue', 
                    url=ng_link)
                    pdf_filename = os.path.join(output_dir_pdf, f"{c1.replace('CH_','')}vs{c2.replace('CH_','')}_{tilename_1.split('/')[-1][:18]}_x{x_loc}_y{y_loc}_z{z_loc}.pdf")
                    with PdfPages(pdf_filename) as pdf:
                        pdf.savefig(fig, bbox_inches='tight')
                    fig.savefig(os.path.join(output_dir_png, f"{c1.replace('CH_','')}vs{c2.replace('CH_','')}_{tilename_1.split('/')[-1][:18]}_x{x_loc}_y{y_loc}_z{z_loc}.png"), dpi=200)
                    plt.close()

    for pair in channel_names_paired:
        print(f"Merging pdfs pair {pair[0]} vs {pair[1]}")

        pdf_list = [os.path.join(output_dir_pdf, f) for f in os.listdir(output_dir_pdf) if f.endswith('.pdf') if pair[0].replace('CH_','') in f and pair[1].replace('CH_','') in f]
        merge_pdfs(pdf_list, os.path.join(output_dir, f"{pair[0].replace('CH_','')}vs{pair[1].replace('CH_','')}_merged.pdf"))

if __name__=='__main__': 
    make_files_and_ng_links()