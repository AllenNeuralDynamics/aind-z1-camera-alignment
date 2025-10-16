"""Generate camera-alignment QC plots and neuroglancer assets from existing JSON metadata.

This module ingests the scientist-provided neuroglancer JSON exports
(`camera_aligned_neuroglancer.json` and `radial_correction_neuroglancer.json`)
for a dataset and reproduces their QC plotting workflow in a more maintainable
way. The entry point `generate_camera_alignment_qc` is intended to be invoked
from the capsule's QC target and can also be used programmatically.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from s3_writer import copy_file_to_s3


# Optional dependencies that enable the advanced QC workflow
try:
    from skimage.feature import blob_dog, match_descriptors, peak_local_max
    from skimage import transform as tf
    from scipy.ndimage import gaussian_filter
    from matplotlib.backends.backend_pdf import PdfPages
    ADVANCED_QC_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - handled at runtime
    print(
        "Warning: Advanced QC features not available. Missing packages: "
        f"{exc}. Install skimage/scipy/matplotlib with PDF support."
    )
    ADVANCED_QC_AVAILABLE = False
    PdfPages = None  # type: ignore[assignment]

try:  # pragma: no cover - runtime optional dependency
    from PyPDF2 import PdfMerger
except ImportError:  # pragma: no cover - runtime optional dependency
    try:
        from PyPDF2 import PdfFileMerger as PdfMerger  # type: ignore[misc]
    except ImportError:
        print(
            "Warning: PDF merging not available. Install PyPDF2 for full functionality."
        )
        PdfMerger = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("/data")
DEFAULT_SCRATCH_ROOT = Path("/scratch")
DEFAULT_RESULTS_ROOT = Path("/results")
DEFAULT_TEMPLATE_PATH = Path("/data/camera_aligned_neuroglancer.json")
AIND_S3_PREFIX = "s3://aind-open-data"


@dataclass(frozen=True)
class QCSettings:
    """Configuration parameters for the QC plotting workflow."""

    dot_num: int = 10000
    dot_threshold: float = 10.0
    top_n_points: int = 4
    n_bins: int = 50
    clip_half_width: int = 30
    tiles_to_check: Sequence[int] = (0, 2, 4, 6)
    pyramid_level: str = "0"
    cross_section_scale: float = 0.2
    projection_scale: float = 80.0


def apply_affine_to_image(image: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    """Apply a 2D affine transform to an image plane."""

    try:
        transform_matrix = np.asarray(affine_matrix, dtype=float)
        return tf.warp(image, transform_matrix, output_shape=image.shape)
    except Exception as exc:  # pragma: no cover - runtime safety
        LOGGER.error("Failed to apply affine transform", exc_info=exc)
        return image


def _determine_s3_json_base(tilenames: Dict[str, List[str]]) -> Optional[str]:
    """Infer the destination S3 prefix for neuroglancer JSON artefacts."""

    for urls in tilenames.values():
        for url in urls:
            if not url.startswith("s3://"):
                continue
            for marker in ("image_radial_correction", "image_radially_corrected"):
                if marker in url:
                    prefix = url.split(marker, 1)[0].rstrip("/")
                    return f"{prefix}/image_cross_image_alignment/"
    return None


def _upload_json_to_s3(local_path: Path, s3_json_base: Optional[str]) -> None:
    """Upload a JSON file to the derived S3 prefix when available."""

    if not s3_json_base:
        return
    destination = f"{s3_json_base}{local_path.name}"
    try:
        copy_file_to_s3(str(local_path), destination)
    except Exception as exc:  # pragma: no cover - runtime safety
        LOGGER.warning(
            "Failed to upload %s to %s: %s", local_path, destination, exc
        )


def ensure_advanced_qc_available() -> None:
    """Raise a helpful error if advanced QC dependencies are missing."""

    if not ADVANCED_QC_AVAILABLE:
        raise RuntimeError(
            "Advanced QC dependencies are not available. Please ensure skimage, "
            "scipy, and matplotlib (with PdfPages support) are installed."
        )
    if PdfPages is None:
        raise RuntimeError(
            "matplotlib PdfPages is unavailable; cannot create PDF outputs."
        )


def load_json(path: Path) -> Dict:
    """Load a JSON document from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_pdfs(pdf_paths: Sequence[Path], output_path: Path) -> None:
    """Merge multiple PDF files into a single PDF if PyPDF2 is available."""

    if PdfMerger is None:
        LOGGER.warning("PyPDF2 unavailable; skipping PDF merge for %s", output_path)
        return

    merger = PdfMerger()
    try:
        for pdf in pdf_paths:
            if pdf.exists():
                merger.append(str(pdf))
        if merger.pages:
            with output_path.open("wb") as handle:
                merger.write(handle)
            LOGGER.info("Merged %d PDFs into %s", len(pdf_paths), output_path)
        else:
            LOGGER.debug("No PDF pages found to merge for %s", output_path)
    finally:
        merger.close()


def convert_s3_to_local(url: str, data_dir: Path) -> Path:
    """Convert an S3 URL to the expected local on-disk zarr path."""

    if not url.startswith(AIND_S3_PREFIX):
        return Path(url)
    relative = url[len(AIND_S3_PREFIX) :].lstrip("/")
    return data_dir / relative


def _clamp_slice(center: int, half_width: int, upper_bound: int) -> slice:
    start = max(0, center - half_width)
    end = min(upper_bound, center + half_width)
    return slice(start, end)


def _resolve_datasets(
    dataset_names: Optional[Sequence[str]],
    data_dir: Path,
) -> List[Path]:
    if dataset_names:
        dataset_paths = [data_dir / name for name in dataset_names]
    else:
        dataset_paths = [
            path
            for path in data_dir.iterdir()
            if path.is_dir() and (path / "image_radial_correction").exists()
        ]
    return [path for path in dataset_paths if path.exists()]


def _prepare_output_dirs(output_dir: Path, dataset_name: str) -> Tuple[Path, Path, Path, Path]:
    output_dir = output_dir / f"{dataset_name}_camera_alignment_QC"
    png_dir = output_dir / "png_files"
    pdf_dir = output_dir / "pdf_files"
    json_dir = output_dir / "json_files"

    for directory in (output_dir, png_dir, pdf_dir, json_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return output_dir, png_dir, pdf_dir, json_dir


def _prepare_channel_layers(
    cc_data: Dict,
    rc_data: Dict,
    channel_names: Sequence[str],
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    cc_layers: Dict[str, Dict] = {}
    rc_layers: Dict[str, Dict] = {}

    for channel in channel_names:
        cc_layer = copy.deepcopy(
            next(layer for layer in cc_data["layers"] if layer["name"] == channel)
        )
        cc_layer["name"] = f"{channel}_cc"
        cc_layers[channel] = cc_layer

        rc_layer = copy.deepcopy(
            next(layer for layer in rc_data["layers"] if layer["name"] == channel)
        )
        rc_layer["name"] = f"{channel}_rc"
        rc_layers[channel] = rc_layer

    return cc_layers, rc_layers


def _extract_affine_metadata(
    cc_layers: Dict[str, Dict],
    ordered_channels: Sequence[str],
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    tilenames: Dict[str, List[str]] = {}
    affine_matrices: Dict[str, np.ndarray] = {}

    last_channel = ordered_channels[-1]
    reference_layer = cc_layers[last_channel]
    first_tile_token = Path(reference_layer["source"][0]["url"]).name[:18]
    offset = np.asarray(reference_layer["source"][0]["transform"]["matrix"], dtype=float)[
        -2:, 5:
    ].T

    for channel, layer in cc_layers.items():
        tilenames[channel] = [entry["url"] for entry in layer["source"]]

        reference_source = next(
            entry for entry in layer["source"] if first_tile_token in entry["url"]
        )
        matrix = np.asarray(reference_source["transform"]["matrix"], dtype=float)
        affine = matrix[2:, 2:]
        affine[:, 3] += affine[:, 0]
        affine = affine[:, 1:]
        affine[1:, -1] -= offset[0]
        affine = affine[::-1, :]
        affine[:, :2] = affine[:, 1::-1]
        affine_matrices[channel] = np.linalg.inv(affine)

    # Order tiles by proximity to the volume center for the reference channel
    distances: List[float] = []
    ref_urls = tilenames[last_channel]
    for entry in reference_layer["source"]:
        matrix = np.asarray(entry["transform"]["matrix"], dtype=float)
        point = matrix[3:, -1:]
        distances.append(float(np.linalg.norm(point)))
    order = np.argsort(distances)
    tilenames[last_channel] = [ref_urls[idx] for idx in order]

    # Replicate ordering for the remaining channels via string replacement
    ref_suffix = last_channel.split("_")[1]
    for channel in ordered_channels[:-1]:
        suffix = channel.split("_")[1]
        tilenames[channel] = [url.replace(ref_suffix, suffix) for url in tilenames[last_channel]]

    return tilenames, affine_matrices


def _create_pair_templates(
    template_data: Dict,
    cc_layers: Dict[str, Dict],
    rc_layers: Dict[str, Dict],
    channel_pairs: Sequence[Tuple[str, str]],
    output_dir: Path,
    s3_json_base: Optional[str],
) -> Dict[Tuple[str, str], Path]:
    pair_template_paths: Dict[Tuple[str, str], Path] = {}

    for channel_a, channel_b in channel_pairs:
        template = copy.deepcopy(template_data)

        layer_a_cc = copy.deepcopy(cc_layers[channel_a])
        layer_a_cc["shader"] = (
            "#uicontrol vec3 color color(default=\"#ff0000\")\n"
            "#uicontrol invlerp normalized\n"
            "void main() {\n"
            "emitRGB(color * normalized());\n"
            "}"
        )

        layer_b_cc = copy.deepcopy(cc_layers[channel_b])
        layer_b_cc["shader"] = (
            "#uicontrol vec3 color color(default=\"#00ff00\")\n"
            "#uicontrol invlerp normalized\n"
            "void main() {\n"
            "emitRGB(color * normalized());\n"
            "}"
        )

        layer_a_rc = copy.deepcopy(rc_layers[channel_a])
        layer_a_rc["shader"] = (
            "#uicontrol vec3 color color(default=\"#00ff00\")\n"
            "#uicontrol invlerp normalized\n"
            "void main() {\n"
            "emitRGB(color * normalized());\n"
            "}"
        )

        layer_b_rc = copy.deepcopy(rc_layers[channel_b])
        layer_b_rc["shader"] = (
            "#uicontrol vec3 color color(default=\"#ff0000\")\n"
            "#uicontrol invlerp normalized\n"
            "void main() {\n"
            "emitRGB(color * normalized());\n"
            "}"
        )

        template["layers"] = [layer_a_cc, layer_b_cc, layer_a_rc, layer_b_rc]
        template["dimensions"] = template.get("dimensions", {})

        output_path = output_dir / f"cc_ng_{channel_a}_{channel_b}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(template, handle, indent=2)

        _upload_json_to_s3(output_path, s3_json_base)

        pair_template_paths[(channel_a, channel_b)] = output_path

    return pair_template_paths


def _load_tile_stack(
    tilename_s3: str,
    data_dir: Path,
    planes: Sequence[int],
    pyramid_level: str,
) -> np.ndarray:
    local_path = convert_s3_to_local(tilename_s3, data_dir)
    tile_zarr = da.from_zarr(local_path.as_posix(), pyramid_level)
    # Expecting shape (C, T, Z, Y, X); select first channel/time to mirror scientist code
    tile_stack = tile_zarr[0, 0, planes, ...].compute()
    return np.asarray(tile_stack)


def _apply_affine_stack(stack: np.ndarray, affine: np.ndarray) -> np.ndarray:
    transformed = np.zeros_like(stack, dtype=np.float32)
    for index, plane in enumerate(stack):
        transformed[index, ...] = apply_affine_to_image(plane, affine)
    return transformed


def _compute_planes(tile_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, int, int]:
    z_dim, y_dim, x_dim = tile_shape
    thickness = z_dim // 3
    spacing = max(1, z_dim // 5)
    planes = np.arange((z_dim - thickness) // 2, (z_dim + thickness) // 2, spacing)
    return planes, y_dim, x_dim


def _generate_histogram_axes(
    num_planes: int,
    num_tiles: int,
    num_pairs: int,
) -> Tuple[np.ndarray, np.ndarray, plt.Figure, plt.Figure]:
    if num_planes == 0 or num_tiles == 0 or num_pairs == 0:
        return (
            np.empty((0, 0)),
            np.empty((0, 0)),
            plt.figure(),
            plt.figure(),
        )

    rows = num_planes * num_tiles
    fig_hist, axs_hist = plt.subplots(
        rows,
        num_pairs,
        figsize=(5 * num_pairs, 5 * rows),
    )
    fig_scatter, axs_scatter = plt.subplots(
        rows,
        num_pairs,
        figsize=(5 * num_pairs, 5 * rows),
    )
    return (
        np.atleast_2d(axs_hist),
        np.atleast_2d(axs_scatter),
        fig_hist,
        fig_scatter,
    )


def _update_histograms(
    ax_hist: plt.Axes,
    ax_scatter: plt.Axes,
    title: str,
    correspond_points_1: np.ndarray,
    correspond_points_2: np.ndarray,
    affine_1: np.ndarray,
    affine_2: np.ndarray,
) -> None:
    if correspond_points_1.size == 0 or correspond_points_2.size == 0:
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])
        return

    homogeneous_1 = np.vstack([correspond_points_1.T, np.ones(correspond_points_1.shape[0])])
    homogeneous_2 = np.vstack([correspond_points_2.T, np.ones(correspond_points_2.shape[0])])

    transformed_1 = (affine_1 @ homogeneous_1)[:2, :].T.astype(int)
    transformed_2 = (affine_2 @ homogeneous_2)[:2, :].T.astype(int)

    distance_pre = np.linalg.norm(correspond_points_1 - correspond_points_2, axis=1)
    distance_post = np.linalg.norm(transformed_1 - transformed_2, axis=1)

    ax_hist.hist(distance_pre, 10, alpha=0.5, label="pre-correction", color="r")
    ax_hist.hist(distance_post, 10, alpha=0.5, label="post-correction", color="g")
    ax_hist.legend()
    ax_hist.set_xlabel("distance (pixels)")
    ax_hist.set_ylabel("number of points")
    ax_hist.set_title(title)

    max_dist = float(np.max(np.hstack([distance_pre, distance_post])))
    ax_scatter.scatter(distance_pre, distance_post)
    ax_scatter.plot([0, max_dist], [0, max_dist], "k--", alpha=0.5)
    ax_scatter.set_xlabel("distance pre-correction (pixels)")
    ax_scatter.set_ylabel("distance post-correction (pixels)")
    ax_scatter.set_title(title)


def _find_peak_regions(
    points_1: np.ndarray,
    points_2: np.ndarray,
    n_bins: int,
    tile_height: int,
    tile_width: int,
    top_n_points: int,
) -> np.ndarray:
    if points_1.size == 0 or points_2.size == 0:
        return np.empty((0, 2))

    hist_1 = gaussian_filter(
        np.histogram2d(
            points_1[:, 0],
            points_1[:, 1],
            bins=n_bins,
            range=[[0, tile_height], [0, tile_width]],
        )[0],
        sigma=1,
    )
    hist_2 = gaussian_filter(
        np.histogram2d(
            points_2[:, 0],
            points_2[:, 1],
            bins=n_bins,
            range=[[0, tile_height], [0, tile_width]],
        )[0],
        sigma=1,
    )

    combined = hist_1 + hist_2
    peak_indices = peak_local_max(
        combined,
        min_distance=2,
        threshold_rel=0.2,
        num_peaks=top_n_points,
    )

    if peak_indices.size == 0:
        return np.empty((0, 2))

    x_locs = (
        np.linspace(0, tile_width, n_bins + 1)[1:] +
        np.linspace(0, tile_width, n_bins + 1)[:-1]
    ) // 2
    y_locs = (
        np.linspace(0, tile_height, n_bins + 1)[1:] +
        np.linspace(0, tile_height, n_bins + 1)[:-1]
    ) // 2

    return np.array([(x_locs[idx[1]], y_locs[idx[0]]) for idx in peak_indices])


def _normalise_channel_range(image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    clipped = np.clip(image, vmin, vmax)
    normalised = (clipped - vmin) / (vmax - vmin)
    return normalised.astype(np.float32)


def _apply_intensity_scaling(
    layer: Dict,
    intensity_range: Tuple[float, float],
    visible: bool,
) -> None:
    layer.setdefault("shaderControls", {}).setdefault("normalized", {})["range"] = [
        float(intensity_range[0]),
        float(intensity_range[1]),
    ]
    layer["visible"] = visible


def _adjust_transform_offsets(json_data: Dict) -> None:
    layers = json_data["layers"]
    cc_a, cc_b, rc_a, rc_b = layers

    cc_a_matrix = cc_a["source"][0]["transform"]["matrix"]
    cc_b_matrix = cc_b["source"][0]["transform"]["matrix"]
    rc_a_matrix = rc_a["source"][0]["transform"]["matrix"]
    rc_b_matrix = rc_b["source"][0]["transform"]["matrix"]

    cc_a_matrix[-1][-1] -= rc_a_matrix[-1][-1]
    cc_a_matrix[-2][-1] -= rc_a_matrix[-2][-1]
    cc_b_matrix[-1][-1] -= rc_b_matrix[-1][-1]
    cc_b_matrix[-2][-1] -= rc_b_matrix[-2][-1]

    rc_a_matrix[-1][-1] = 0.0
    rc_a_matrix[-2][-1] = 0.0
    rc_b_matrix[-1][-1] = 0.0
    rc_b_matrix[-2][-1] = 0.0


def _create_zoom_visualisation(
    channel_pair: Tuple[str, str],
    dataset_name: str,
    tile_index: int,
    plane_index: int,
    plane_value: int,
    peaks: np.ndarray,
    tilenames: Dict[str, List[str]],
    pair_template_path: Path,
    output_dirs: Tuple[Path, Path, Path, Path],
    s3_json_base: Optional[str],
    data_dir: Path,
    settings: QCSettings,
    tile_1_stack: np.ndarray,
    tile_2_stack: np.ndarray,
    tile_1_transformed: np.ndarray,
    tile_2_transformed: np.ndarray,
) -> None:
    output_dir, png_dir, pdf_dir, json_dir = output_dirs
    channel_a, channel_b = channel_pair

    for peak in peaks:
        x_loc = int(peak[0])
        y_loc = int(peak[1])

        y_slice = _clamp_slice(y_loc, settings.clip_half_width, tile_1_stack.shape[1])
        x_slice = _clamp_slice(x_loc, settings.clip_half_width, tile_1_stack.shape[2])

        tile_1_clip = tile_1_stack[plane_index, y_slice, x_slice]
        tile_2_clip = tile_2_stack[plane_index, y_slice, x_slice]
        tile_1_transformed_clip = tile_1_transformed[plane_index, y_slice, x_slice]
        tile_2_transformed_clip = tile_2_transformed[plane_index, y_slice, x_slice]

        vmin_1, vmax_1 = np.percentile(tile_1_clip, [10, 99.99])
        vmin_2, vmax_2 = np.percentile(tile_2_clip, [10, 99.99])
        vmin_1_trans, vmax_1_trans = np.percentile(tile_1_transformed_clip, [10, 99.99])
        vmin_2_trans, vmax_2_trans = np.percentile(tile_2_transformed_clip, [10, 99.99])

        base_json = load_json(pair_template_path)
        for layer in base_json["layers"]:
            if layer["name"].startswith(channel_a):
                target_url = tilenames[channel_a][tile_index]
                layer["source"] = [s for s in layer["source"] if s["url"] == target_url]
                if layer["name"].endswith("_cc"):
                    _apply_intensity_scaling(layer, (vmin_1_trans, vmax_1_trans), True)
                else:
                    _apply_intensity_scaling(layer, (vmin_1, vmax_1), False)
            elif layer["name"].startswith(channel_b):
                target_url = tilenames[channel_b][tile_index]
                layer["source"] = [s for s in layer["source"] if s["url"] == target_url]
                if layer["name"].endswith("_cc"):
                    _apply_intensity_scaling(layer, (vmin_2_trans, vmax_2_trans), True)
                else:
                    _apply_intensity_scaling(layer, (vmin_2, vmax_2), False)

        base_json["position"] = [x_loc, y_loc, int(plane_value), 0]
        base_json["crossSectionScale"] = settings.cross_section_scale
        base_json["projectionScale"] = settings.projection_scale

        _adjust_transform_offsets(base_json)

        tile_token = Path(tilenames[channel_a][tile_index]).name[:18]
        json_filename = (
            f"cc_ng_{channel_a}_{channel_b}_{tile_token}_x{x_loc}_y{y_loc}_z{plane_value}_zoomed.json"
        )
        json_path = json_dir / json_filename
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(base_json, handle, indent=2)

        _upload_json_to_s3(json_path, s3_json_base)

        if s3_json_base:
            ng_link = (
                "https://neuroglancer-demo.appspot.com/#!"
                f"{s3_json_base}{json_filename}"
            )
        else:
            ng_link = (
                "https://neuroglancer-demo.appspot.com/#!"
                f"s3://aind-open-data/{dataset_name}/image_cross_image_alignment/{json_filename}"
            )

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(
            (
                f"{channel_a.replace('CH_', '')} vs. {channel_b.replace('CH_', '')}, "
                f"tile = {tile_token} - x={x_loc}, y={y_loc}, z={plane_value}"
            ),
            y=0.95,
        )

        pre_rgb = np.zeros((tile_1_clip.shape[0], tile_1_clip.shape[1], 3), dtype=np.float32)
        pre_rgb[..., 0] = _normalise_channel_range(tile_1_clip, vmin_1, vmax_1)
        pre_rgb[..., 1] = _normalise_channel_range(tile_2_clip, vmin_2, vmax_2)

        post_rgb = np.zeros((tile_1_transformed_clip.shape[0], tile_1_transformed_clip.shape[1], 3), dtype=np.float32)
        post_rgb[..., 0] = _normalise_channel_range(tile_1_transformed_clip, vmin_1_trans, vmax_1_trans)
        post_rgb[..., 1] = _normalise_channel_range(tile_2_transformed_clip, vmin_2_trans, vmax_2_trans)

        axes[0].imshow(pre_rgb, aspect="auto")
        axes[0].set_title("Pre-correction")
        axes[0].axis("off")

        axes[1].imshow(post_rgb, aspect="auto")
        axes[1].set_title("Post-correction")
        axes[1].axis("off")

        fig.text(0.9, 0.08, "neuroglancer link", ha="right", color="blue", url=ng_link)

        pdf_filename = (
            f"{channel_a.replace('CH_', '')}vs{channel_b.replace('CH_', '')}_{tile_token}"
            f"_x{x_loc}_y{y_loc}_z{plane_value}.pdf"
        )
        png_filename = pdf_filename.replace(".pdf", ".png")

        pdf_path = pdf_dir / pdf_filename
        png_path = png_dir / png_filename

        if PdfPages is None:  # pragma: no cover - guarded earlier
            raise RuntimeError("PdfPages unavailable; cannot write QC PDFs.")

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _process_channel_pair(
    channel_pair: Tuple[str, str],
    dataset_name: str,
    tile_index: int,
    tile_row_index: int,
    planes: Sequence[int],
    axs_hist: np.ndarray,
    axs_scatter: np.ndarray,
    settings: QCSettings,
    tilenames: Dict[str, List[str]],
    affine_matrices: Dict[str, np.ndarray],
    output_dirs: Tuple[Path, Path, Path, Path],
    pair_template_paths: Dict[Tuple[str, str], Path],
    s3_json_base: Optional[str],
    data_dir: Path,
    tile_height: int,
    tile_width: int,
    channel_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]],
) -> None:
    channel_a, channel_b = channel_pair

    def fetch_channel_stack(channel: str) -> Tuple[np.ndarray, np.ndarray]:
        key = (channel, tile_index)
        if key not in channel_cache:
            stack = _load_tile_stack(
                tilenames[channel][tile_index],
                data_dir,
                planes,
                settings.pyramid_level,
            )
            transformed = _apply_affine_stack(stack, affine_matrices[channel])
            channel_cache[key] = (stack, transformed)
        return channel_cache[key]

    tile_1_stack, tile_1_transformed = fetch_channel_stack(channel_a)
    tile_2_stack, tile_2_transformed = fetch_channel_stack(channel_b)

    peaks_per_plane: Dict[int, np.ndarray] = {}

    for plane_offset, plane_value in enumerate(planes):
        row_index = tile_row_index * len(planes) + plane_offset
        ax_hist_row = axs_hist[row_index]
        ax_scatter_row = axs_scatter[row_index]
        pair_index = list(pair_template_paths.keys()).index(channel_pair)
        ax_hist = ax_hist_row[pair_index]
        ax_scatter = ax_scatter_row[pair_index]

        title = (
            f"{channel_a.replace('CH_', '')} vs. {channel_b.replace('CH_', '')}, "
            f"tile {tile_index} - z={plane_value}"
        )

        points_1 = blob_dog(
            tile_1_stack[plane_offset, ...].T.astype(np.float32),
            min_sigma=1,
            max_sigma=1.5,
            threshold=settings.dot_threshold,
        )
        points_2 = blob_dog(
            tile_2_stack[plane_offset, ...].T.astype(np.float32),
            min_sigma=1,
            max_sigma=1.5,
            threshold=settings.dot_threshold,
        )

        if points_1.size == 0 or points_2.size == 0:
            _update_histograms(ax_hist, ax_scatter, title, np.empty((0, 2)), np.empty((0, 2)), affine_matrices[channel_a], affine_matrices[channel_b])
            peaks_per_plane[plane_value] = np.empty((0, 2))
            continue

        intensities_1 = tile_1_stack[plane_offset, ...].T[
            points_1[:, 0].astype(np.uint16), points_1[:, 1].astype(np.uint16)
        ]
        intensities_2 = tile_2_stack[plane_offset, ...].T[
            points_2[:, 0].astype(np.uint16), points_2[:, 1].astype(np.uint16)
        ]

        points_1 = points_1[np.flip(np.argsort(intensities_1))[: settings.dot_num], :-1]
        points_2 = points_2[np.flip(np.argsort(intensities_2))[: settings.dot_num], :-1]

        correspond_indices = match_descriptors(
            points_1,
            points_2,
            max_distance=6,
            max_ratio=0.8,
        )

        if correspond_indices.size == 0:
            _update_histograms(ax_hist, ax_scatter, title, np.empty((0, 2)), np.empty((0, 2)), affine_matrices[channel_a], affine_matrices[channel_b])
            peaks_per_plane[plane_value] = np.empty((0, 2))
            continue

        correspond_points_1 = points_1[correspond_indices[:, 0], :]
        correspond_points_2 = points_2[correspond_indices[:, 1], :]

        _update_histograms(
            ax_hist,
            ax_scatter,
            title,
            correspond_points_1,
            correspond_points_2,
            affine_matrices[channel_a],
            affine_matrices[channel_b],
        )

        peak_regions = _find_peak_regions(
            correspond_points_1,
            correspond_points_2,
            settings.n_bins,
            tile_height,
            tile_width,
            settings.top_n_points,
        )
        peaks_per_plane[plane_value] = peak_regions

    for plane_offset, plane_value in enumerate(planes):
        peak_regions = peaks_per_plane.get(plane_value, np.empty((0, 2)))
        if peak_regions.size == 0:
            continue
        _create_zoom_visualisation(
            channel_pair,
            dataset_name,
            tile_index,
            plane_offset,
            int(plane_value),
            peak_regions,
            tilenames,
            pair_template_paths[channel_pair],
            output_dirs,
            s3_json_base,
            data_dir,
            settings,
            tile_1_stack,
            tile_2_stack,
            tile_1_transformed,
            tile_2_transformed,
        )


def _merge_pair_pdfs(
    channel_pairs: Sequence[Tuple[str, str]],
    pdf_dir: Path,
    output_dir: Path,
) -> None:
    for channel_a, channel_b in channel_pairs:
        pdf_list = [
            pdf_path
            for pdf_path in pdf_dir.glob("*.pdf")
            if channel_a.replace("CH_", "") in pdf_path.name
            and channel_b.replace("CH_", "") in pdf_path.name
        ]
        if not pdf_list:
            continue
        merged_path = output_dir / (
            f"{channel_a.replace('CH_', '')}vs{channel_b.replace('CH_', '')}_merged.pdf"
        )
        merge_pdfs(pdf_list, merged_path)


def generate_camera_alignment_qc(
    dataset_names: Optional[Sequence[str]] = None,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    scratch_root: Path = DEFAULT_RESULTS_ROOT,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    settings: Optional[QCSettings] = None,
) -> Dict[str, Path]:
    """Generate QC artefacts for one or more datasets.

    Parameters
    ----------
    dataset_names:
        Specific dataset names to process. If omitted, all datasets containing
        `image_radial_correction` are processed.
    data_dir:
        Root directory containing dataset folders and neuroglancer JSON files.
    scratch_root:
        Destination for generated QC artefacts.
    template_path:
        Path to the neuroglancer layer template JSON file.
    settings:
        Optional QCSettings override.

    Returns
    -------
    dict
        Mapping from dataset name to the output directory containing QC results.
    """

    ensure_advanced_qc_available()
    settings = settings or QCSettings()

    datasets = _resolve_datasets(dataset_names, data_dir)
    if not datasets:
        raise FileNotFoundError("No datasets found for QC processing.")

    template_data = load_json(template_path)
    results: Dict[str, Path] = {}

    for dataset_path in datasets:
        dataset_name = dataset_path.name
        LOGGER.info("Processing camera-alignment QC for dataset %s", dataset_name)

        cc_json = dataset_path / "camera_aligned_neuroglancer.json"
        rc_json = dataset_path / "radially_corrected_neuroglancer.json"

        if not cc_json.exists() or not rc_json.exists():
            raise FileNotFoundError(
                f"Required neuroglancer JSON files missing for {dataset_name}: "
                f"{cc_json}, {rc_json}"
            )

        cc_data = load_json(cc_json)
        rc_data = load_json(rc_json)

        channel_names = sorted(layer["name"] for layer in cc_data["layers"])
        if "CH_405" in channel_names:
            channel_names.remove("CH_405")
        if len(channel_names) < 2:
            raise ValueError("QC generation requires at least two channels.")

        channel_pairs = [
            (channel_names[idx], channel_names[idx + 1])
            for idx in range(len(channel_names) - 1)
        ]

        cc_layers, rc_layers = _prepare_channel_layers(cc_data, rc_data, channel_names)
        tilenames, affine_matrices = _extract_affine_metadata(cc_layers, channel_names)
        s3_json_base = _determine_s3_json_base(tilenames)

        output_dirs = _prepare_output_dirs(scratch_root, dataset_name)
        output_dir, _, pdf_dir, _ = output_dirs

        pair_template_paths = _create_pair_templates(
            template_data,
            cc_layers,
            rc_layers,
            channel_pairs,
            output_dir,
            s3_json_base,
        )

        first_channel = channel_names[0]
        example_tile = convert_s3_to_local(tilenames[first_channel][0], data_dir)
        tile_shape = da.from_zarr(example_tile.as_posix(), settings.pyramid_level).shape[2:]
        planes, tile_height, tile_width = _compute_planes(tile_shape)

        tiles_available = len(tilenames[first_channel])
        tiles_to_process = [
            idx for idx in settings.tiles_to_check if idx < tiles_available
        ]

        if not tiles_to_process:
            LOGGER.warning(
                "No tiles available within configured indices %s for dataset %s",
                settings.tiles_to_check,
                dataset_name,
            )
            continue

        axs_hist, axs_scatter, fig_hist, fig_scatter = _generate_histogram_axes(
            len(planes),
            len(tiles_to_process),
            len(channel_pairs),
        )

        channel_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}

        for tile_row_index, tile_index in enumerate(tiles_to_process):
            for channel_pair in channel_pairs:
                _process_channel_pair(
                    channel_pair,
                    dataset_name,
                    tile_index,
                    tile_row_index,
                    planes,
                    axs_hist,
                    axs_scatter,
                    settings,
                    tilenames,
                    affine_matrices,
                    output_dirs,
                    pair_template_paths,
                    s3_json_base,
                    data_dir,
                    tile_height,
                    tile_width,
                    channel_cache,
                )

        plt.close(fig_hist)
        plt.close(fig_scatter)

        _merge_pair_pdfs(channel_pairs, pdf_dir, output_dir)
        results[dataset_name] = output_dir

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_camera_alignment_qc()
