"""Colony segmentation: locate a single bacterial colony biofilm in a plate photograph.

The pipeline is deliberately classical (no learned models): flatten uneven
illumination, threshold with Otsu's method, clean up the binary mask
morphologically, and keep only the largest connected component. Each stage
is returned so the GUI can show the user exactly what happened and why a
segmentation succeeded or failed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, io, measure, morphology, segmentation, transform


@dataclass
class SegmentationResult:
    """Intermediate and final outputs of the segmentation pipeline.

    Attributes
    ----------
    original : np.ndarray
        The input image, unmodified (H, W, 3) uint8.
    gray : np.ndarray
        Grayscale version of the input, float64 in [0, 1].
    illumination_corrected : np.ndarray
        Grayscale image after background flattening, float64 in [0, 1].
    threshold_mask : np.ndarray
        Raw boolean mask straight out of thresholding, before cleanup.
    cleaned_mask : np.ndarray
        Boolean mask after morphological cleanup and hole filling.
    colony_mask : np.ndarray
        Final boolean mask containing only the largest connected component.
        This is the mask downstream metrics should be computed on.
    threshold_value : float
        The grayscale threshold actually used (Otsu value + offset, or the
        manual override).
    """

    original: np.ndarray
    gray: np.ndarray
    illumination_corrected: np.ndarray
    threshold_mask: np.ndarray
    cleaned_mask: np.ndarray
    colony_mask: np.ndarray
    threshold_value: float


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as an RGB uint8 array.

    Parameters
    ----------
    path : str
        Path to a .jpg, .jpeg, .png, or .tif image file.

    Returns
    -------
    np.ndarray
        Image array of shape (H, W, 3), dtype uint8. Alpha channels and
        grayscale-only source files are converted to 3-channel RGB so the
        rest of the pipeline can assume a consistent shape.
    """
    image = io.imread(path)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Grayscale image, float64 in [0, 1].
    """
    return color.rgb2gray(image)


def correct_illumination(
    gray: np.ndarray, kernel_radius: int = 200, downsample_size: int = 200
) -> np.ndarray:
    """Flatten uneven plate illumination via morphological background subtraction.

    Photographs of Petri dishes are rarely lit perfectly evenly (lightbox hot
    spots, shadowing near the dish rim). A morphological opening with a disk
    larger than the colony gives a smooth estimate of the background that
    "skips over" the colony entirely (fills it in with the surrounding agar
    tone); subtracting that estimate removes slow lighting drift while
    leaving the colony's own contrast against the agar intact. This is the
    "rolling ball" background subtraction trick commonly used in microscopy.

    Grayscale opening with a large disk is expensive directly on a full-size
    photograph (disk footprints aren't separable, so cost grows sharply with
    radius). Since illumination drift and colony shape are both low
    frequency, the background is instead estimated on a small downsampled
    copy of the image and then resized back up — this keeps the cost
    roughly constant regardless of photo resolution or `kernel_radius`.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image, float64 in [0, 1].
    kernel_radius : int, default 200
        Radius (pixels, in the *original* image's scale) of the disk
        structuring element used to estimate background. Must exceed the
        colony's radius, otherwise opening reconstructs the colony instead
        of skipping over it and the correction erases the colony's contrast.
        Increase this if a large/spread colony vanishes after correction.
    downsample_size : int, default 200
        Long-edge size (pixels) of the working copy used for the
        background estimate. Smaller is faster; too small blurs away real
        illumination structure. 200 px is a reasonable balance.

    Returns
    -------
    np.ndarray
        Illumination-corrected grayscale image, float64, rescaled to [0, 1].
    """
    # Disk-footprint opening is only cheap for small radii (its cost grows
    # sharply with radius, since the footprint isn't separable). Choose a
    # downsample scale that keeps the *working* radius small regardless of
    # how large `kernel_radius` or the source image are, so this stays fast
    # and memory-safe even for a huge photo or an aggressive kernel radius.
    max_working_radius = 25
    scale = min(1.0, max_working_radius / kernel_radius, downsample_size / max(gray.shape))
    if scale < 1.0:
        small_shape = (round(gray.shape[0] * scale), round(gray.shape[1] * scale))
        small = transform.resize(gray, small_shape, anti_aliasing=True)
        small_radius = max(1, round(kernel_radius * scale))
    else:
        small = gray
        small_radius = kernel_radius

    selem = morphology.disk(small_radius)
    small_background = morphology.opening(small, selem)
    background = transform.resize(small_background, gray.shape, anti_aliasing=True)

    corrected = gray - background
    # Rescale so downstream Otsu thresholding sees a full-contrast image
    # regardless of how much the correction compressed the dynamic range.
    corrected_min, corrected_max = corrected.min(), corrected.max()
    if corrected_max > corrected_min:
        corrected = (corrected - corrected_min) / (corrected_max - corrected_min)
    return corrected


def threshold_colony(
    corrected: np.ndarray,
    threshold_offset: float = 0.0,
    manual_threshold: float | None = None,
    colony_darker_than_background: bool = True,
) -> tuple[np.ndarray, float]:
    """Binarise the illumination-corrected image to separate colony from agar.

    Parameters
    ----------
    corrected : np.ndarray
        Illumination-corrected grayscale image, float64 in [0, 1], as
        produced by `correct_illumination`.
    threshold_offset : float, default 0.0
        Added to the Otsu threshold before applying it. When the colony is
        darker than the background (the default), raising the threshold
        classifies more pixels as colony (grows the foreground); lowering
        it shrinks the foreground. This is reversed if
        `colony_darker_than_background` is False. Use this to tune
        segmentation without abandoning automatic thresholding.
    manual_threshold : float or None, default None
        If given, overrides Otsu's method entirely and is used as-is
        (ignoring `threshold_offset`). Useful for plates where automatic
        thresholding fails, e.g. very low colony/background contrast.
    colony_darker_than_background : bool, default True
        Whether the colony is expected to be darker than the surrounding
        agar in the corrected image. Flip this if your lighting setup makes
        colonies appear brighter than the plate (e.g. top-lit opaque
        biofilms against a dark background).

    Returns
    -------
    mask : np.ndarray
        Boolean array, True where a pixel is classified as colony.
    threshold_value : float
        The threshold actually applied.
    """
    if manual_threshold is not None:
        threshold_value = manual_threshold
    else:
        threshold_value = filters.threshold_otsu(corrected) + threshold_offset

    if colony_darker_than_background:
        mask = corrected < threshold_value
    else:
        mask = corrected > threshold_value
    return mask, threshold_value


def clean_mask(
    mask: np.ndarray,
    min_object_size: int = 500,
    morph_kernel_size: int = 5,
    clear_border: bool = True,
) -> np.ndarray:
    """Morphologically clean a raw threshold mask and fill interior holes.

    Removes speckle noise (opening), reconnects slightly broken colony
    edges (closing), discards small unconnected debris below a size
    threshold, optionally discards anything touching the frame edge, and
    fills any holes left inside the colony outline (e.g. from surface
    texture or specular highlights) since we want a solid footprint, not a
    perforated one.

    Parameters
    ----------
    mask : np.ndarray
        Raw boolean mask from thresholding.
    min_object_size : int, default 500
        Minimum object area (pixels) to keep. Anything smaller is treated
        as noise, not colony.
    morph_kernel_size : int, default 5
        Radius of the disk structuring element used for opening/closing.
    clear_border : bool, default True
        Discard objects touching the image border before hole-filling. A
        single colony photographed with the dish fully in frame should
        never itself touch the border; things that do (the bench/backdrop
        outside the dish, a partially cropped dish) are not the colony, and
        left in place they can bridge into a single border-to-border blob
        once holes are filled. Disable only if the colony is expected to
        run off the edge of the frame.

    Returns
    -------
    np.ndarray
        Cleaned boolean mask.
    """
    selem = morphology.disk(morph_kernel_size)
    cleaned = morphology.opening(mask, selem)
    cleaned = morphology.closing(cleaned, selem)
    cleaned = morphology.remove_small_objects(cleaned, max_size=min_object_size)
    if clear_border:
        cleaned = segmentation.clear_border(cleaned)
    cleaned = ndi.binary_fill_holes(cleaned)
    return cleaned


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask.

    A plate photograph can contain contamination, condensation droplets, or
    dish-rim artefacts that survive cleanup as small separate blobs. Since
    this tool assumes a single colony per plate, the largest connected
    region is taken to be the colony and everything else is discarded.

    Parameters
    ----------
    mask : np.ndarray
        Cleaned boolean mask, possibly containing multiple objects.

    Returns
    -------
    np.ndarray
        Boolean mask containing only the largest connected component. If
        the input mask is empty, an all-False mask of the same shape is
        returned.
    """
    labels = measure.label(mask)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    regions = measure.regionprops(labels)
    largest = max(regions, key=lambda r: r.area)
    return labels == largest.label


def segment_colony(
    image: np.ndarray,
    threshold_offset: float = 0.0,
    manual_threshold: float | None = None,
    colony_darker_than_background: bool = True,
    min_object_size: int = 500,
    morph_kernel_size: int = 5,
    illumination_kernel_radius: int = 200,
    clear_border: bool = True,
) -> SegmentationResult:
    """Run the full segmentation pipeline on a single plate photograph.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), as returned by `load_image`.
    threshold_offset : float, default 0.0
        See `threshold_colony`.
    manual_threshold : float or None, default None
        See `threshold_colony`.
    colony_darker_than_background : bool, default True
        See `threshold_colony`.
    min_object_size : int, default 500
        See `clean_mask`.
    morph_kernel_size : int, default 5
        See `clean_mask`.
    illumination_kernel_radius : int, default 200
        See `correct_illumination`.
    clear_border : bool, default True
        See `clean_mask`.

    Returns
    -------
    SegmentationResult
        All intermediate stages plus the final single-component colony mask.
    """
    gray = to_grayscale(image)
    corrected = correct_illumination(gray, kernel_radius=illumination_kernel_radius)
    raw_mask, threshold_value = threshold_colony(
        corrected,
        threshold_offset=threshold_offset,
        manual_threshold=manual_threshold,
        colony_darker_than_background=colony_darker_than_background,
    )
    cleaned = clean_mask(
        raw_mask,
        min_object_size=min_object_size,
        morph_kernel_size=morph_kernel_size,
        clear_border=clear_border,
    )
    colony_mask = largest_component(cleaned)

    return SegmentationResult(
        original=image,
        gray=gray,
        illumination_corrected=corrected,
        threshold_mask=raw_mask,
        cleaned_mask=cleaned,
        colony_mask=colony_mask,
        threshold_value=threshold_value,
    )


if __name__ == "__main__":
    # Quick eyeball check: `python -m src.segment path/to/image.png`
    # This is a developer convenience, not the package's CLI entry point
    # (that's src/cli.py).
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Preview colony segmentation on one image.")
    parser.add_argument("image_path", help="Path to a plate photograph.")
    parser.add_argument("--threshold-offset", type=float, default=0.0)
    parser.add_argument("--min-object-size", type=int, default=500)
    parser.add_argument("--morph-kernel-size", type=int, default=5)
    args = parser.parse_args()

    img = load_image(args.image_path)
    result = segment_colony(
        img,
        threshold_offset=args.threshold_offset,
        min_object_size=args.min_object_size,
        morph_kernel_size=args.morph_kernel_size,
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(result.original)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(result.gray, cmap="gray")
    axes[0, 1].set_title("Grayscale")
    axes[0, 2].imshow(result.illumination_corrected, cmap="gray")
    axes[0, 2].set_title("Illumination-corrected")
    axes[1, 0].imshow(result.threshold_mask, cmap="gray")
    axes[1, 0].set_title(f"Raw threshold (t={result.threshold_value:.3f})")
    axes[1, 1].imshow(result.cleaned_mask, cmap="gray")
    axes[1, 1].set_title("Cleaned mask")
    axes[1, 2].imshow(result.original)
    axes[1, 2].contour(result.colony_mask, colors="lime", linewidths=2)
    axes[1, 2].set_title("Final colony boundary")

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    plt.show()
