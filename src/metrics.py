"""Shape and texture metrics for a segmented colony biofilm footprint.

Every function takes a boolean colony mask (as produced by
`segment.segment_colony`) and returns a single scalar. Shape metrics
optionally convert to millimetres given a `mm_per_pixel` calibration factor
from `calibrate.py`; texture metrics are computed on the underlying
grayscale pixel values and are unit-independent.
"""

from __future__ import annotations

import numpy as np
from skimage import feature, measure

# Small constant to avoid log2(0) in entropy calculations.
_EPS = 1e-12


def area(mask: np.ndarray, mm_per_pixel: float | None = None) -> float:
    """Colony footprint area.

    Biological interpretation
    --------------------------
    The total substrate area colonised by the biofilm. Combined with
    inoculation density and incubation time, this is the standard readout
    for radial expansion rate in colony biofilm assays.

    Parameters
    ----------
    mask : np.ndarray
        Boolean colony mask.
    mm_per_pixel : float or None, default None
        Calibration factor from `calibrate.py`. If None, area is returned
        in pixels squared.

    Returns
    -------
    float
        Area in mm^2 if calibrated, else px^2.
    """
    pixel_area = float(mask.sum())
    if mm_per_pixel is not None:
        return pixel_area * mm_per_pixel**2
    return pixel_area


def perimeter(mask: np.ndarray, mm_per_pixel: float | None = None) -> float:
    """Colony boundary length.

    Biological interpretation
    --------------------------
    A rough, highly folded biofilm margin has a much longer perimeter than
    a smooth colony of the same area. On its own perimeter is dominated by
    colony size, which is why it is normally interpreted alongside area via
    `circularity` rather than in isolation.

    Uses the Crofton perimeter estimator (`measure.perimeter_crofton`),
    which is less biased by pixel-grid staircasing than counting boundary
    pixels directly, and so gives a more accurate circularity for
    near-circular colonies.

    Parameters
    ----------
    mask : np.ndarray
        Boolean colony mask.
    mm_per_pixel : float or None, default None
        Calibration factor from `calibrate.py`. If None, perimeter is
        returned in pixels.

    Returns
    -------
    float
        Perimeter in mm if calibrated, else px.
    """
    pixel_perimeter = measure.perimeter_crofton(mask, directions=4)
    if mm_per_pixel is not None:
        return pixel_perimeter * mm_per_pixel
    return pixel_perimeter


def circularity(mask: np.ndarray) -> float:
    """Isoperimetric ratio 4*pi*Area / Perimeter^2 -- a wrinkling proxy.

    Biological interpretation
    --------------------------
    Equals 1.0 for a perfect circle and decreases as the margin becomes
    more convoluted relative to the area it encloses. Matrix-deficient
    B. subtilis mutants (e.g. eps or tasA knockouts) produce smooth,
    near-circular colonies with circularity close to 1; wild-type colonies
    develop wrinkled, radially furrowed architecture that drives
    circularity well below 1 even though the colony may still be
    roughly disc-shaped in outline. This is scale- and calibration-free:
    since area scales as length^2 and perimeter as length, the ratio is
    dimensionless and identical whether computed in pixels or mm.

    Parameters
    ----------
    mask : np.ndarray
        Boolean colony mask.

    Returns
    -------
    float
        Circularity in (0, 1] for a simply-connected shape; 0.0 if the
        mask is empty.
    """
    pixel_area = float(mask.sum())
    pixel_perimeter = measure.perimeter_crofton(mask, directions=4)
    if pixel_perimeter == 0:
        return 0.0
    return 4 * np.pi * pixel_area / pixel_perimeter**2


def solidity(mask: np.ndarray) -> float:
    """Ratio of colony area to its convex hull area -- a lobing/notching proxy.

    Biological interpretation
    --------------------------
    Captures margin indentation independently of fine-scale surface
    wrinkling: a colony with deep radial furrows or finger-like lobes
    encloses much less area than its convex hull, giving low solidity,
    whereas a colony that is wrinkled on its *surface* but still has a
    smooth, unbroken outer margin can have high solidity despite low
    circularity. Comparing solidity and circularity together separates
    margin shape (solidity) from surface/edge roughness (circularity).

    Parameters
    ----------
    mask : np.ndarray
        Boolean colony mask, expected to be a single connected component
        (as produced by `segment.largest_component`).

    Returns
    -------
    float
        Solidity in (0, 1]; 0.0 if the mask is empty.
    """
    if not mask.any():
        return 0.0
    labeled = measure.label(mask)
    region = measure.regionprops(labeled)[0]
    return float(region.solidity)


def equivalent_diameter(mask: np.ndarray, mm_per_pixel: float | None = None) -> float:
    """Diameter of a circle with the same area as the colony.

    Biological interpretation
    --------------------------
    A size summary that is easier to compare against ruler/caliper
    measurements from manual ImageJ workflows than raw area, and is the
    natural companion figure to report alongside circularity and solidity.

    Parameters
    ----------
    mask : np.ndarray
        Boolean colony mask.
    mm_per_pixel : float or None, default None
        Calibration factor from `calibrate.py`. If None, diameter is
        returned in pixels.

    Returns
    -------
    float
        Equivalent diameter in mm if calibrated, else px.
    """
    pixel_area = float(mask.sum())
    pixel_diameter = np.sqrt(4 * pixel_area / np.pi)
    if mm_per_pixel is not None:
        return pixel_diameter * mm_per_pixel
    return pixel_diameter


def _masked_glcm(
    gray: np.ndarray,
    mask: np.ndarray,
    distance: int = 1,
    levels: int = 8,
) -> np.ndarray:
    """Build a gray-level co-occurrence matrix restricted to pixels inside a mask.

    Colony pixels are quantised into `levels - 1` bins spanning their own
    intensity range (not the whole image's), so texture contrast isn't
    diluted by the large brightness gap between colony and agar. Pixels
    outside the mask are assigned bin 0 as a placeholder; co-occurrence
    counts involving bin 0 are then dropped so background-colony edge
    pixels never contribute a spurious "texture" signal, and the remaining
    colony-colony counts are renormalised to sum to 1.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image, float64 in [0, 1], same shape as `mask`.
    mask : np.ndarray
        Boolean colony mask.
    distance : int, default 1
        Pixel pair distance offset (pixels).
    levels : int, default 8
        Number of gray levels colony pixels are quantised into (bins
        1..levels-1; bin 0 is reserved for background).

    Returns
    -------
    np.ndarray
        Normalised (levels-1) x (levels-1) co-occurrence matrix, averaged
        over the four principal directions (0, 45, 90, 135 degrees) for
        rotational invariance, since a biofilm's wrinkle pattern has no
        preferred orientation on the plate.
    """
    rows, cols = np.where(mask)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    gray_crop = gray[r0:r1, c0:c1]
    mask_crop = mask[r0:r1, c0:c1]

    colony_values = gray_crop[mask_crop]
    lo, hi = colony_values.min(), colony_values.max()
    if hi > lo:
        quantised = np.clip(
            ((gray_crop - lo) / (hi - lo) * (levels - 2)).astype(int), 0, levels - 2
        )
    else:
        quantised = np.zeros_like(gray_crop, dtype=int)
    quantised = quantised + 1  # reserve 0 for background
    quantised[~mask_crop] = 0

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = feature.graycomatrix(
        quantised, distances=[distance], angles=angles, levels=levels, symmetric=True
    )
    glcm = glcm[:, :, 0, :].sum(axis=2)  # average directions -> sum then renormalise
    glcm = glcm[1:, 1:]  # drop background row/column

    total = glcm.sum()
    if total == 0:
        return glcm.astype(float)
    return glcm / total


def texture_contrast(
    gray: np.ndarray, mask: np.ndarray, distance: int = 1, levels: int = 8
) -> float:
    """GLCM contrast of the colony surface -- a local roughness proxy.

    Biological interpretation
    --------------------------
    Measures how sharply pixel intensity varies between neighbouring
    points on the colony surface, computed only from pixels inside the
    colony mask so the agar background never contributes. A smooth,
    featureless colony (matrix mutant) gives low contrast; a wrinkled
    colony with fine light/shadow ridges from surface topology gives high
    contrast. This is complementary to `circularity`, which captures
    macroscopic outline shape rather than surface microtexture.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image, float64 in [0, 1], same shape as `mask`.
    mask : np.ndarray
        Boolean colony mask.
    distance : int, default 1
        Pixel pair distance offset (pixels).
    levels : int, default 8
        Number of gray levels used for quantisation.

    Returns
    -------
    float
        GLCM contrast, >= 0. 0.0 if the mask is empty.
    """
    if not mask.any():
        return 0.0
    glcm = _masked_glcm(gray, mask, distance=distance, levels=levels)
    n = glcm.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return float(np.sum(glcm * (i - j) ** 2))


def texture_entropy(
    gray: np.ndarray, mask: np.ndarray, distance: int = 1, levels: int = 8
) -> float:
    """GLCM entropy of the colony surface -- a texture disorder proxy.

    Biological interpretation
    --------------------------
    Measures how unpredictable neighbouring pixel intensities are on the
    colony surface. A uniform, matte colony gives low entropy (a few
    intensity pairs dominate); an intricately wrinkled colony with
    irregular ridging across many scales gives high entropy. Used
    alongside `texture_contrast`: contrast captures the *magnitude* of
    local intensity change, entropy captures how *disordered* the pattern
    of change is.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image, float64 in [0, 1], same shape as `mask`.
    mask : np.ndarray
        Boolean colony mask.
    distance : int, default 1
        Pixel pair distance offset (pixels).
    levels : int, default 8
        Number of gray levels used for quantisation.

    Returns
    -------
    float
        GLCM entropy in bits, >= 0. 0.0 if the mask is empty.
    """
    if not mask.any():
        return 0.0
    glcm = _masked_glcm(gray, mask, distance=distance, levels=levels)
    # max(0, ...) guards against the -eps-scale negative value that
    # log2(1 + _EPS) produces when a single bin holds all the probability
    # mass (a perfectly uniform colony) -- entropy is 0 there, not negative.
    return float(max(0.0, -np.sum(glcm * np.log2(glcm + _EPS))))


def compute_all_metrics(
    mask: np.ndarray, gray: np.ndarray, mm_per_pixel: float | None = None
) -> dict[str, float]:
    """Compute the full metric set for one segmented colony.

    Parameters
    ----------
    mask : np.ndarray
        Boolean colony mask.
    gray : np.ndarray
        Grayscale image, float64 in [0, 1], same shape as `mask`.
    mm_per_pixel : float or None, default None
        Calibration factor from `calibrate.py`. If None, area/perimeter/
        diameter are reported in pixel units.

    Returns
    -------
    dict[str, float]
        Keys: area, perimeter, circularity, solidity, equivalent_diameter,
        texture_contrast, texture_entropy.
    """
    return {
        "area": area(mask, mm_per_pixel),
        "perimeter": perimeter(mask, mm_per_pixel),
        "circularity": circularity(mask),
        "solidity": solidity(mask),
        "equivalent_diameter": equivalent_diameter(mask, mm_per_pixel),
        "texture_contrast": texture_contrast(gray, mask),
        "texture_entropy": texture_entropy(gray, mask),
    }
