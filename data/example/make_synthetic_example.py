"""One-off generator for a synthetic plate photo used to demo the pipeline.

Not part of the package; run manually to (re)create data/example/synthetic_colony.png.
Real photographs should be dropped into this folder alongside/instead of it.
"""

import numpy as np
from skimage import draw, io, filters

rng = np.random.default_rng(0)

size = 800
image = np.full((size, size, 3), 40, dtype=np.uint8)  # dark bench background

center = (size // 2, size // 2)
dish_radius = 340

# Petri dish: agar fill + a slightly darker rim ring.
rr, cc = draw.disk(center, dish_radius, shape=image.shape[:2])
image[rr, cc] = 190  # agar
rr_rim, cc_rim = draw.circle_perimeter(center[0], center[1], dish_radius, shape=image.shape[:2])
for dr in range(-3, 4):
    rr_r = np.clip(rr_rim + dr, 0, size - 1)
    image[rr_r, cc_rim] = 140

# Colony: an irregular blob darker than the agar, built from a base circle
# perturbed by low-frequency angular noise so the margin looks biologically
# plausible (lobed, not a perfect circle) - roughly analogous to a
# wild-type wrinkled colony footprint.
theta = np.linspace(0, 2 * np.pi, 360)
base_radius = 150
wobble = 25 * np.sin(5 * theta + 0.7) + 12 * rng.standard_normal(360).cumsum() / 360
radius = base_radius + wobble
colony_rr = (center[0] + radius * np.sin(theta)).astype(int)
colony_cc = (center[1] + radius * np.cos(theta)).astype(int)
colony_mask = draw.polygon2mask(image.shape[:2], np.stack([colony_rr, colony_cc], axis=1))

colony_gray = 90 + 15 * rng.standard_normal(image.shape[:2])
colony_gray = filters.gaussian(colony_gray, sigma=3)
colony_gray = np.clip(colony_gray, 60, 120).astype(np.uint8)
for ch in range(3):
    image[..., ch] = np.where(colony_mask, colony_gray, image[..., ch])

# Mild illumination gradient (simulated lightbox hot spot) to exercise
# the background-flattening step.
yy, xx = np.mgrid[0:size, 0:size]
hotspot = 30 * np.exp(-(((xx - 200) ** 2 + (yy - 200) ** 2)) / (2 * 300**2))
image = np.clip(image.astype(float) + hotspot[..., None], 0, 255).astype(np.uint8)

io.imsave("data/example/synthetic_colony.png", image)
print("Wrote data/example/synthetic_colony.png")
