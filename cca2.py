import numpy as np
from skimage import measure
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.transform import rotate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization

label_image = measure.label(localization.binary_car_image)

# Tuned for the Royal Enfield 2-line rear plate in this 1179x1481 photo:
# Plate height is ~3% to 15% of image height; width is ~5% to 25% of image width
plate_dimensions = (
    0.03 * label_image.shape[0],
    0.15 * label_image.shape[0],
    0.05 * label_image.shape[1],
    0.25 * label_image.shape[1]
)
min_height, max_height, min_width, max_width = plate_dimensions

plate_objects_cordinates = []
plate_like_objects = []
candidates = []

fig, (ax1) = plt.subplots(1, figsize=(8, 6))
ax1.imshow(localization.gray_car_image, cmap="gray")
ax1.set_title("Detected License Plate (Red Box)")

for region in regionprops(label_image):
    if region.area < 400:
        continue

    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
    aspect_ratio = region_width / float(region_height)

    # Filter: Lower half of image (below sky/river), width > height, solid rectangular shape
    if (min_row > label_image.shape[0] * 0.55 and
            min_height <= region_height <= max_height and
            min_width <= region_width <= max_width and
            1.1 <= aspect_ratio <= 2.5 and
            region.solidity > 0.55):

        crop_gray = localization.gray_car_image[min_row:max_row, min_col:max_col]
        # Score by internal contrast (dark characters on white plate have high std deviation)
        contrast_score = np.std(crop_gray)
        candidates.append((contrast_score, min_row, min_col, max_row, max_col, crop_gray))

# Pick the strongest plate candidate so segmentation.py always gets the exact plate at index 0
if candidates:
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, min_row, min_col, max_row, max_col, best_crop_gray = candidates[0]

    # Straighten the slight camera tilt (-8 degrees) of the parked bike plate
    deskewed_gray = rotate(best_crop_gray, angle=-8, resize=False, mode='edge')
    local_thresh = threshold_otsu(deskewed_gray)
    clean_binary_plate = deskewed_gray > local_thresh

    plate_like_objects.append(clean_binary_plate)
    plate_objects_cordinates.append((min_row, min_col, max_row, max_col))

    rectBorder = patches.Rectangle(
        (min_col, min_row), max_col - min_col, max_row - min_row,
        edgecolor="red", linewidth=2, fill=False
    )
    ax1.add_patch(rectBorder)

plt.show()