import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cca2

# Invert plate so the black letters become white foreground objects
license_plate = np.invert(cca2.plate_like_objects[0])

labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1, figsize=(6, 4))
ax1.imshow(license_plate, cmap="gray")
ax1.set_title("2-Line Character Segmentation (TN38D / J9751)")

# Tuned for 2-line motorcycle plate (Top row: TN38D, Bottom row: J9751)
# Each character is 20% to 52% of plate height, and 5% to 22% of plate width
character_dimensions = (
    0.20 * license_plate.shape[0],
    0.52 * license_plate.shape[0],
    0.05 * license_plate.shape[1],
    0.22 * license_plate.shape[1]
)
min_height, max_height, min_width, max_width = character_dimensions

characters = []
column_list = []

for regions in regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    region_height = y1 - y0
    region_width = x1 - x0

    # Ignore outer plate borders touching edges and ensure character is taller than wide
    if (min_height < region_height < max_height and
            min_width < region_width < max_width and
            region_height >= region_width * 1.1):

        roi = license_plate[y0:y1, x0:x1]

        rect_border = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            edgecolor="lime", linewidth=2, fill=False
        )
        ax1.add_patch(rect_border)

        # Pad the rectangular character into a square before resizing to 20x20 so it doesn't stretch
        pad = max(region_height, region_width)
        square_char = np.zeros((pad, pad), dtype=bool)
        y_off = (pad - region_height) // 2
        x_off = (pad - region_width) // 2
        square_char[y_off:y_off + region_height, x_off:x_off + region_width] = roi

        resized_char = resize(square_char, (20, 20))
        characters.append(resized_char)

        # 2-Line Sorting Key:
        # Top row characters (y0 < 40% of plate height) get +0; Bottom row characters get +1000
        # This guarantees Top Row (TN38D) is read before Bottom Row (J9751)!
        row_offset = 0 if y0 < (license_plate.shape[0] * 0.40) else 1000
        column_list.append(row_offset + x0)

plt.show()