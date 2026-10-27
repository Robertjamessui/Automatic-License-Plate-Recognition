from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Load your Royal Enfield image directly
car_image = imread("RE TN38DJ9751.jpeg", as_gray=True)
print("Image shape:", car_image.shape)

gray_car_image = car_image * 255

# Calculate threshold using the lower half of the photo (where the bike is)
# so the bright sky and mountain reflection don't wash out the white number plate
lower_half = gray_car_image[int(gray_car_image.shape[0] * 0.50):, :]
threshold_value = threshold_otsu(lower_half)
binary_car_image = gray_car_image > (threshold_value * 1.15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(gray_car_image, cmap="gray")
ax1.set_title("Grayscale Image (RE TN38DJ9751)")

ax2.imshow(binary_car_image, cmap="gray")
ax2.set_title("Binary Image")

plt.show()
