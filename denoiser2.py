import numpy as np
import matplotlib.pyplot as plt
from skimage import util

# Generate a simple clean image (grayscale gradient)
def generate_clean_image():
    image = np.linspace(0, 1, 256)
    image = np.tile(image, (256, 1))
    return image

# Generate noisy images
def add_gaussian_noise(image, var=0.01):
    return util.random_noise(image, mode='gaussian', var=var)

def add_poisson_noise(image):
    return util.random_noise(image, mode='poisson')

# Create the clean image
clean_image = generate_clean_image()

# Add Gaussian noise
gaussian_noisy_image = add_gaussian_noise(clean_image, var=0.01)

# Add Poisson noise
poisson_noisy_image = add_poisson_noise(clean_image)

# Plot the clean and noisy images
plt.figure(figsize=(18, 6))

# Clean Image
plt.subplot(1, 3, 1)
plt.title("Clean Image")
plt.imshow(clean_image, cmap='gray')
plt.axis('off')

# Gaussian Noisy Image
plt.subplot(1, 3, 2)
plt.title("Gaussian Noisy Image")
plt.imshow(gaussian_noisy_image, cmap='gray')
plt.axis('off')

# Poisson Noisy Image
plt.subplot(1, 3, 3)
plt.title("Poisson Noisy Image")
plt.imshow(poisson_noisy_image, cmap='gray')
plt.axis('off')

plt.show()
