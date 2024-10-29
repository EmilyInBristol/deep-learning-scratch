import numpy as np
import matplotlib.pyplot as plt
from skimage import util, restoration

# Generate a clean image (grayscale gradient)
def generate_clean_image():
    image = np.linspace(0, 1, 256)
    clean_image = np.tile(image, (256, 1))
    return clean_image

# Generate a noisy image
def generate_noisy_image(clean_image):
    noisy_image = util.random_noise(clean_image, mode='gaussian', var=0.01)
    return noisy_image

# Denoising using Total Variation
clean_image = generate_clean_image()
noisy_image = generate_noisy_image(clean_image)
tv_denoised = restoration.denoise_tv_chambolle(noisy_image, weight=0.1)

# Plot the images
plt.figure(figsize=(15, 5))

# Clean image
plt.subplot(1, 3, 1)
plt.imshow(clean_image, cmap='gray')
plt.title('Clean Image')
plt.axis('off')

# Noisy image
plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

# TV denoised image
plt.subplot(1, 3, 3)
plt.imshow(tv_denoised, cmap='gray')
plt.title('TV Denoised Image')
plt.axis('off')

plt.tight_layout()
plt.show()
