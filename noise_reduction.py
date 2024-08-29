import cv2
import numpy as np

class NoiseReduction:
    def __init__(self, noise_reduction_level=10):
        self.noise_reduction_level = noise_reduction_level

    def reduce_noise(self, image):
        # Apply a fast Non-Local Means Denoising filter
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, self.noise_reduction_level, self.noise_reduction_level, 7, 21)
        return denoised_image

if __name__ == "__main__":
    image = cv2.imread("sample_photo.jpg")
    noise_reducer = NoiseReduction(noise_reduction_level=15)
    denoised_image = noise_reducer.reduce_noise(image)
    cv2.imwrite("denoised_photo.jpg", denoised_image)
