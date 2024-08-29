import cv2
import numpy as np
from tensorflow.keras.models import load_model

class MotionDeblurring:
    def __init__(self, model_path="deblur_model.h5"):
        self.model = load_model(model_path)

    def deblur_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        deblurred = self.model.predict(image)
        deblurred = np.squeeze(deblurred, axis=0)
        deblurred = (deblurred * 255).astype(np.uint8)
        return cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)

if __name__ == "__main__":
    image = cv2.imread("sample_blurry_photo.jpg")
    deblurring = MotionDeblurring()
    deblurred_image = deblurring.deblur_image(image)
    cv2.imwrite("deblurred_photo.jpg", deblurred_image)
