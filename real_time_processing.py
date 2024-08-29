import cv2
from noise_reduction import NoiseReduction
from motion_deblurring import MotionDeblurring
from detail_enhancement import DetailEnhancement

class RealTimeProcessing:
    def __init__(self):
        self.noise_reducer = NoiseReduction(noise_reduction_level=15)
        self.motion_deblurrer = MotionDeblurring(model_path="deblur_model.h5")
        self.detail_enhancer = DetailEnhancement(enhancement_factor=1.5)

    def process_image(self, image):
        image = self.noise_reducer.reduce_noise(image)
        image = self.motion_deblurrer.deblur_image(image)
        image = self.detail_enhancer.enhance_details(image)
        return image

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    processor = RealTimeProcessing()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = processor.process_image(frame)
        cv2.imshow("Real-Time Photo Enhancement", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
