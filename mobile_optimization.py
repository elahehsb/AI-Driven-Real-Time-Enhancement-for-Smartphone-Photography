import cv2

class MobileOptimization:
    def __init__(self):
        pass

    def optimize_image(self, image):
        # Downscale image for faster processing
        small_image = cv2.resize(image, (320, 240))
        return small_image

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    optimizer = MobileOptimization()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        optimized_frame = optimizer.optimize_image(frame)
        cv2.imshow("Mobile Optimized Frame", optimized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
