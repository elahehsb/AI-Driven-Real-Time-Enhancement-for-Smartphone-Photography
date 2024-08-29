import cv2

class DetailEnhancement:
    def __init__(self, enhancement_factor=1.5):
        self.enhancement_factor = enhancement_factor

    def enhance_details(self, image):
        # Apply Unsharp Masking
        gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
        enhanced_image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return enhanced_image

if __name__ == "__main__":
    image = cv2.imread("sample_photo.jpg")
    enhancer = DetailEnhancement(enhancement_factor=1.5)
    enhanced_image = enhancer.enhance_details(image)
    cv2.imwrite("enhanced_photo.jpg", enhanced_image)
