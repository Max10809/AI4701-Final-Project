# perform feature extraction here
# return the feature vector
import cv2


def extract_features(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


import matplotlib.pyplot as plt
import os

os.makedirs('SIFT', exist_ok=True)

if __name__ == '__main__':
    # Load the image

    images = [f'images/{i:04d}.jpg' for i in range(1, 63)]

    for imp in images:

        image = cv2.imread(imp)
        if image is None:
            print("Error: Image could not be read.")

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a SIFT object
        MAX_SIFT = 50000
        sift = cv2.SIFT_create(nfeatures=MAX_SIFT)

        # Detect SIFT features
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        # Draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None,
                                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Convert image to RGB for displaying in matplotlib
        image_with_keypoints = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)

        # Display the image with keypoints
        plt.figure(figsize=(10, 8))
        plt.imshow(image_with_keypoints)
        plt.title("Image with SIFT keypoints")
        plt.axis('off')
        plt.savefig(os.path.join('SIFT', os.path.basename(imp)))