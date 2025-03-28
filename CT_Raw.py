import cv2
import matplotlib.pyplot as plt

# Load the noisy medical image (e.g., MRI, CT scan)
input_image_path = r"D:\DevaNanthu\ProjectData\MRI_noisy.tif"  # Replace with your image path
noisy_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if noisy_image is None:
    print("Error: Could not load the image. Please check the file path.")
else:
    # Apply Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(noisy_image, None, h=30, templateWindowSize=7, searchWindowSize=21)

    # Display the original noisy and denoised images
    plt.figure(figsize=(10, 5))

    # Noisy Image
    plt.subplot(1, 2, 1)
    plt.title("Noisy Image")
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')

    # Denoised Image
    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image, cmap='gray')
    plt.axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()
