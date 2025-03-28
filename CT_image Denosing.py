import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma

# Function to Load and Display DICOM Image
def load_dicom_image(filepath):
    ds = pydicom.dcmread(filepath)
    image_array = ds.pixel_array.astype(float)  # Convert to float for processing
    return image_array

# Calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(original, denoised):
    noise = original - denoised
    noise_variance = np.mean(noise**2) + 1e-10  # Prevent division by zero
    signal_variance = np.mean(original**2)
    snr = 10 * np.log10(signal_variance / noise_variance)
    return snr

# Poisson Noise Reduction using Non-Local Means Denoising
def reduce_poisson_noise(image_array):
    sigma_est = np.mean(estimate_sigma(image_array))  # Estimate noise standard deviation
    denoised_poisson = denoise_nl_means(image_array, h=1.15 * sigma_est, fast_mode=True)
    return denoised_poisson

# Periodic Noise Reduction using Fourier Transform Filtering
def reduce_periodic_noise(image_array):
    f = np.fft.fft2(image_array)  # Apply Fast Fourier Transform
    fshift = np.fft.fftshift(f)
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # Radius of the mask to block low frequencies
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0  # Create a circular mask
    fshift *= mask  # Apply the mask to the FFT
    f_ishift = np.fft.ifftshift(fshift)
    denoised_periodic = np.abs(np.fft.ifft2(f_ishift))  # Inverse FFT to get the filtered image
    return denoised_periodic

# Combine Denoising Techniques
def denoise_image(image_array):
    denoised_poisson = reduce_poisson_noise(image_array)  # Apply Poisson noise reduction
    denoised_periodic = reduce_periodic_noise(denoised_poisson)  # Apply periodic noise reduction
    return denoised_periodic

# Main Execution
if __name__ == "__main__":
    # Set the file path for your DICOM file
    filepath = r"D:\DevaNanthu\ProjectData\Vida_Head.MR.Comp_DR-Gain_DR.1005.1.2021.04.27.14.20.13.818.14380335.dcm"  # Adjust to your file path
    
    # Load the original DICOM image
    image_array = load_dicom_image(filepath)
    
    # Denoise the image
    denoised_image = denoise_image(image_array)
    
    # Calculate SNR for original and denoised images
    snr_original = calculate_snr(image_array, np.zeros_like(image_array))  # Original noise (SNR is undefined here)
    snr_denoised = calculate_snr(image_array, denoised_image)
    
    # Display the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title("Original Image\nSNR: Undefined (reference image)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image, cmap='gray')
    plt.title("Denoised Image\nSNR: {:.2f} dB".format(snr_denoised))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print SNR details
    print("SNR of Original Image: Undefined (reference image)")
    print("SNR of Denoised Image: {:.2f} dB".format(snr_denoised))
