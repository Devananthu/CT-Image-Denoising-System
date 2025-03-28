# 🏥 CT Image Denoising System

## 📌 Project Overview
This project focuses on enhancing the quality of CT scan images by reducing **periodic and Poisson noise** without compromising critical medical data. The denoising algorithm improves the clarity of CT images, making it easier for medical professionals to analyze and diagnose conditions accurately.

## 🚀 Features
- 🏥 **Medical Image Processing**: Works with DICOM images to improve clarity.
- 🧠 **Noise Reduction**: Removes periodic and Poisson noise effectively.
- 📊 **Machine Learning Integration**: Uses AI models for advanced noise removal.
- 🔬 **Data Integrity Preservation**: Ensures no loss of medical information.
- 💾 **DICOM Support**: Compatible with standard CT scan formats.

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `numpy`, `scipy` - For numerical computations
  - `opencv`, `skimage` - Image processing
  - `pydicom` - Handling DICOM medical images
  - `tensorflow`, `pytorch` - Deep learning-based denoising
  - `matplotlib` - Visualization of images
- **Machine Learning**:
  - CNN-based denoising autoencoders
  - Wavelet transform filtering

## 🔧 How It Works
1. **Load DICOM Image**: Read the CT scan image using `pydicom`.
2. **Preprocessing**: Convert the image into a format suitable for processing.
3. **Denoising Algorithm**: Apply digital filters or ML-based models.
4. **Reconstruction**: Generate a high-quality, noise-free output.
5. **Visualization**: Display the enhanced image for analysis.

## 📌 Future Enhancements
- Implement real-time CT scan processing.
- Use GANs for high-quality image restoration.
- Integrate with medical diagnostic tools.

