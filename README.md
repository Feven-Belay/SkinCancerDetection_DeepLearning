# Skin Cancer Detection Using Diversified CNN Architectures

## Project Overview
This project addresses the critical challenge of detecting human skin cancer using advanced deep learning techniques. With skin cancer being one of the most prevalent cancers globally, early detection is crucial for improving survival rates, particularly for melanoma—the deadliest form of skin cancer. This project applies multiple Convolutional Neural Network (CNN) architectures to classify skin lesions, leveraging diverse datasets and state-of-the-art methodologies.

## Objectives
1. Classify skin lesions into two primary categories:
   - **Melanoma (Malignant)**: A serious and potentially life-threatening cancer.
   - **Non-Melanoma (Benign)**: Less aggressive and not life-threatening.

2. Further classification into nine lesion categories:
   - Melanoma
   - Melanocytic Nevus
   - Basal Cell Carcinoma
   - Actinic Keratosis
   - Benign Keratosis
   - Dermatofibroma
   - Vascular Lesion
   - Squamous Cell Carcinoma
   - None of the Others

## CNN Architectures
The project implemented and evaluated three CNN architectures:
1. **XceptionNet**: A lightweight and fast model using depthwise separable convolutions.
2. **EfficientNet-B5**: Balances model size, speed, and accuracy by scaling depth, width, and resolution.
3. **Dual-Path CNN**: Combines raw images and segmentation masks for enhanced feature extraction.

## Methodology
### Dataset
- Primary datasets: ISIC-2019 Challenge, HAM10000.
- Total images: Over 25,000 dermoscopic images across nine diagnostic categories.

### Preprocessing
- Image resizing, normalization, and augmentation techniques like rotation, zoom, flipping, and shifting.
- Data split into training (64%), validation (16%), and testing (20%) subsets using stratified sampling.

### Implementation Details
1. **XceptionNet**:
   - Input size: 224x224 pixels.
   - Training: 30 epochs + 10 fine-tuning epochs.
   - Achieved test accuracy: 68.4%.

2. **EfficientNet-B5**:
   - Input size: 456x456 pixels.
   - Training: 5 epochs with fine-tuning.
   - Achieved test accuracy: 35% (limited by computational resources).

3. **Dual-Path CNN**:
   - Input size: 512x512 pixels for images and segmentation masks.
   - Training: K-Fold Cross-Validation.
   - Achieved test accuracy: 80%.

### Evaluation Metrics
- Precision, Recall, F1-Score
- Confusion Matrix
- Training and Validation Accuracy/Loss Trends

### Libraries Used
- **Deep Learning**: TensorFlow, Keras
- **Preprocessing**: NumPy, Pandas, OpenCV
- **Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: Scikit-learn

## Results
- The Dual-Path CNN model achieved the best performance with an 80% test accuracy, demonstrating the benefit of combining segmentation masks with raw images.
- Computational limitations restricted the use of full datasets, affecting the performance of other models.

## Future Work
To improve model accuracy, the following steps are planned:
1. Use larger datasets with extensive augmentation.
2. Experiment with advanced architectures and preprocessing techniques.
3. Leverage enhanced computational resources for full-scale training.

## References
1. D. S. Charan et al., "Method to classify skin lesions using dermoscopic images," 2022.
2. F. Mahmood et al., "An interpretable deep learning approach for skin cancer categorization," IEEE Transactions on Medical Imaging, 2023.
3. J. Kawahara et al., "Seven-point checklist and skin lesion classification using multitask multimodal neural nets," IEEE Transactions on Biomedical Engineering, 2018.

## Project Links
- **XceptionNet Implementation**: [Colab Link](https://colab.research.google.com/drive/17msc3wBjO5_PYLtURQqSMflS-3-Et017?usp=sharing)
- **EfficientNet-B5 Implementation**: [Colab Link](https://colab.research.google.com/drive/172sUZYIPi0DRWGT7EBjq2GT2calPV5jt?usp=sharing)
- **Dual-Path CNN Implementation**: [Colab Link](https://colab.research.google.com/drive/1FNvGyt2rqI_Lht_QpJ32XdqoPecjkkUI?usp=sharing)

---

This README provides a concise overview of the project, implementation details, and future directions. For full details, refer to the project report.
