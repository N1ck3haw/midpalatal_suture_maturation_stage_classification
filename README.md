# midpalatal_suture_maturation_stage_classification

# Midpalatal Suture Maturity Classification Repository

## Overview

This GitHub repository is dedicated to the development and validation of a deep learning model for classifying the maturity stages of the midpalatal suture. The model integrates a 3D Convolutional Neural Network (CNN) for processing image data from Cone Beam Computed Tomography (CBCT) images with fully connected networks for handling tabular data, facilitating multi-modal data fusion for accurate classification.

## Dataset

The dataset used in this study includes both tabular data and corresponding CBCT images. The tabular data features include:
- Gender
- Midpalatal Suture Maturation Stage
- Dental Age
- Cervical Spine Stage
- Bone Density Ratio
- Palate Length
- Palate Depth
- Palatal Midline Gray Density (GDs)
- Maxilla Palate Protrusion Gray Density (GDppm)
- Soft Palatal Gray Density (GDsp)

Each row in the table is paired with a corresponding CBCT image, serving as the visual modality in the multi-modal dataset. The dataset can be downloaded from [http://ds.smartstudio.cc/](http://ds.smartstudio.cc/).

## Data Processing

To facilitate the classification process:
- Categorical labels in the tabular data were converted into numerical values using `LabelEncoder` from `sklearn.preprocessing`.
- CBCT images were processed with sharpening to enhance edge details, followed by random augmentations including translations, scaling, and mirroring. Images were resized to a uniform dimension of 128 × 128 × 128 voxels.

## Model Architecture

The model architecture consists of three main modules:
1. **3D CNN Module**: Extracts spatial features from 3D image inputs.
2. **Tabular Data Module**: Processes tabular data through fully connected layers.
3. **Concatenation and Classification Module**: Combines features from both modules for classification.

## Training

The model was trained on an NVIDIA Tesla V100-SXM2-32GB GPU with a learning rate of 0.001. Training was terminated early at epoch 246 when the validation accuracy exceeded 0.95.

## Gradient-weighted Class Activation Mapping (Grad-CAM)

Post-training, Grad-CAM was employed to produce three-dimensional heatmaps, highlighting the spatial features influencing the model's predictions.

## Usage

This repository provides the necessary code and documentation to reproduce the results of the study. Users can download the dataset, preprocess the data, and train the model using the provided scripts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any queries or further information, please contact [Your Name] at [Your Email].
