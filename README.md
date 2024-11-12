# Bird Species Classification with ResNet-50

A deep learning project utilizing ResNet-50 to classify images of 100 different bird species. This model uses transfer learning with data augmentation, learning rate scheduling, early stopping, and cross-entropy loss to achieve accurate classification.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Prediction](#prediction)
- [Usage](#usage)
- [Results](#results)

## Overview
This project implements a bird species classifier using a modified ResNet-50 architecture trained on a custom dataset with 100 bird species. Data augmentation is applied to improve model generalization, and training metrics are logged to track model performance.

## Requirements
To set up and run this project, install the following packages:

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib
- tqdm
- psutil
- PIL (Pillow)

```bash
pip install torch torchvision scikit-learn pandas numpy seaborn matplotlib tqdm psutil pillow
```

## Dataset
The dataset consists of three folders: `train`, `val`, and `test`, located in the `data_dir` directory. Each folder contains images categorized by bird species. This project uses a `[70-20-10]` split for training, validation, and testing.

## Model Architecture
This project uses a pre-trained ResNet-50 architecture modified to classify 100 bird species:

- **Transfer Learning**: Pre-trained weights on ImageNet are used to initialize the model.
- **Final Layer Adjustment**: The final fully connected layer is modified to match the number of bird classes.
- **Optimization**: The model is trained using the Adam optimizer with weight decay to prevent overfitting.

## Training Pipeline
- **Data Augmentation**: Random transformations are applied to the training dataset.
- **Training Loop**: The model is trained with `CrossEntropyLoss` and the Adam optimizer, using `ReduceLROnPlateau` to adjust the learning rate based on validation loss.
- **Early Stopping**: Training stops early if validation loss does not improve for a set patience period.

Training metrics are saved to `training_log_fix_final.csv` for analysis, and the best model is saved as `best_model_fix_final.pth`.

## Prediction
The prediction pipeline allows inference on new bird images using the trained ResNet-50 model:

- The model expects images to be resized and normalized to match the training distribution.
- The `predict_image()` function loads the image, applies transformations, and returns the predicted bird species.

## Usage
- **Training**: Run `training.ipynb` to train the model. Adjust `data_dir`, `num_epochs`, and `batch_size` as needed.
- **Prediction**: Use `prediction.py` to classify images. Update `image_path` and `model_path` accordingly.

## Results
The final model achieves notable accuracy on the validation set. Check the saved logs for per-epoch metrics and performance analysis.

