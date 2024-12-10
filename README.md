# Bird Species Classification using convolutional neural network

A deep learning project utilizing ResNet-50 to classify images of 100 different bird species. This model uses transfer learning with data augmentation, learning rate scheduling, early stopping, and cross-entropy loss to achieve accurate classification.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Classes](#Classes)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Prediction](#prediction)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgement](#Acknowledgement)

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

## Classes

The dataset contains the following bird species along with an "unknown" class:

1. Black-footed Albatross  
2. Laysan Albatross  
3. Sooty Albatross  
4. Groove-billed Ani  
5. Crested Auklet  
6. Least Auklet  
7. Parakeet Auklet  
8. Rhinoceros Auklet  
9. Brewer Blackbird  
10. Red-winged Blackbird  
11. Rusty Blackbird  
12. Yellow-headed Blackbird  
13. Bobolink  
14. Indigo Bunting  
15. Lazuli Bunting  
16. Painted Bunting  
17. Cardinal  
18. Spotted Catbird  
19. Gray Catbird  
20. Yellow-breasted Chat  
21. Eastern Towhee  
22. Chuck-will's-widow  
23. Brandt's Cormorant  
24. Red-faced Cormorant  
25. Pelagic Cormorant  
26. Bronzed Cowbird  
27. Shiny Cowbird  
28. Brown Creeper  
29. American Crow  
30. Fish Crow  
31. Black-billed Cuckoo  
32. Mangrove Cuckoo  
33. Yellow-billed Cuckoo  
34. Gray-crowned Rosy Finch  
35. Purple Finch  
36. Northern Flicker  
37. Acadian Flycatcher  
38. Great Crested Flycatcher  
39. Least Flycatcher  
40. Olive-sided Flycatcher  
41. Scissor-tailed Flycatcher  
42. Vermilion Flycatcher  
43. Yellow-bellied Flycatcher  
44. Frigatebird  
45. Northern Fulmar  
46. Gadwall  
47. American Goldfinch  
48. European Goldfinch  
49. Boat-tailed Grackle  
50. Eared Grebe  
51. Horned Grebe  
52. Pied-billed Grebe  
53. Western Grebe  
54. Blue Grosbeak  
55. Evening Grosbeak  
56. Pine Grosbeak  
57. Rose-breasted Grosbeak  
58. Pigeon Guillemot  
59. California Gull  
60. Glaucous-winged Gull  
61. Heermann's Gull  
62. Herring Gull  
63. Ivory Gull  
64. Ring-billed Gull  
65. Slaty-backed Gull  
66. Western Gull  
67. Anna's Hummingbird  
68. Ruby-throated Hummingbird  
69. Rufous Hummingbird  
70. Green Violetear  
71. Long-tailed Jaeger  
72. Pomarine Jaeger  
73. Blue Jay  
74. Florida Jay  
75. Green Jay  
76. Dark-eyed Junco  
77. Tropical Kingbird  
78. Gray Kingbird  
79. Belted Kingfisher  
80. Green Kingfisher  
81. Pied Kingfisher  
82. Ringed Kingfisher  
83. White-breasted Kingfisher  
84. Red-legged Kittiwake  
85. Horned Lark  
86. Pacific Loon  
87. Mallard  
88. Western Meadowlark  
89. Hooded Merganser  
90. Red-breasted Merganser  
91. Mockingbird  
92. Nighthawk  
93. Clark's Nutcracker  
94. White-breasted Nuthatch  
95. Baltimore Oriole  
96. Hooded Oriole  
97. Orchard Oriole  
98. Scott's Oriole  
99. Ovenbird  
100. Brown Pelican  
101. Unknown


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

## Acknowledgement
this project is done under guidance of [Dr Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu)
