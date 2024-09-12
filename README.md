# Face Recognition System Using Convolutional Neural Network (CNN)
<p>
  <img src="https://github.com/user-attachments/assets/215c118e-e4f6-4440-90bd-9b472b963869">
</p>
This project aims to develop a robust face recognition system using Convolutional Neural Networks (CNN). The model leverages deep learning techniques to accurately identify and verify individuals based on their facial features. The CNN architecture is designed to extract and learn intricate patterns from facial images, making it highly effective for face recognition tasks. <br>
<br>
Dataset: Labeled Face in the Wild (LFW) by University of Massachusetts. <br>
Link dataset: https://vis-www.cs.umass.edu/lfw/

### Key Features:
**1. Data Preprocessing**: The dataset consists of labeled facial images, which are preprocessed to enhance quality and consistency. Techniques such as normalization, resizing, and augmentation are applied to improve model performance. <br>
**2. CNN Architecture**: The model employs a multi-layer CNN to capture spatial hierarchies in the images. Layers include convolutional, pooling, and fully connected layers, optimized for feature extraction and classification. <br>
**3. Training and Validation**: The model is trained on a diverse dataset of facial images, with a portion reserved for validation to ensure accuracy and generalization. Techniques like dropout and batch normalization are used to prevent overfitting. <br>
**4. Performance Evaluation**: The system’s performance is evaluated using metrics such as accuracy and Confusion Matrix. These metrics help in fine-tuning the model for optimal results. <br>
**5. Applications**: This face recognition system can be applied in various domains, including security, authentication, and personalized user experiences. <br>

## Convolutional Neural Network (CNN) Model Architecture
<p>
  <img src="https://github.com/user-attachments/assets/57808859-0c2f-4067-85cf-0fddb34a125e">
</p>

## Dataset Samples
<p align="center">
  <img src="https://github.com/user-attachments/assets/4677dd4f-fb7d-4b34-b9b9-ec095f8cde36" width="720">
</p>
<p align="center">
<img src="https://github.com/user-attachments/assets/c67ed7e6-275a-4717-a4b4-9545f6caf2de">
</p>

### Pre-Processing Dataset using Augmentation Data
Data augmentation is a technique used in machine learning to artificially increase the size and diversity of a training dataset by creating modified versions of existing data. This helps improve the performance and robustness of machine learning models, especially when the original dataset is small or imbalanced. <br>
<p align="center">
<img src="https://github.com/user-attachments/assets/8888a88b-eb4b-4c58-a375-d178ae35bfe1" width="580">
</p>

### Dataset Splits
<p align="center">
<img src="https://github.com/user-attachments/assets/8b66b7d9-eff2-49db-b525-96b99e28e985" width="670">
</p>

## Cross-Validation Method
<p align="center">
<img src="https://github.com/user-attachments/assets/3586fec8-4e4e-4d16-9e3e-b84829aaac91">
</p>
Cross-validation is a technique used in machine learning to evaluate the performance of a model and ensure it generalizes well to unseen data. Cross-validation involves splitting the dataset into multiple subsets, training the model on some subsets, and validating it on the remaining subsets. This process is repeated several times, and the results are averaged to provide a more accurate estimate of the model’s performance. <br>

### Why Use Cross-Validation?
**Avoid Overfitting**: By testing the model on different subsets of data, cross-validation helps detect overfitting, where the model performs well on training data but poorly on new data. <br>
**Better Performance Estimation**: It provides a more reliable estimate of the model’s performance compared to a single train-test split.<br>

For more about Cross-Validation: https://scikit-learn.org/stable/modules/cross_validation.html. <br>

### 5-Fold Cross-Validation Result
<div align="center">
  
![fold 1](https://github.com/user-attachments/assets/167d3346-145a-4533-a892-348f8be298f2) <br>
![fold 2](https://github.com/user-attachments/assets/1c7540f1-112b-40c8-b1dc-ba971cba62d6) <br>
![fold 3](https://github.com/user-attachments/assets/a4114d22-5dda-4807-ab18-78b778b7f898) <br>
![fold 4](https://github.com/user-attachments/assets/c925dcb7-3aac-47a0-9452-f20c54a66b07) <br>
![fold 5](https://github.com/user-attachments/assets/223a57ac-512e-46fd-90c4-1befa7c176cc) <br>

</div>

### Best Fold
To select the best model, the highest value of the average accuracy of the five models produced was taken, so that Fold 3 was selected as the best model. <br>
<div align="center">
  
![fold 3](https://github.com/user-attachments/assets/a4114d22-5dda-4807-ab18-78b778b7f898) <br>

</div>

## Result
<div align="center">
  
| Authorized Access | Unauthorized Access |
|----------|----------|
| ![Detect Face](https://github.com/user-attachments/assets/f4fd2735-acf9-422f-9e00-f05bb262e450) | ![Detect Face2](https://github.com/user-attachments/assets/270226cb-7f39-4d8e-8c4e-d00e1a6431fc) |

</div>
