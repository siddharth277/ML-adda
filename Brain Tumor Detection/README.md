# 🧠 Brain Tumor Detection using Deep Learning

## 📝 Abstract

Brain tumors are a serious medical condition affecting both children and adults, constituting 85 to 90 percent of all primary Central Nervous System (CNS) tumors. Annually, around 11,700 people receive a brain tumor diagnosis, with a 5-year survival rate of approximately 34 percent for men and 36 percent for women. Proper treatment, planning, and accurate diagnostics are crucial to improving patient life expectancy.

This project focuses on automated classification techniques using Deep Learning Algorithms such as Convolutional Neural Network (CNN), Transfer Learning (TL), and Artificial Neural Network (ANN). These techniques offer higher accuracy than manual classification, aiding doctors worldwide in efficient detection and classification of brain tumors.

## 🌐 Context

Brain tumors present complexities in size and location, requiring expertise for accurate analysis. Developing countries often face challenges due to a shortage of skilled doctors and insufficient knowledge about tumors. An automated system on the cloud can address these issues, providing a faster and more accessible solution.

## Methodology

### Project Overview
The Brain Tumor Detection project aims to develop a deep learning model to classify brain tumor images into different categories. Three different models, a Convolutional Neural Network (CNN), a Multilayer Perceptron (MLP) based on TensorFlow, and a VGG16 transfer learning model, are explored for this task.

### Project Directory Structure
```
Brain Tumor Detection
  |- brain_tumor.ipynb
  |- README.md
```

### Methodology
1. **Importing Libraries:**  
   - Libraries such as NumPy, Pandas, TensorFlow, and others are imported for data manipulation, visualization, and model building.

2. **Loading the Dataset:**
   - The training and testing datasets are loaded into dataframes. File paths and labels are extracted for each image in the dataset.

3. **Data Preprocessing:**
   - Data balance is checked to ensure an even distribution of classes.
   - The testing dataset is split into validation and test sets.
   - ImageDataGenerator is used to convert dataframes to numpy arrays for model training.

4. **Model Structure:**
   - Three models are explored: 
     - CNN: A CNN model is created using Keras Sequential API with convolutional and pooling layers followed by dense layers for classification.
     - MLP: An MLP model is created with Flatten, Dense, and Dropout layers.
     - VGG16: A VGG16 transfer learning model is used with a custom dense layer for classification.

5. **Training the Models:**
   - Each model is compiled using the Adamax optimizer and categorical cross-entropy loss.
   - Models are trained on the training dataset for a specified number of epochs, with validation data for evaluation.

6. **Model Performance:**
   - Training and validation loss and accuracy are plotted over epochs to visualize the model's performance.
   - The best epoch based on validation loss and accuracy is noted for each model.

### Model Performance
#### Convolutional Neural Network (CNN)

- **Classification Report**  
   ```
           precision    recall  f1-score   support

  glioma       0.91      0.64      0.75       212
  meningioma       0.85      0.43      0.57       204
     notumor       0.70      1.00      0.83       186
   pituitary       0.69      0.99      0.82       198

    accuracy                           0.76       800
   macro avg       0.79      0.76      0.74       800
   weighted avg       0.79      0.76      0.74       800
   ```

#### Multilayer Perceptron (MLP) Based on TensorFlow

- **Classification Report**  
  ``` 
               precision    recall  f1-score   support

  glioma       0.62      0.58      0.60       212
  meningioma       0.45      0.45      0.45       204
     notumor       0.65      0.87      0.75       186
   pituitary       0.77      0.58      0.66       198

    accuracy                           0.61       800
   macro avg       0.62      0.62      0.61       800
  weighted avg       0.62      0.61      0.61       800
   ```

#### VGG16 Transfer Learning Model

  ```
                 precision    recall  f1-score   support

  glioma       0.98      0.60      0.75       212
  meningioma       0.78      0.79      0.79       204
  notumor       0.85      1.00      0.92       186
  pituitary       0.81      0.99      0.89       198

    accuracy                           0.84       800
   macro avg       0.85      0.85      0.84       800
  weighted avg       0.86      0.84      0.83       800
   ```

