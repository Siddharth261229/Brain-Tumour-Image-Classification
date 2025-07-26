# Brain Tumour Image Classification

This repository contains a Convolutional Neural Network (CNN) model for classifying brain MRI images into four categories: Meningioma, Glioma, Pituitary, and No Tumor. The model is built using TensorFlow/Keras and trained on a dataset of brain MRI scans.

## Table of Contents

- [Project Overview](https://www.google.com/search?q=%23project-overview)
- [Dataset](https://www.google.com/search?q=%23dataset)
- [Model Architecture](https://www.google.com/search?q=%23model-architecture)
- [Training Process](https://www.google.com/search?q=%23training-process)
- [Results](https://www.google.com/search?q=%23results)
- [How to Run](https://www.google.com/search?q=%23how-to-run)
- [Dependencies](https://www.google.com/search?q=%23dependencies)
- [Future Work](https://www.google.com/search?q=%23future-work)
- [License](https://www.google.com/search?q=%23license)
- [Contact](https://www.google.com/search?q=%23contact)

## Project Overview

The goal of this project is to develop an image classification model that can accurately identify different types of brain tumors (Meningioma, Glioma, Pituitary) or determine if no tumor is present, based on MRI scans. This can serve as a preliminary diagnostic tool to assist medical professionals.

## Dataset

The model utilizes a dataset of brain MRI images. The notebook specifies the data loaded from Google Drive, structured into `train`, `valid`, and `test` directories.

- **Training Set**: 1695 images belonging to 4 classes.
- **Validation Set**: 502 images belonging to 4 classes.
- **Test Set**: 246 images belonging to 4 classes.

The classes are:

1.  `meningioma`
2.  `glioma`
3.  `pituitary`
4.  `no_tumor`

Images are resized to `128x128` pixels for input to the CNN. A sample of the images from different classes is displayed in the notebook.

## Model Architecture

The CNN model is built using `tf.keras.models.Sequential`. It consists of the following layers:

1.  **Rescaling Layer**: Normalizes pixel values to `[0, 1]` by dividing by 255.
2.  **Convolutional Block 1**:
    - `Conv2D` layer with 32 filters, `(3,3)` kernel, and `relu` activation.
    - `MaxPooling2D` layer with `(2,2)` pool size.
3.  **Convolutional Block 2**:
    - `Conv2D` layer with 64 filters, `(3,3)` kernel, and `relu` activation.
    - `MaxPooling2D` layer with `(2,2)` pool size.
4.  **Convolutional Block 3**:
    - `Conv2D` layer with 128 filters, `(3,3)` kernel, and `relu` activation.
    - `MaxPooling2D` layer with `(2,2)` pool size.
5.  **Flatten Layer**: Flattens the 3D output of the convolutional layers into a 1D vector.
6.  **Dropout Layer 1**: Applies a dropout rate of `0.5` to reduce overfitting.
7.  **Dense Layer 1**: A fully connected layer with 128 units and `relu` activation.
8.  **Dropout Layer 2**: Applies a dropout rate of `0.5`.
9.  **Output Dense Layer**: A fully connected layer with 4 units (one for each class) and `softmax` activation for multi-class classification.

**Model Summary:**

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                   Output Shape          Param #
=================================================================
 conv2d (Conv2D)                (None, 126, 126, 32)  896
 max_pooling2d (MaxPooling2D)   (None, 63, 63, 32)    0
 conv2d_1 (Conv2D)              (None, 61, 61, 64)    18496
 max_pooling2d_1 (MaxPooling2D) (None, 30, 30, 64)    0
 conv2d_2 (Conv2D)              (None, 28, 28, 128)   73856
 max_pooling2d_2 (MaxPooling2D) (None, 14, 14, 128)   0
 flatten (Flatten)              (None, 25088)         0
 dropout (Dropout)              (None, 25088)         0
 dense (Dense)                  (None, 128)           3211392
 dropout_1 (Dropout)            (None, 128)           0
 dense_1 (Dense)                (None, 4)             516
=================================================================
 Total params: 3,305,156 (12.61 MB)
 Trainable params: 3,305,156 (12.61 MB)
 Non-trainable params: 0 (0.00 B)
_________________________________________________________________
```

## Training Process

The model is compiled with the `adam` optimizer and `categorical_crossentropy` loss function (suitable for multi-class classification with one-hot encoded labels). Accuracy is used as the primary metric.

An `EarlyStopping` callback is implemented to stop training when a satisfactory accuracy level (specifically, `> 0.95` training accuracy) is reached, preventing unnecessary computation and potential overfitting to the training data.

The model was trained for a maximum of 100 epochs, but training stopped early due to the callback condition.

## Results

### Training History

- **Epoch 1**: Training Accuracy: \~0.47, Validation Accuracy: \~0.69
- **Epoch 11**: Training Accuracy: \~0.96 (training stopped here due to `myCallback`)
  - Validation Accuracy: \~0.89

### Test Set Evaluation

The model's performance on the unseen test set is as follows:

- **Loss in the test**: `0.356`
- **Test Accuracy**: `0.898` (\~90%)

### Classification Report

```
              precision    recall  f1-score   support

    No Tumor       0.97      0.90      0.94        80
  Meningioma       0.79      0.87      0.83        63
      Glioma       0.85      0.82      0.83        49
   Pituitary       0.98      1.00      0.99        54

    accuracy                           0.90       246
   macro avg       0.90      0.90      0.90       246
weighted avg       0.90      0.90      0.90       246
```

### Confusion Matrix

[Include the confusion matrix plot here. You can drag and drop the image output from your notebook.]

The confusion matrix visually represents the true vs. predicted labels, showing how well the model classified each type of tumor.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/Brain-Tumour-Image-Classification.git
    cd Brain-Tumour-Image-Classification
    ```
2.  **Download the dataset:**
    The notebook uses data mounted from Google Drive. You will need to obtain the brain tumor MRI dataset used in this project. (Please provide a link or instructions on where to get the dataset if it's publicly available, or mention if it's private.)
    Place the `train`, `valid`, and `test` directories in a structure accessible to the notebook (e.g., within `/content/drive/MyDrive/Tumour/` if running on Google Colab, or a local `data/` directory if running locally).
3.  **Open the Jupyter Notebook:**
    If you have Jupyter installed locally:
    ```bash
    jupyter notebook BRAIN_TUMOUR_IMAGE_CLASSFICATION_.ipynb
    ```
    Alternatively, you can open it directly in Google Colab by uploading the `.ipynb` file.
4.  **Run the cells:**
    Execute each cell sequentially in the notebook. Ensure your dataset paths are correctly configured.

## Dependencies

The following libraries are required to run the notebook:

- `tensorflow`
- `keras`
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `opencv-python` (cv2)
- `Pillow` (PIL)

You can install them using pip:

```bash
pip install tensorflow keras numpy pandas seaborn matplotlib scikit-learn opencv-python Pillow
```

(It's highly recommended to create a `requirements.txt` file using `pip freeze > requirements.txt` and include it in your repo for easier dependency management.)

## Future Work

- **Data Augmentation**: Implement more robust data augmentation techniques to improve generalization.
- **Transfer Learning**: Experiment with pre-trained models (e.g., VGG16, ResNet) for potentially higher accuracy and faster training.
- **Hyperparameter Tuning**: Optimize model hyperparameters (learning rate, batch size, number of layers/neurons) using techniques like GridSearchCV or Keras Tuner.
- **Model Interpretability**: Use tools like Grad-CAM to visualize what parts of the image the CNN is focusing on, to build trust and understanding.
- **Deployment**: Deploy the model as a web application or API for practical use.

## License

MIT License

Copyright (c) [2025] [Siddharth]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

## Contact

If you have any questions or suggestions, feel free to reach out\!

- **Email**: [sidshekhawat26@gmail.com]

---
