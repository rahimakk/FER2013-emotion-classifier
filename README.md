# Facial Expression Recognition on FER-2013 using CNN

This project implements a Convolutional Neural Network (CNN) to classify facial expressions from grayscale images using the FER-2013 dataset. The goal is to predict one of seven emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

---

##  Dataset Overview

The FER-2013 dataset contains 48x48 grayscale images divided into:
- **Training set**: Used to train the model
- **Test set**: Used to evaluate final performance

Images are labeled with one of 7 emotion classes:
`['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']`

---

##  Libraries Used

```python
numpy, pandas
matplotlib, seaborn
tensorflow / keras
cv2 (for image reading)
os (file path operations)


##  Data Preprocessing

###  Image Settings
- All images were resized to **48x48** pixels.
- Color mode was set to **Grayscale**.
- Pixel values were **normalized to the range [0, 1]**.

### Data Augmentation
Used `ImageDataGenerator` from Keras to perform real-time data augmentation on the training images.

**Augmentation Techniques Applied:**
- `rotation_range=20` → Randomly rotate images by up to 20 degrees.
- `zoom_range=0.2` → Random zoom within 20%.
- `horizontal_flip=True` → Randomly flip images horizontally.

**Validation Split:**
- 20% of the training data was reserved for validation using `validation_split=0.2`.

This augmentation helped improve model generalization and reduce overfitting by increasing the diversity of the training data.


## Data Loaders

Three data generators were created using `ImageDataGenerator`:

- **`train_generator`**:  
  - Includes augmentation  
  - Used for training the model  

- **`val_generator`**:  
  - No augmentation  
  - Used for validating the model during training  

- **`test_generator`**:  
  - No augmentation  
  - Used for final model evaluation  

---

##  Data Visualization

To verify data integrity and class correctness:

- Displayed a sample of **10 images** from the training set using `matplotlib`
- Each image was shown with its corresponding **class label**
- Purpose:
  - Confirm images load correctly
  - Ensure labels match the image content

---

## Callbacks for Efficient Training

Used two key Keras callbacks to enhance training performance:

### ReduceLROnPlateau
- Automatically reduces the learning rate when the validation loss stops improving
- Helps fine-tune the model to reach a better local minimum  
```python
ReduceLROnPlateau(factor=0.2, patience=4)


### EarlyStopping

Implemented **EarlyStopping** to prevent overfitting and reduce unnecessary training time.

- **Function:** Stops training early if the **validation loss** doesn't improve for a specified number of epochs.
- **Benefit:** Restores the **best model weights** achieved during training.

```python
EarlyStopping(patience=8, restore_best_weights=True)

---
##  Training Visualization

To monitor the model's performance during training, plotted the following metrics:

- **Training Accuracy vs. Validation Accuracy**
- **Training Loss vs. Validation Loss**

These plots help:

- Visualize the model’s learning progress over epochs
- Detect signs of **overfitting** or **underfitting**

Used `matplotlib` to generate the plots after training was complete.


---
## Epoch-by-Epoch Training Summary

| Phase            | Epoch Range | Observations                                                                 | Learning Rate          |
|------------------|-------------|------------------------------------------------------------------------------|------------------------|
|  Initial Learning | 1–5         | Rapid accuracy increase (22% → 52%), val accuracy still unstable             | 0.001                  |
|  Steady Improvement | 6–10        | Both training and validation accuracy improved (~58%), lower val loss       | 0.001                  |
|  Plateau Begins   | 11–14       | Accuracy gains slowed, val accuracy plateaued around ~58–60%                | 0.001                  |
|  Gradual Gains     | 15–24       | Model improved to ~65%, val accuracy became more consistent                 | 0.001                  |
|  LR Reduced       | 25          | `ReduceLROnPlateau` triggered due to stagnant val loss                      | ↓ 0.0002               |
|  Fine-Tuning       | 26–33       | Accuracy rose to ~71%, val accuracy peaked near 65%, then plateaued again  | 0.0002                 |
| LR Reduced Again | 33          | Second LR reduction triggered by no val improvement                         | ↓ 0.00004              |
|  Final Tuning     | 34–39       | Minor improvements, validation stabilized, third LR drop at epoch 39       | ↓ 0.000008             |
|  Final Phase       | 40–50       | Accuracy peaked ~72%, best val accuracy (~66%) at **epoch 47**, training ends | 0.0000016              |

>  **Model weights restored from epoch 47**, where validation performance was highest.

---
##  Final Evaluation

After training completion and restoring the best model weights (from epoch 47), the model was evaluated on the unseen **test dataset**.

- ** Test Accuracy:** `0.6704`  
- ** Test Loss:** `0.9450`

These results indicate that the model generalizes well on unseen data, maintaining consistent performance with validation accuracy. While there's still room for improvement, the model demonstrates effective learning of facial expression features from the FER-2013 dataset.

---
## 🖼️ Prediction Visualization

A custom visualization block was created to evaluate model predictions on the test set:

- Displayed **50 random test images**.
- Showed **predicted vs. true class labels**.
- Highlighted **misclassified images** for quick error analysis.
- Included the **confidence score** for each prediction.
- Displayed the **exact file path** of each image to trace samples easily.

This visual inspection helped assess how well the model generalized to unseen data and identify patterns in misclassifications.

---

##  Conclusion

The project successfully demonstrates the end-to-end pipeline of training a CNN on the **FER-2013** facial expression dataset using deep learning best practices.

###  Achievements:
- Effective **data augmentation** using `ImageDataGenerator`
- Employed **learning rate scheduling** (`ReduceLROnPlateau`)
- Implemented **EarlyStopping** to avoid overfitting
- Thorough **evaluation** and **visual debugging** of predictions

###  Next Steps:
- Experiment with deeper models like **ResNet** or **VGG16** with fine-tuning
- Integrate **attention mechanisms** or use **ensemble techniques**
- Perform **hyperparameter optimization** (e.g., batch size, learning rate, dropout)


