# Vehicle-Type-Classification
  Vehicle Type Classification 

A deep-learning project for classifying **four vehicle types** — **Car, Bus, Truck, Motorcycle** — using a convolutional neural network trained on the **Kaggle Vehicle Type Recognition Dataset**.

This repository includes:
- Model training code (Jupyter Notebook)
- Confusion matrix & evaluation metrics
- Accuracy curves
- PPT project presentation
- Clean, reproducible workflow

---

##  Dataset
**Source:** *Vehicle Type Recognition* — Kaggle  
Contains labeled images of:
- Car  
- Bus  
- Truck  
- Motorcycle  

Dataset must be downloaded manually from Kaggle:  
 https://www.kaggle.com/datasets/ashwinsn/vehicle-type-recognition

Place the dataset inside:/data/
/Car/
/Bus/
/Truck/
/Motorcycle/

---

##  Project Workflow
This project follows a complete ML pipeline:

1. **Data Import**
   - Load dataset from local directory
   - Train/validation/test split

2. **Preprocessing**
   - Resize images
   - Normalize pixels
   - Data augmentation:
     - Random flip
     - Rotation
     - Zoom

3. **Model Architecture**
   - CNN / Transfer learning base (e.g., MobileNetV2)
   - GlobalAveragePooling + Dense layers
   - Softmax output for 4 classes

4. **Training**
   - 22 epochs  
   - Optimizer: Adam  
   - Loss: Categorical Cross-Entropy  

5. **Evaluation**
   - Accuracy/Loss curves  
   - Confusion matrix  
   - Test accuracy scoring  

6. **Prediction**
   - Input an image → Output predicted vehicle class

---

##  Results

### ✔ Accuracy
- **Training Accuracy:** ~95–97%  
- **Validation Accuracy:** ~94–97%  
- **Test Accuracy:** **98.75%**

### ✔ Confusion Matrix Summary
| Class | Correct Predictions / 20 |
|-------|---------------------------|
| Bus | 20 |
| Car | 19 |
| Truck | 20 |
| Motorcycle | 20 |

Only **1 image misclassified** (Car → Truck).

---

##  Visual Outputs

### Confusion Matrix  
(1 misclassification across 4 classes)

### Training & Validation Accuracy  
Stable learning curve with no major overfitting.



---

##  Libraries Used
```txt
TensorFlow / Keras
NumPy
Matplotlib
Seaborn
scikit-learn
Python-PIL

