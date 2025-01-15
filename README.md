
# **Brain Tumor Detection using MRI Images**
Dear reviewer Incase you will only find a notebook nly this morning, please have patience with me. I am finalizing the project I will push anytime soon. I had issues which prevented me from finishing on time. Thank you
## **Problem Statement**
Brain tumors are life-threatening conditions that require timely and accurate diagnosis. Traditional methods of diagnosis through manual examination of MRI scans can be time-consuming and prone to human error. This project aims to automate the process of brain tumor detection using machine learning and deep learning techniques to improve diagnostic accuracy and efficiency.

---

## **Dataset**
The dataset used for this project was sourced from [Kaggle: Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).  
The dataset contains MRI images categorized into four classes:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**



---

## **Project Workflow**

### **1. Data Preprocessing**
- Loaded the dataset and divided it into **training** and **testing** sets.
- Resized all images to a uniform size of `150x150 pixels`.
- Applied data augmentation to the training data:
  - **Zoom Range:** 0.1
  - **Vertical Flip:** Enabled
  - **Brightness Adjustment:** 0.1
  - **Random Rotations:** 0.2
  - **Width and Height Shifts:** 10%
  - **Shear Transformations:** 10%

*Note:* After applying these augmentations it only made the performance worse I do not know  the reason but I am keeping it for future improvement, so I only cmmented in the notebook

---

### **2. Model Selection**
The project used **Xception**, a pre-trained convolutional neural network (CNN) model known for its strong performance in image classification tasks.

#### **Why Xception?**
- Built on depthwise separable convolutions, offering computational efficiency.
- Pre-trained on the ImageNet dataset, which allows transfer learning for high accuracy on specific tasks.
- Proven track record in solving complex image classification problems.

#### **Transfer Learning**
The Xception model was fine-tuned for **multi-class classification** (four classes) by:
- Replacing the top layer with a fully connected dense layer.

---

### **3. Model Training**
- **Loss Function:** Categorical Crossentropy (for multi-class classification).
- **Optimizer:** Adam.
- **Metrics:** Accuracy.


---

### **4. Model Evaluation**
- The model was evaluated on the validation and test datasets.

- Achieved high classification accuracy, indicating effective learning.

---

### **5. Results**
- **Training Accuracy:** ~91%
- **Validation Accuracy:** ~86%
- **Test Accuracy:** ~89%
- The model successfully classified MRI images into the correct categories, demonstrating its potential to assist in real-world diagnostic applications.
---
This project requires further finetuning which is the future work 
---
###**Acknowledgement**
-I would like to thank DataTalks Club in partnership with Saturn Cloud in providing me with the GPUs as well as the knowledge to execute this project successfully.


