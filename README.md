
# **Brain Tumor Detection using MRI Images**
<img src="https://extension.usu.edu/aging/images/a-healthy-brain.jpg" alt="Brain Image" width="1200" height="300"/>


---

## **Problem Statement**
Brain tumors are life-threatening conditions that require timely and accurate diagnosis. Traditional methods of diagnosis through manual examination of MRI scans can be time-consuming and prone to human error. This project aims to automate the process of brain tumor detection using machine learning and deep learning techniques for multiclass classification  to improve diagnostic accuracy and efficiency. 

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

### **1. Data Preprocessing (EDA)**
- Loaded the dataset and divided it into **training** and **testing** sets.
- The training data had balanced data so I did not perform any sampling techniques
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


### Model Download

The model file can be downloaded from [Google Drive](https://drive.google.com/file/d/10LWo1w0Q1Qw4ETkfAPAZy_KoM5oa5q6z/view?usp=sharing).


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

## **6. Installation**
<h4>Clone the repository</h4>
<pre><code>git clone https://github.com/Gujeah/Brain-Tumor-Detection.git</code></pre> <br><br>
<h4>environment setup</h4>
<pre><code> 
  #Used pipenv
  pipenv install tensorflow==1.20.0 python 3.9 numpy 1.24 pillow<br>
  pipenv run
  
</code></pre>
<h4>setting and running a docker image</h4>
<pre><code> 
  #building a docker image
  docker build -t brain-tumour-model . <br><br>
  #running an image
  docker run -d -p 8080:8080 brain-tumour-model 
  
</code></pre>


This project requires further finetuning which is the future work 

---

###**Acknowledgement**
-I would like to thank DataTalks Club in partnership with Saturn Cloud in providing me with the GPUs as well as the knowledge to execute this project successfully.


