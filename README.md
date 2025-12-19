# 🧠 Rock–Paper–Scissors Image Classification using CNN (PyTorch)

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify hand gesture images into **Rock, Paper, and Scissors**.
The model is trained on a **standard Rock–Paper–Scissors dataset** and evaluated on **real-world smartphone images** captured by the author to assess generalization beyond controlled datasets.

The entire pipeline is **fully automated**, reproducible, and designed to run end-to-end in **Google Colab** without any manual file uploads.

---

## 🎯 Project Objectives

* Build a complete **CNN image classification workflow** in PyTorch
* Train on a **standard dataset**
* Perform **essential image preprocessing**
* Test the trained model on **custom phone images**
* Visualize performance using professional evaluation tools
* Analyze **real-world generalization limitations**

---

## 🗂️ Dataset Description

### 🔹 Standard Dataset

* **Name:** Rock–Paper–Scissors Dataset
* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
* **Classes:** rock, paper, scissors
* **Image Type:** RGB
* **Loading Method:** `torchvision.datasets.ImageFolder`

> **Note:** This dataset is not natively available in `torchvision.datasets`.
> Therefore, `ImageFolder` is used, which is an officially supported torchvision dataset loader.

---

### 🔹 Custom Dataset (Phone Images)

* **Source:** Smartphone camera
* **Number of Images:** 10
* **Classes:** Rock, Paper, Scissors
* **Conditions:**

  * Plain background (table surface)
  * Single hand per image
  * Natural lighting
* **Purpose:** Real-world model evaluation

---

## 📁 Repository Structure

```text
cnn-rps-pytorch/
│
├── dataset/
│   ├── rps/
│   │   ├── rock/
│   │   ├── paper/
│   │   └── scissors/
│   │
│   └── phone/
│       ├── rock_1.jpg
│       ├── paper_1.jpg
│       ├── scissors_1.jpg
│       └── ...
│
├── model/
│   └── rps_cnn.pth
│
├── 190110.ipynb
└── README.md
```

---

## 🔄 Data Preprocessing

All images (standard dataset and phone images) are processed to ensure **consistent tensor formatting**.

### 🔹 Training Transform (with Data Augmentation)

* Resize to **224 × 224**
* Random horizontal flip
* Random rotation
* Color jitter (brightness & contrast)
* Convert to tensor
* Normalize using ImageNet mean and standard deviation

### 🔹 Validation & Phone Transform

* Resize to **224 × 224**
* Convert to tensor
* Normalize using the **same mean/std**

```text
Mean: [0.485, 0.456, 0.406]
Std:  [0.229, 0.224, 0.225]
```

> Data augmentation is applied **only during training**, following best practices.

---

## 🏗️ CNN Architecture

The model consists of:

* **3 Convolutional Blocks**

  * Convolution → ReLU → MaxPooling
* **Fully Connected Classifier**

  * Dense layer with ReLU
  * Dropout for regularization
  * Output layer with 3 neurons

### 🔧 Training Configuration

* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam
* **Batch Size:** 64
* **Epochs:** 10
* **Device:** GPU (if available)

---

## 📊 Evaluation & Visualizations

### 📈 Training Curves

* Loss vs Epochs
* Accuracy vs Epochs (Training & Validation)

### 🔍 Confusion Matrix

A heatmap visualizing classification performance on the validation set.

### 📸 Real-World Prediction Gallery

Predictions on custom smartphone images with confidence scores.

Example:

```text
Predicted: Paper (98.6%)
```

---

## ⚠️ Real-World Generalization Analysis

The model achieves **high accuracy on the standard dataset** but shows **reduced performance on custom phone images**.

### 🔍 Reason

This performance gap is primarily caused by:

* **Domain shift** between controlled dataset images and real-world photos
* **Background bias**, as all phone images share a similar surface
* Differences in lighting, camera angle, and hand appearance

Although data augmentation improves robustness, full generalization requires **greater real-world data diversity or domain adaptation**.

This behavior highlights a **known and important limitation of CNNs** when deployed outside their training distribution.

---

## 🚀 How to Run (Fully Automated)

1. Open the Colab notebook: `190110.ipynb`
2. Click **Runtime → Run All**

The notebook will automatically:

1. Clone this GitHub repository
2. Load the dataset
3. Train the CNN (or load saved weights)
4. Generate evaluation plots
5. Predict custom phone images

🚫 **No manual file uploads are required**

---

## 📌 Key Features

* Fully automated CNN pipeline
* Professional preprocessing workflow
* Real-world testing with phone images
* Clear visualization and error analysis
* Assignment-compliant and reproducible

---

## 📎 Submission Links

* **GitHub Repository:** [https://github.com/YOUR_USERNAME/cnn-rps-pytorch](https://github.com/YOUR_USERNAME/cnn-rps-pytorch)
* **Google Colab Notebook:** *(Paste Colab link here)*

---

## 👨‍🎓 Author

* **Name:** Adnan Zaman Niloy
* **Degree:** B.Sc. in Computer Science & Engineering
* **Interests:** Deep Learning, Computer Vision, Machine Learning

---

## 📝 Acknowledgements

* Kaggle Rock–Paper–Scissors Dataset
* PyTorch & Torchvision Libraries

---

### ✅ Final Note

This project demonstrates not only model implementation but also a **critical understanding of real-world limitations**, which is an essential learning outcome in deep learning systems.

---

