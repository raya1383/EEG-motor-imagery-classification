# EEG-motor-imagery-classification
Here is a clean, professional **README.md** you can copy directly into your GitHub repo.

---

# 🧠 Motor Imagery EEG Classification Project

## 📌 Overview

This project focuses on processing and analyzing **EEG (Electroencephalogram)** signals for **motor imagery classification** using machine learning techniques.
Motor imagery refers to imagining a movement (e.g., left hand or foot movement) without physically performing it. EEG signals recorded during this process can be used in **Brain–Computer Interface (BCI)** systems.

The goal of this project is to build a complete EEG signal processing pipeline, from preprocessing to classification and clustering.

Dataset used: **BCI Competition IV – Dataset 1a** (`BCICIV_calib_ds1a.mat`)

---

## 🎯 Objectives

* Understand EEG motor imagery data
* Preprocess and clean EEG signals
* Extract meaningful features from EEG
* Perform classification using machine learning
* Evaluate model performance
* Apply clustering methods for exploration

---

## 🗂️ Project Pipeline

### 1️⃣ Data Loading & Segmentation

* Load EEG dataset (`BCICIV_calib_ds1a.mat`)
* Segment continuous signals into time windows
* Extract trials using event positions (`pos`)
* Create labeled dataset (features + labels)
* Split data:

  * **75% training**
  * **25% testing**

---

### 2️⃣ Preprocessing

#### Band-pass filtering

Focus on frequency bands relevant to motor imagery:

* **Mu band:** 8–13 Hz
* **Beta band:** 13–30 Hz

Apply band-pass filters to remove noise and keep relevant brain activity.

#### Visualization

Plot EEG signals for selected channels:

* Channel 0
* Channel 15
* Channel 30
* Channel 45
* Channel 59

---

### 3️⃣ Feature Extraction

#### Common Spatial Patterns (CSP)

* Extract spatial features from multi-channel EEG
* Reduce dimensionality
* Improve class separability

#### t-SNE Visualization

* Visualize data in 2D space
* Compare feature distribution:

  * Before CSP
  * After CSP

---

### 4️⃣ Classification

#### Main model: Kernel SVM (RBF)

* Implement SVM with RBF kernel
* Train on extracted features
* Compare with `scikit-learn` implementation

#### Evaluation metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC curve

---

### 5️⃣ Comparison with Other Models

Apply and compare with other ML models (using scikit-learn), for example:

* Random Forest
* KNN
* Logistic Regression
* Decision Tree

Compare performance with SVM results.

---

### 6️⃣ Clustering

#### K-means clustering

* Apply K-means on extracted features
* Visualize clusters in 2D space

#### Optimal number of clusters

Use:

* WCSS (Elbow method)
* Silhouette score

---

## 🛠️ Technologies Used

* Python
* NumPy
* SciPy
* Matplotlib
* Scikit-learn
* MNE (optional for EEG)
* Seaborn

---

## 📊 Expected Outputs

* Preprocessed EEG signals
* Filtered signals (mu & beta bands)
* CSP feature visualization
* t-SNE scatter plots
* Classification metrics and plots
* Clustering visualizations

---

## 📁 Dataset

BCI Competition IV – Dataset 1a
File used:

```
BCICIV_calib_ds1a.mat
```

Download from official BCI Competition website.

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
pip install -r requirements.txt
```

Run notebooks or scripts step by step.

---
