# 🔬 Advanced EDA for Genomic Data Analysis
**Identifying Genetic Variation through Visualization & Clustering**

## 📌 Overview
This project applies **Autoencoders & K-Means Clustering** on genomic datasets from `gnomAD`, visualizing patterns of genetic variations.

### 🚀 Features:
✅ **Autoencoder-based dimensionality reduction**  
✅ **K-Means clustering for variant segmentation**  
✅ **Interactive Web App (Flask & HTML/CSS)**  
✅ **Visualization with PCA & Seaborn**  
✅ **Model evaluation with Silhouette Score & Davies-Bouldin Index**  


---

## 📊 **Visualization**
### 🧬 PCA-Based Cluster Plot:
![Clustering Visualization](outputs/clustering_visualization.png)

---

## 🔥 **How to Run the Project**
### 1️⃣ Install Dependencies:
```sh
pip install -r requirements.txt

### 2️⃣ Train the Autoencoder & K-Means:
```sh
python src/model_training.py  
python src/clustering.py  

### 3️⃣ Start Flask Web App:
```sh
python app.py  

## 🛠️ Model Evaluation
📌 Silhouette Score: 0.47 (Higher is better)
📌 Davies-Bouldin Index: 0.74 (Lower is better)

These metrics confirm well-separated clusters with good data representation!

