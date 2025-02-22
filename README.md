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

## 📊 **Flask App**
![image_alt]()



---

## 📊 **Visualization**
### 🧬 PCA-Based Cluster Plot:
###### ![Clustering Visualization](https://github.com/bhavana-1011/Advanced-EDA-for-Genomic-Data-Analysis/blob/main/clustering_visualization.png)

---

## 🔥**How to Run the Project**

### 1️⃣ Install dependencies
pip install -r requirements.txt  

### 2️⃣ Train the Autoencoder model  
python src/model_training.py  

### 3️⃣ Perform clustering  
python src/clustering.py  

### 4️⃣ Run the Flask web app  
python app.py  

## 🛠️ Model Evaluation
📌 Silhouette Score: 0.47 (Higher is better)
📌 Davies-Bouldin Index: 0.74 (Lower is better)

These metrics confirm well-separated clusters with good data representation!

