## 🛒 Task 02: Retail Customer Grouping (Unsupervised Learning)

### 📌 Project Overview
This project focuses on **Market Basket Analysis** and **Customer Segmentation**. Using the **K-Means Clustering** algorithm, I developed a system that categorizes retail customers into distinct groups based on their purchasing power (Annual Income) and behavior (Spending Score).

### 🚀 Key Features
- **Unsupervised Learning:** Implemented `K-Means++` to automatically identify 5 natural customer segments.
- **Centroid Logic:** Precisely calculated the center-point of each group to define typical customer profiles.
- **Automated Labeling:** The system interprets raw cluster data into human-readable categories:
  - *High-Value Targets* (High Income, High Spend)
  - *Sensible Shoppers* (High Income, Low Spend)
  - *Standard Consumers* (Average Profile)
  - *Impulsive Buyers* (Low Income, High Spend)
  - *Conservative Clients* (Low Income, Low Spend)
- **Visual Analytics:** Generates a 2D cluster map with clearly marked centroids for business reporting.

### 🛠️ Tech Stack
- **Language:** Python 3.10
- **Algorithm:** K-Means Clustering (Scikit-Learn)
- **Visualization:** Matplotlib (Cluster Mapping)
- **Data Handling:** Pandas & NumPy

### 📊 How to Run
    Run the script:
   python SCT_02.py
