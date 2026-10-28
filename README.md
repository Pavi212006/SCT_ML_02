# Task 02: Customer Segmentation with K-Means

## Overview

This repository contains an educational example of **K-Means clustering**, an unsupervised machine-learning algorithm. It groups generated records using two numeric features: annual income and spending score.

**Important limitation:** the script generates random synthetic values. It does not use a real retail/customer dataset, and the generated points do not represent measured customer behaviour. Treat this as an algorithm demonstration, not a validated customer-segmentation business tool.

## What the script does

1. Generates 250 reproducible synthetic records.
2. Standardizes the two features before clustering.
3. Fits K-Means with five clusters and a fixed random seed.
4. Prints the average feature values and record count for each cluster.
5. Calculates a silhouette score as a rough measure of how separated the clusters are in this generated dataset.
6. Assigns one hypothetical example record to a cluster.
7. Plots the records and cluster centroids in the original feature scale.

Cluster IDs (0–4) are arbitrary labels. They do not automatically mean “high-value,” “impulsive,” or any other business category.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- scikit-learn
- Matplotlib

Install dependencies:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

## Run

```bash
python SCT_02.py
```

The program prints cluster profiles and opens a plot. Close the plot window to finish the run.

## Concepts demonstrated

- **Unsupervised learning:** finding groups without pre-existing class labels.
- **K-Means:** assigns each record to the nearest centroid and iteratively updates centroids.
- **Feature scaling:** standardizes features so their numeric ranges do not dominate Euclidean distance.
- **Centroid:** the centre of a cluster in feature space.
- **Silhouette score:** estimates how well-separated clusters are; values nearer 1 generally indicate better separation, values near 0 indicate overlap, and negative values can indicate questionable assignments.

## Limitations and next steps

- Randomly generated features are not a substitute for real customer data.
- A silhouette score alone does not prove the clusters are useful or meaningful.
- The choice of five clusters is fixed for demonstration rather than justified by a business requirement.
- Before using this approach for a real business problem, use an appropriate, documented dataset, inspect missing values and outliers, compare different values of *k*, evaluate stability, and validate the segments with domain experts.
