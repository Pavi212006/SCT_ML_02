import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. AUTO-DATA GENERATION
print("Generating Secure Customer Database")
np.random.seed(42)
income_data = np.random.randint(20, 110, 250)
spending_data = np.random.randint(5, 95, 250)
client_registry = pd.DataFrame({'Income': income_data, 'Score': spending_data})

# 2. TRAINING MODEL
features = client_registry.values
segment_model = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
cluster_labels = segment_model.fit_predict(features)

# 3. AUTO-PREDICTION
print("Running Auto-Analysis for Test Customer")
test_pred = segment_model.predict([[65, 40]])
print(f"REPORT: Customer Segment identified as: {test_pred[0]}")

# 4. SHOW GRAPH
plt.figure(figsize=(10, 7))
plt.scatter(features[:, 0], features[:, 1], c=cluster_labels, cmap='plasma', alpha=0.7)
plt.scatter(segment_model.cluster_centers_[:, 0], segment_model.cluster_centers_[:, 1], s=200, c='black', marker='X')
plt.title('SkillCraft Task 02: Final Customer Segments')
plt.show()
