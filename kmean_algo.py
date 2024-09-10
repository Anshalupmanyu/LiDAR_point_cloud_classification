import laspy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Load LAS file
input_file = "refined_classified_points.las"
las = laspy.read(input_file)

# Extract the point cloud data
points = np.vstack((las.x, las.y, las.z)).transpose()

# Apply K-Means clustering
n_clusters = 3  # Define the number of clusters (adjust based on your data)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(points)

# Get cluster labels from K-Means
labels = kmeans.labels_

# Create a dataframe to hold point cloud data and cluster labels
df = pd.DataFrame(np.column_stack((las.x, las.y, las.z, labels)), columns=['x', 'y', 'z', 'cluster_label'])

# Analyze cluster characteristics
cluster_summary = df.groupby('cluster_label').agg({
    'z': ['mean', 'min', 'max'],
    'x': 'count'
})
print(cluster_summary)

# Manually adjust cluster label assignments based on inspection
df['classification'] = 'Unclassified'  # Default
df.loc[df['cluster_label'] == 0, 'classification'] = 'Building'  # Adjust based on cluster characteristics
df.loc[df['cluster_label'] == 1, 'classification'] = 'Ground'
df.loc[df['cluster_label'] == 2, 'classification'] = 'Vegetation'

# Define colors (for visualization if needed)
color_map = {'Ground': 'Black', 'Building': 'red', 'Vegetation': 'green'}

# Convert classification to numeric code for LAS format
classification_codes = {'Ground': 2, 'Building': 6, 'Vegetation': 3, 'Unclassified': 1}
df['classification_code'] = df['classification'].map(classification_codes)

# Create a new LAS file to store
output_file = "kmeans_classified_pc.las"
las_out = laspy.LasData(las.header)

# Write the point cloud data
las_out.x = df['x'].values
las_out.y = df['y'].values
las_out.z = df['z'].values
las_out.classification = df['classification_code'].astype(np.uint8).values  # Fixed data type

# Write the LAS file to disk
las_out.write(output_file)
