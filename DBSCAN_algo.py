import laspy
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt  # Fixed plt import
import matplotlib.colors as mcolors

# Load LAS file
input_file = "classified_points.las"
las = laspy.read(input_file)

# Extract the point cloud data
points = np.vstack((las.x, las.y, las.z)).transpose()

# Apply DBSCAN clustering
eps = 2.5
min_samples = 10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

# Get cluster labels
labels = db.labels_

# Create a dataframe
df = pd.DataFrame(np.column_stack((las.x, las.y, las.z, labels)), columns=['x', 'y', 'z', 'cluster_label'])

# Classify clusters by 'z' and other features
df['classification'] = 'Unclassified'

# Threshold classification
df.loc[df['z'] < 8.5, 'classification'] = 'Ground'
df.loc[(df['z'] > 8.5) & (df['z'] < 9.5), 'classification'] = 'Vegetation'
df.loc[df['z'] > 9.5, 'classification'] = 'Building'

# Define colors (for visualization if needed)
color_map = {'Ground': 'Black', 'Building': 'red', 'Vegetation': 'green'}

# Convert classification to numeric code
classification_codes = {'Ground': 2, 'Building': 6, 'Vegetation': 3, 'Unclassified': 1}
df['classification_code'] = df['classification'].map(classification_codes)

# Create a new LAS file to store
output_file = "classified_pc1.las"
las_out = laspy.LasData(las.header)

# Write the point cloud data
las_out.x = df['x'].values
las_out.y = df['y'].values
las_out.z = df['z'].values
las_out.classification = df['classification_code'].astype(np.uint8).values  # Fixed data type

# Write the LAS file to disk
las_out.write(output_file)  # Corrected the write method

