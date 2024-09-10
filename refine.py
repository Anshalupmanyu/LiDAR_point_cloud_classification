import laspy
import numpy as np

# Load the LAS file with classified points (ground and off-ground)
las = laspy.read("classified_points.las")

# Convert the point cloud to a numpy array for processing
points = np.vstack([las.x, las.y, las.z]).T

# Assume ground points are already classified in the LAS file (e.g., classification code 2 for ground)
# We'll refine the off-ground points (non-ground) to classify buildings based on height

# Get mask for ground points (already classified as ground, assuming class 2)
ground_points_mask = las.classification == 2

# Get mask for off-ground points (anything not classified as ground)
off_ground_points_mask = ~ground_points_mask

# Extract off-ground points for further classification
off_ground_points = points[off_ground_points_mask]

# Define a height threshold for building classification
height_threshold = 2.5  # Adjust this threshold based on your data

# Classify off-ground points as buildings if they exceed the height threshold
building_points_mask = off_ground_points[:, 2] > height_threshold

# Update classification in the original LAS file
# Set classification: 6 for buildings, retain existing classification for other points
las.classification[off_ground_points_mask] = np.where(building_points_mask, 6, las.classification[off_ground_points_mask])

# Create a new LAS file to save the refined data
header = las.header
output_las = laspy.create(point_format=header.point_format, file_version=header.version)
output_las.points = las.points
output_las.classification = las.classification

# Save the refined LAS data
output_las.write("refined_classified_points.las")