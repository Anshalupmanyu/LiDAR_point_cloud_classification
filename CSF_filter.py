import laspy
import numpy as np
import CSF  # Ensure this module is correctly installed and available

# Read the LAS file
inFile = laspy.read(r"pc.las")
points = inFile.points
xyz = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()  # Extract x, y, z into a NumPy array

# Initialize the CSF filter
csf = CSF.CSF()

# Parameter settings including 'relief'
csf.params.bSloopSmooth = False
csf.params.cloth_resolution = 0.5
csf.params.relief = 2.0  # Adjust this value based on your terrain needs

# Set the point cloud
csf.setPointCloud(xyz)

# Create empty lists to hold the indices of ground and non-ground points
ground = CSF.VecInt()  # A list to hold the index of ground points
non_ground = CSF.VecInt()  # A list to hold the index of non-ground points

# Perform filtering
csf.do_filtering(ground, non_ground)

# Convert indices to NumPy arrays
ground_indices = np.array(ground)
non_ground_indices = np.array(non_ground)

# Create the output LAS file for ground points
ground_las = laspy.LasData(inFile.header)
ground_las.points = points[ground_indices]  # Extract ground points and save them

# Save the ground points to a LAS file
ground_las.write(r"ground_points.las")

# Create the output LAS file for off-ground (non-ground) points
non_ground_las = laspy.LasData(inFile.header)
non_ground_las.points = points[non_ground_indices]  # Extract off-ground points and save them

# Save the off-ground points to a LAS file
non_ground_las.write(r"off_ground_points.las")

print("Ground and off-ground points have been saved successfully.")
