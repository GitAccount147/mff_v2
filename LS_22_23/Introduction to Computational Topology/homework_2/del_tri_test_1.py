import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# A 2D array of points
points = np.array([[3.0, 0.0], [2.0, 0.0], [2.0, 0.75],
                   [2.5, 0.75], [2.5, 0.6], [2.25, 0.6],
                   [2.25, 0.2], [3.0, 0.2], [3.0, 0.0]])

# Perform Delaunay triangulation
tri = Delaunay(points)

# Visualize the triangulation
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()