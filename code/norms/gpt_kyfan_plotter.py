import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Generate all vertices that satisfy the constraints
verts = set()
for x in [-1, -0.5, 0, 0.5, 1]:
    for y in [-1, -0.5, 0, 0.5, 1]:
        for z in [-1, -0.5, 0, 0.5, 1]:
            if abs(x) + abs(y) <= 1 + 1e-9 and abs(x) + abs(z) <= 1 + 1e-9 and abs(y) + abs(z) <= 1 + 1e-9:
                verts.add((x, y, z))

verts = [
    ( 1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    ( 0.0, 1.0, 0.0), ( 0.0,-1.0, 0.0),
    ( 0.0, 0.0, 1.0), ( 0.0, 0.0,-1.0),
]
for sx in (+0.5, -0.5):
    for sy in (+0.5, -0.5):
        for sz in (+0.5, -0.5):
            verts.append((sx, sy, sz))
verts = np.array(sorted(list(verts)))
# Convex hull of points
from scipy.spatial import ConvexHull
points = np.array(verts)
hull = ConvexHull(points)

# Plot
fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(111, projection='3d')
for simplex in hull.simplices:
    tri = [points[i] for i in simplex]
    ax.add_collection3d(Poly3DCollection([tri], alpha=0.95, edgecolor='darkblue', linewidths=1.5))
ax.scatter(verts[:,0], verts[:,1], verts[:,2], s=20, c='darkblue')

ax.set_box_aspect([1,1,1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])


ax.set_xticks(np.arange(-1, 1.1, 0.5))
ax.set_yticks(np.arange(-1, 1.1, 0.5))
ax.set_zticks(np.arange(-1, 1.1, 0.5))
plt.show()

fig.savefig('KyFan.pdf', dpi=300, bbox_inches='tight')

