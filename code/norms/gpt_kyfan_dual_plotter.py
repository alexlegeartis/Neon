import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Rebuild after kernel reset

# Define the 12 vertices (all permutations of (±1, ±1, 0))
verts = []
base = [(1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0)]
permute = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
unique = set()
for x,y,z in base:
    for p in permute:
        v = ( (x,y,z)[p[0]], (x,y,z)[p[1]], (x,y,z)[p[2]] )
        unique.add(tuple(v))
verts = np.array(sorted(list(unique)))

# 6 square faces on planes x=±1, y=±1, z=±1
def sq_face(plane, sgn):
    idx = {'x':0,'y':1,'z':2}[plane]
    face = [v for v in verts if np.isclose(v[idx], sgn)]
    other = [i for i in [0,1,2] if i != idx]
    center = np.mean(face, axis=0)
    angles = []
    for v in face:
        vec = np.array([v[other[0]]-center[other[0]], v[other[1]]-center[other[1]]])
        angle = np.arctan2(vec[1], vec[0])
        angles.append(angle)
    order = np.argsort(angles)
    return [face[i] for i in order]

square_faces = []
for plane in ['x','y','z']:
    for sgn in [-1, 1]:
        square_faces.append(sq_face(plane, sgn))

# 8 triangular faces in planes |x|+|y|+|z|=2 (one per octant)
tri_faces = []
for sx in [-1,1]:
    for sy in [-1,1]:
        for sz in [-1,1]:
            tri_faces.append([(sx*1, sy*1, 0), (sx*1, 0, sz*1), (0, sy*1, sz*1)])

faces = square_faces + tri_faces

# Plot solid polyhedron
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
poly = Poly3DCollection(faces, alpha=0.95, edgecolor='darkblue', linewidths=1.5)
ax.add_collection3d(poly)
ax.scatter(verts[:,0], verts[:,1], verts[:,2], s=20, c='darkblue')

res = 41
coords = np.linspace(-1, 1, res)
points = []
for x in coords:
    for y in coords:
        for z in coords:
            arr = np.array([x,y,z])
            # compute norm
            #top2 = np.sort(np.abs(arr))[-2:]
            #norm = np.sum(top2)
            norm = np.max([(np.abs(x)+np.abs(y)+np.abs(z))/2, np.max(np.abs([x,y,z]))])
            if norm <= 1.0001:
                points.append(arr)
points = np.array(points)
# ax.scatter(points[:,0], points[:,1], points[:,2], s=2, alpha=0.1) - to check that polyhedron is correct

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.set_title('Dual Unit Ball (solid) in R^3 for r=2: max(||·||_∞, ||·||_1/2) ≤ 1')
ax.set_box_aspect([1,1,1])
lim = 1.1
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])
ax.set_xticks(np.arange(-1, 1.1, 0.5))
ax.set_yticks(np.arange(-1, 1.1, 0.5))
ax.set_zticks(np.arange(-1, 1.1, 0.5))
plt.show()
fig.savefig('KyFanDual.pdf', dpi=300, bbox_inches='tight')
