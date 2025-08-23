import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def minkowski_sum_boundary(alpha, num_angles=2000):
    r1 = alpha            # L1-ball radius (diamond)
    r2 = 1.0 - alpha      # L2-ball radius (circle)
    thetas = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    pts = []
    for th in thetas:
        u = np.array([np.cos(th), np.sin(th)])  # direction
        
        # Support point on r2 * L2 ball is simply r2 * u
        s2 = r2 * u
        
        # Support point on r1 * L1 ball:
        # pick the axis with larger |component|, place full mass there with the sign
        if abs(u[0]) >= abs(u[1]):
            s1 = np.array([r1 * np.sign(u[0]), 0.0])
        else:
            s1 = np.array([0.0, r1 * np.sign(u[1])])
        
        pts.append(s1 + s2)
    return np.array(pts)

def l1_ball_boundary(radius, num=800):
    t = np.linspace(0, 2*np.pi, num, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    denom = np.maximum(np.abs(x) + np.abs(y), 1e-12)
    return radius * x / denom, radius * y / denom

def l2_ball_boundary(radius, num=800):
    t = np.linspace(0, 2*np.pi, num, endpoint=False)
    return radius * np.cos(t), radius * np.sin(t)

# List of alpha values to test
alphas = [0.25, 0.5, 0.75]
colormap = cm.get_cmap('viridis')
colors = [colormap(i / (len(alphas) - 1)) for i in range(len(alphas) - 1, -1, -1)]
# Plot
plt.figure(figsize=(8, 8))

# Plot Minkowski sums for each alpha
for alpha, color in zip(alphas, colors):
    sum_pts = minkowski_sum_boundary(alpha)
    plt.plot(sum_pts[:, 0], sum_pts[:, 1], label=f"F*$^\dagger$-ball (α={alpha})", 
             linewidth=2, color=color)

# Plot reference L1 and L2 balls
x1, y1 = l1_ball_boundary(1)
x2, y2 = l2_ball_boundary(1)
plt.plot(x1, y1, linestyle="--", label="L1-ball", color="black", alpha=0.9)
plt.plot(x2, y2, linestyle="--", label="L2-ball", color="gray", alpha=0.5)

# Configure plot
plt.gca().set_aspect("equal", adjustable="box")
# plt.title("Minkowski sums: α L1-ball ⊕ (1-α) L2-ball for different α")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper right")
plt.grid(True, linestyle=":")
plt.xticks(np.arange(-1, 1.1, 0.5))
plt.yticks(np.arange(-1, 1.1, 0.5))
plt.savefig("fstardualball.pdf", dpi=300, bbox_inches='tight')
plt.show()
