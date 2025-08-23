import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

def generate_ball_points(norm_func, num_points=1000, bounds=(-2, 2)):
    """
    Generate points on the surface of the ball defined by the given norm function.
    
    Args:
        norm_func: Function that computes the norm of a point
        num_points: Number of points to generate
        bounds: Bounds for the search space
    
    Returns:
        points: Array of points on the surface
    """
    points = []
    
    # Generate random points and filter those with norm close to 1
    for _ in range(num_points * 10):  # Generate more points to account for filtering
        if len(points) >= num_points:
            break
            
        # Random point in the search space
        point = np.random.uniform(bounds[0], bounds[1], 3)
        
        # Compute norm
        norm = norm_func(point)
        
        # Check if point is on the surface (within tolerance)
        if abs(norm - 1.0) < 0.05:
            points.append(point)
    
    return np.array(points)

def max_norm_inf_l1(point, alpha=0.5):
    """
    Compute the norm max{||y||_∞, α||y||_1}
    
    Args:
        point: 3D point (x, y, z)
        alpha: Weight for L1 norm (default 0.5)
    
    Returns:
        norm value
    """
    linf_norm = np.max(np.abs(point))
    l1_norm = np.sum(np.abs(point))
    return max(linf_norm, alpha * l1_norm)

def plot_3d_ball(norm_func, title="1-Ball in R³", num_points=2000, alpha=0.5):
    """
    Plot the 1-ball in R³ as a solid 3D body.
    
    Args:
        norm_func: Function that computes the norm
        title: Plot title
        num_points: Number of points to generate
        alpha: Weight for L1 norm
    """
    # Generate points on the surface
    surface_points = generate_ball_points(norm_func, num_points)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface points
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], 
               c='blue', alpha=0.6, s=1, label='Surface points')
    
    # Add some interior points to make it look more solid
    interior_points = []
    for _ in range(num_points // 2):
        point = np.random.uniform(-0.8, 0.8, 3)
        if norm_func(point) < 1.0:
            interior_points.append(point)
    
    if interior_points:
        interior_points = np.array(interior_points)
        ax.scatter(interior_points[:, 0], interior_points[:, 1], interior_points[:, 2], 
                   c='lightblue', alpha=0.3, s=0.5, label='Interior points')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\nNorm: max{{||y||_∞, {alpha}||y||_1}}')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set axis limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    # Add coordinate axes
    ax.plot([0, 1.5], [0, 0], [0, 0], 'r-', linewidth=2, label='X-axis')
    ax.plot([0, 0], [0, 1.5], [0, 0], 'g-', linewidth=2, label='Y-axis')
    ax.plot([0, 0], [0, 0], [0, 1.5], 'b-', linewidth=2, label='Z-axis')
    
    plt.tight_layout()
    return fig, ax

def plot_3d_ball_enhanced(norm_func, title="1-Ball in R³", num_points=5000, alpha=0.5):
    """
    Enhanced 3D plot of the 1-ball with better solid body visualization.
    
    Args:
        norm_func: Function that computes the norm
        title: Plot title
        num_points: Number of points to generate
        alpha: Weight for L1 norm
    """
    # Generate more points for better surface coverage
    surface_points = generate_ball_points(norm_func, num_points * 2)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a more solid appearance by adding multiple layers
    # Surface layer
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], 
               c='darkblue', alpha=0.8, s=2, label='Surface')
    
    # Interior layers for solid appearance
    for layer in range(1, 4):
        scale = 0.9 - layer * 0.2
        if scale > 0:
            interior_points = []
            for _ in range(num_points // 2):
                point = np.random.uniform(-scale, scale, 3)
                if norm_func(point) < scale:
                    interior_points.append(point)
            
            if interior_points:
                interior_points = np.array(interior_points)
                color_intensity = 0.8 - layer * 0.2
                ax.scatter(interior_points[:, 0], interior_points[:, 1], interior_points[:, 2], 
                           c='lightblue', alpha=color_intensity, s=1, label=f'Layer {layer}' if layer == 1 else "")
    
    # Add coordinate axes with better visibility
    ax.plot([0, 1.5], [0, 0], [0, 0], 'r-', linewidth=3, label='X-axis')
    ax.plot([0, 0], [0, 1.5], [0, 0], 'g-', linewidth=3, label='Y-axis')
    ax.plot([0, 0], [0, 0], [0, 1.5], 'b-', linewidth=3, label='Z-axis')
    
    # Add origin point
    ax.scatter([0], [0], [0], c='black', s=100, marker='o', label='Origin')
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'{title}\nNorm: max{{||y||_∞, {alpha}||y||_1}}', fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio and limits
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    
    # Add grid and styling
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add some key points with labels
    key_points = [
        ([1, 0, 0], 'A'),
        ([0, 1, 0], 'B'),
        ([0, 0, 1], 'C'),
        ([1, 1, 0], 'D'),
        ([0.5, 0.5, 0.5], 'E')
    ]
    
    for point, label in key_points:
        ax.scatter(point[0], point[1], point[2], c='red', s=50, marker='s')
        ax.text(point[0] + 0.05, point[1] + 0.05, point[2] + 0.05, label, 
                fontsize=12, fontweight='bold', color='red')
    
    plt.tight_layout()
    return fig, ax

def create_parametric_surface(norm_func, alpha=0.5, resolution=100):
    """
    Create a parametric surface representation of the ball.
    
    Args:
        norm_func: Function that computes the norm
        alpha: Weight for L1 norm
        resolution: Resolution of the surface grid
    
    Returns:
        X, Y, Z: Surface coordinates
    """
    # Create spherical coordinates
    phi = np.linspace(0, 2*np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)
    PHI, THETA = np.meshgrid(phi, theta)
    
    # Initialize surface arrays
    X = np.zeros_like(PHI)
    Y = np.zeros_like(PHI)
    Z = np.zeros_like(PHI)
    
    # For each direction, find the distance to the surface
    for i in range(resolution):
        for j in range(resolution):
            # Direction vector
            direction = np.array([
                np.sin(THETA[i, j]) * np.cos(PHI[i, j]),
                np.sin(THETA[i, j]) * np.sin(PHI[i, j]),
                np.cos(THETA[i, j])
            ])
            
            # Binary search for the surface point
            left, right = 0, 2
            tolerance = 0.001
            
            while right - left > tolerance:
                mid = (left + right) / 2
                point = mid * direction
                norm_val = norm_func(point)
                
                if norm_val < 1.0:
                    left = mid
                else:
                    right = mid
            
            # Store the surface point
            X[i, j] = left * direction[0]
            Y[i, j] = left * direction[1]
            Z[i, j] = left * direction[2]
    
    return X, Y, Z

def plot_parametric_surface(norm_func, alpha=0.5, resolution=50):
    """
    Plot the ball using parametric surface representation.
    
    Args:
        norm_func: Function that computes the norm
        alpha: Weight for L1 norm
        resolution: Resolution of the surface grid
    """
    print("Creating parametric surface...")
    X, Y, Z = create_parametric_surface(norm_func, alpha, resolution)
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surface = ax.plot_surface(X, Y, Z, 
                              cmap='viridis', 
                              alpha=0.8,
                              linewidth=0,
                              antialiased=True)
    
    # Add colorbar
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'Parametric Surface of 1-Ball\nNorm: max{{||y||_∞, {alpha}||y||_1}}', 
                 fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio and limits
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    
    # Add coordinate axes
    ax.plot([0, 1.5], [0, 0], [0, 0], 'r-', linewidth=3, label='X-axis')
    ax.plot([0, 0], [0, 1.5], [0, 0], 'g-', linewidth=3, label='Y-axis')
    ax.plot([0, 0], [0, 0], [0, 1.5], 'b-', linewidth=3, label='Z-axis')
    
    plt.tight_layout()
    return fig, ax

def plot_2d_slices(norm_func, alpha=0.5):
    """
    Plot 2D slices of the 3D ball to better understand its shape.
    
    Args:
        norm_func: Function that computes the norm
        alpha: Weight for L1 norm
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slice_titles = ['XY-plane (z=0)', 'XZ-plane (y=0)', 'YZ-plane (x=0)']
    
    for i, (ax, title) in enumerate(zip(axes, slice_titles)):
        # Generate grid points
        x = np.linspace(-1.5, 1.5, 200)
        y = np.linspace(-1.5, 1.5, 200)
        X, Y = np.meshgrid(x, y)
        
        # Create the third coordinate (0 for the slice)
        Z = np.zeros_like(X)
        
        # Compute norms for each point
        norms = np.zeros_like(X)
        for i_x in range(X.shape[0]):
            for i_y in range(X.shape[1]):
                if i == 0:  # XY-plane
                    point = np.array([X[i_x, i_y], Y[i_x, i_y], 0])
                elif i == 1:  # XZ-plane
                    point = np.array([X[i_x, i_y], 0, Y[i_x, i_y]])
                else:  # YZ-plane
                    point = np.array([0, X[i_x, i_y], Y[i_x, i_y]])
                
                norms[i_x, i_y] = norm_func(point)
        
        # Plot contour where norm = 1
        contour = ax.contour(X, Y, norms, levels=[1.0], colors='red', linewidths=2)
        
        # Fill the interior
        ax.contourf(X, Y, norms, levels=[0, 1.0], alpha=0.3, colors='lightblue')
        
        ax.set_xlabel('X' if i != 2 else 'Y')
        ax.set_ylabel('Y' if i != 1 else 'Z')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
    
    plt.suptitle(f'2D Slices of the 1-Ball\nNorm: max{{||y||_∞, {alpha}||y||_1}}')
    plt.tight_layout()
    return fig

def analyze_norm_properties(norm_func, alpha=0.5):
    """
    Analyze and print properties of the norm.
    
    Args:
        norm_func: Function that computes the norm
        alpha: Weight for L1 norm
    """
    print(f"Norm: max{{||y||_∞, {alpha}||y||_1}}")
    print("=" * 50)
    
    # Test some key points
    test_points = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1, 0]),
        np.array([0.5, 0.5, 0.5]),
        np.array([0.8, 0.8, 0.8])
    ]
    
    print("Test points and their norms:")
    for i, point in enumerate(test_points):
        norm = norm_func(point)
        print(f"Point {i+1}: {point} → Norm: {norm:.3f}")
    
    # Find the maximum extent in each direction using binary search
    print("\nMaximum extent in each direction:")
    for direction in ['x', 'y', 'z']:
        left, right = 0, 2
        tolerance = 0.001
        
        # Binary search for the maximum extent
        while right - left > tolerance:
            mid = (left + right) / 2
            if direction == 'x':
                point = np.array([mid, 0, 0])
            elif direction == 'y':
                point = np.array([0, mid, 0])
            else:
                point = np.array([0, 0, mid])
            
            norm_val = norm_func(point)
            if norm_val < 1.0:
                left = mid
            else:
                right = mid
        
        max_extent = left
        print(f"{direction.upper()}-direction: ±{max_extent:.3f}")
    
    # Additional analysis
    print("\nNorm properties:")
    print(f"- For points on coordinate axes: ||(a,0,0)|| = max{{|a|, {alpha}|a|}} = |a|")
    print(f"- For points like (a,a,0): ||(a,a,0)|| = max{{|a|, {alpha}*2|a|}} = max{{|a|, {2*alpha}|a|}}")
    if 2*alpha > 1:
        print(f"  Since {2*alpha} > 1, ||(a,a,0)|| = {2*alpha}|a| when |a| > 0")
    else:
        print(f"  Since {2*alpha} ≤ 1, ||(a,a,0)|| = |a| for all a")
    
    # Find the transition point where L∞ and L1 norms are equal
    if alpha < 1:
        transition_point = 1 / alpha
        print(f"- Transition point: ||y||_∞ = {alpha}||y||_1 when ||y||_1 = {transition_point}")
        print(f"  This means for points with L1 norm > {transition_point}, the L1 term dominates")

def main():
    """
    Main function to create all plots and analysis.
    """
    # Define the norm function with alpha = 0.5
    alpha = 0.5
    norm_func = lambda point: max_norm_inf_l1(point, alpha)
    
    print("Creating 3D plot of the 1-ball...")
    fig_3d, ax_3d = plot_3d_ball(norm_func, alpha=alpha)
    
    print("Creating enhanced 3D plot...")
    fig_enhanced, ax_enhanced = plot_3d_ball_enhanced(norm_func, alpha=alpha)
    
    print("Creating parametric surface plot...")
    fig_param, ax_param = plot_parametric_surface(norm_func, alpha=alpha, resolution=40)
    
    print("Creating 2D slice plots...")
    fig_2d = plot_2d_slices(norm_func, alpha=alpha)
    
    print("Analyzing norm properties...")
    analyze_norm_properties(norm_func, alpha=alpha)
    
    # Show all plots
    plt.show()
    
    # Save plots
    fig_3d.savefig('1_ball_3d.png', dpi=300, bbox_inches='tight')
    fig_enhanced.savefig('1_ball_3d_enhanced.png', dpi=300, bbox_inches='tight')
    fig_param.savefig('1_ball_parametric.png', dpi=300, bbox_inches='tight')
    fig_2d.savefig('1_ball_2d_slices.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved as:")
    print("- '1_ball_3d.png' (basic 3D)")
    print("- '1_ball_3d_enhanced.png' (enhanced solid body)")
    print("- '1_ball_parametric.png' (parametric surface)")
    print("- '1_ball_2d_slices.png' (2D slices)")

if __name__ == "__main__":
    main()
