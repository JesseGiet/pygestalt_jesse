import numpy as np
import torch
import random
from collections import deque
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import rotate, zoom
import heapq

def make_gabor_kernels(ksize, sigma, lam, gamma=0.5, K=20, device='cpu'):
    """
    Generate K Gabor kernels with different orientations.
    Returns: tensor of shape (K, 1, ksize, ksize)
    """
    kernels = []
    angles = []
    for theta in np.linspace(0, np.pi, K, endpoint=False):  # 0 to 180 degrees
        x, y = np.meshgrid(np.arange(ksize)-ksize//2, np.arange(ksize)-ksize//2)
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gb = np.exp(-0.5 * (x_theta**2 + (gamma*y_theta)**2)/sigma**2) * np.cos(2*np.pi*x_theta/lam)
        gb = gb - gb.mean()  # zero mean
        kernels.append(gb)
        angles.append(theta)
    kernels = np.stack(kernels)[:, None, :, :]  # (K,1,ksize,ksize)
    return torch.tensor(kernels, dtype=torch.float32, device=device), np.array(angles)

def cluster(heatmaps, threshold_ratio=0.8):
    """
    heatmaps: [1, C, H, W] tensor
    threshold_ratio: fraction of max value to consider part of a cluster
    Returns: tensor([value, channel, x, y]) of the centroid of the strongest cluster
    """
    maps = heatmaps[0].cpu().numpy()  # [C, H, W]
    C, H, W = maps.shape

    best_value = -np.inf
    best_coords = None
    best_channel = None

    for c in range(C):
        heatmap = maps[c]
        # threshold map
        thresh = heatmap > (threshold_ratio * heatmap.max())
        # label connected components
        labeled, n = label(thresh)
        if n == 0:
            continue

        # compute cluster values and find the strongest
        for cluster_id in range(1, n+1):
            mask = labeled == cluster_id
            cluster_value = heatmap[mask].sum()
            if cluster_value > best_value:
                best_value = cluster_value
                # centroid weighted by values
                cy, cx = center_of_mass(heatmap, labels=labeled, index=cluster_id)
                best_coords = (cx, cy)
                best_channel = c

    return best_value, best_channel, int(best_coords[0]), int(best_coords[1])

def clusters(heatmaps, threshold_ratio=0.9):
    """
    heatmaps: [1, C, H, W] tensor
    threshold_ratio: fraction of max value to consider part of a cluster
    Returns: list of 14 tuples [(value, channel, x, y), ...] for the top 10 clusters
    Efficient implementation using a min-heap.
    """
    maps = heatmaps[0].cpu().numpy()  # [C, H, W]
    C, H, W = maps.shape

    top_clusters_heap = []  # min-heap

    for c in range(C):
        heatmap = maps[c]
        thresh = heatmap > (threshold_ratio * heatmap.max())
        labeled, n = label(thresh)
        if n == 0:
            continue

        for cluster_id in range(1, n + 1):
            mask = labeled == cluster_id
            cluster_value = heatmap[mask].sum()
            cy0, cx0 = center_of_mass(heatmap, labels=labeled, index=cluster_id)
            cy1 = cy0 + random.uniform(-15, 15)
            cx1 = cx0 + random.uniform(-15, 15)
            # Clamp to stay within the heatmap bounds
            cy2 = max(0, min(cy1, 511))
            cx2 = max(0, min(cx1, 511))
            cluster_tuple = (cluster_value, c, int(cx2), int(cy2))

            if len(top_clusters_heap) < 14:
                heapq.heappush(top_clusters_heap, cluster_tuple)
            else:
                # Only keep cluster if it's bigger than the smallest in heap
                if cluster_value > top_clusters_heap[0][0]:
                    heapq.heapreplace(top_clusters_heap, cluster_tuple)

    # convert heap to sorted list (descending by value)
    top_clusters = sorted(top_clusters_heap, key=lambda x: x[0], reverse=True)
    return top_clusters

def weakest_channel(heatmaps, x, y):
    """
    heatmaps: [1, C, H, W] tensor
    x, y: pixel coordinates (integers)
    Returns: channel index (int) and value (float) of the lowest channel at that coordinate
    """
    # remove batch dimension (assume batch=1)
    maps = heatmaps[0]  # [C, H, W]

    # get values at (y, x) for all channels
    values = maps[:, y, x]  # shape: [C]

    # find channel with min value
    min_val, min_c = torch.min(values, dim=0)

    return int(min_c.item()), float(min_val.item())

def find_closest(image, start_x, start_y):
    """
    Finds the closest 1 in a 2D grid to the given start coordinate (x, y)
    using Euclidean distance.
    
    Args:
        grid (list of list of int) or torch.Tensor: 2D grid of 0s and 1s
        start (tuple): (x, y) coordinate to start from

    Returns:
        tuple or None: Coordinate of the closest 1 as (x, y),
                       or None if no 1 exists
    """
    # Convert grid to tensor if needed
    grid = torch.tensor(image)
    
    # Get coordinates of all 1s
    ys, xs = torch.where(grid == 1)
    
    if len(xs) == 0:  # No 1 in grid
        return None
    
    # Compute Euclidean distances in a vectorized way
    distances = torch.sqrt((xs - start_x)**2 + (ys - start_y)**2)
    
    # Find the index of the minimum distance
    min_idx = torch.argmin(distances)
    
    return int(xs[min_idx]), int(ys[min_idx])

def find_connected(image, start_x, start_y):
    """
    Finds all pixels with value 1 connected to the start pixel (4-connectivity).
    """
    grid = torch.tensor(image)
    rows, cols = grid.shape
    
    if grid[start_y, start_x] != 1:
        return set()  # Starting pixel is not 1

    visited = set()
    queue = deque()
    queue.append((start_x, start_y))
    visited.add((start_x, start_y))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny, nx] == 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return visited

def bounding_square(coords):
    """
    Finds the smallest centered square that contains all given coordinates.

    Args:
        coords (set of tuples): Set of (x, y) points.

    Returns:
        (top_left, bottom_right, side) where:
            top_left     = (x0, y0)
            bottom_right = (x1, y1)
            side         = side length of the square
    """

    if not coords:
        return None, None, 0

    # Extract coordinate arrays
    xs = [x for x, y in coords]
    ys = [y for x, y in coords]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Original bounding box dimensions
    width = x_max - x_min
    height = y_max - y_min

    # Square side is the larger dimension
    side = max(width, height)

    # Center of the rectangle
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    # Top-left and bottom-right of the centered square
    half = side / 2

    x0 = cx - half
    y0 = cy - half
    x1 = cx + half
    y1 = cy + half

    # Round to ints if desired
    # (comment these out if you want float coordinates)
    x0 = int(x0)-2
    y0 = int(y0)-2
    x1 = int(x1)+2
    y1 = int(y1)+2
    side = int(side)+4

    return [(x0, y0), (x1, y1), side]

def rotate_img(img, angle_rad, upscale=4):
    """
    Rotates a binary (0/1) image by angle_rad.
    Uses upscaling before rotation to preserve length better.
    """

    angle_degrees = np.rad2deg(angle_rad)

    # 1. Upscale (nearest neighbor)
    img_big = zoom(img.astype(float), upscale, order=0)

    # 2. Rotate with nearest-neighbor AND no resizing
    rotated_big = rotate(
        img_big,
        angle=angle_degrees,
        reshape=False,
        order=0,          # NN interpolation preserves binary pattern
        mode='constant',
        cval=0.0
    )

    # 3. Downscale back to original size (nearest)
    rotated = zoom(rotated_big, 1/upscale, order=0)

    # ensure 0/1 output
    return rotated.astype(np.uint8)

def draw_segment(angle, N, thickness=2.5, length=12.5, end_round=0.0):
    """
    Generate a centered, blocky line segment of given pixel thickness and length.

    angle      — direction in radians.
    N          — grid size (>= 13, guaranteed by user).
    thickness  — full width of the segment in pixels (≈2.5).
    length     — full length of the segment in pixels (≈12.5).
    end_round  — rounding of the caps (0 = squared ends).
    """

    # Convert full width/length to radius/half-length
    r = thickness / 2        # half-thickness
    L = length / 2           # half-length along the line

    # Generate centered grid
    c = (N - 1) / 2
    Y, X = np.mgrid[0:N, 0:N]
    X = X - c
    Y = Y - c

    # Orientation
    dx = np.cos(angle)
    dy = np.sin(angle)

    # Coordinates relative to line segment
    # t = distance along the direction of the line
    # d = perpendicular distance from line
    t = X * dx + Y * dy
    d = np.abs(-dy * X + dx * Y)

    # Rectangle body of the segment
    body = (np.abs(t) <= L) & (d <= r)

    # Optional rounded caps on the ends
    if end_round > 0:
        cap = (t**2) / (L**2) + (d**2) / ((r + end_round)**2)
        ends = (cap <= 1.0)
        mask = body | ends
    else:
        mask = body

    return mask.astype(np.uint8)