# -*- coding: utf-8 -*-
"""
Optimized all-in-one version with parallel processing - maintains exact same predictions
Uses all available CPU cores automatically
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import math
import pickle
import ot
from math import tanh
from scipy.spatial.distance import pdist, squareform
from numba import jit, prange
from joblib import Parallel, delayed
from tslearn.metrics import dtw
import warnings
import os
warnings.filterwarnings('ignore')

# Create Results directory if it doesn't exist
os.makedirs('Results', exist_ok=True)

# =============================================================================
# Optimized Functions with Numba
# =============================================================================

def cluster_based_on_threshold_dtw(array_list,threshold):
    """
    Compute a pairwise DTW distance matrix for a list of 4D trajectories using tslearn.
    
    Parameters:
        l_mat (List[np.ndarray]): List of Nx4 numpy arrays (trajectories).
        
    Returns:
        np.ndarray: Symmetric distance matrix of shape (len(l_mat), len(l_mat))
    """
    n = len(l_mat)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist = dtw(l_mat[i], l_mat[j])  # tslearn handles multi-dimensional DTW
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric
    cluster_labels = np.full(len(array_list), -1)
    current_cluster = 0
    for i in range(len(array_list)):
        if cluster_labels[i] == -1:
            cluster_labels[i] = current_cluster
            for j in range(i + 1, len(array_list)):
                if dist_matrix[i, j] <= threshold:
                    cluster_labels[j] = current_cluster
            current_cluster += 1
    return cluster_labels

@jit(nopython=True)
def compute_overlap_numba(row1, row2):
    """Numba-optimized overlap computation"""
    overlap = 1.0
    union = 1.0
    for i in range(3):
        low1, high1 = row1[2*i], row1[2*i+1]
        low2, high2 = row2[2*i], row2[2*i+1]
        intersect_low = max(low1, low2)
        intersect_high = min(high1, high2)
        union_low = min(low1, low2)
        union_high = max(high1, high2)
        
        if intersect_high > intersect_low:
            overlap *= (intersect_high - intersect_low)
        else:
            return 0.0
        union *= (union_high - union_low)
    return overlap / union if union > 0 else 0.0

def filter_overlaps(df):
    """Optimized with numba"""
    df_array = df.iloc[:, :7].values
    n_rows = len(df)
    
    while True:
        to_drop = set()
        for i in range(n_rows):
            if i in to_drop:
                continue
            for j in range(i + 1, n_rows):
                if j in to_drop:
                    continue
                overlap_ratio = compute_overlap_numba(df_array[i], df_array[j])
                if overlap_ratio > 0.25:
                    if df_array[i, 6] > df_array[j, 6]:
                        to_drop.add(i)
                    else:
                        to_drop.add(j)
        
        if not to_drop:
            break
            
        keep_idx = [i for i in range(n_rows) if i not in to_drop]
        df = df.iloc[keep_idx]
        df_array = df.iloc[:, :7].values
        n_rows = len(df)
    
    return df

@jit(nopython=True)
def euclidean_distance_numba(x1, y1, x2, y2):
    """Numba-optimized euclidean distance"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class UnionFind:
    """Unchanged - already efficient"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def group_points(points, threshold):
    """Optimized with numba distance calculation"""
    n = len(points)
    uf = UnionFind(n)
    
    # Convert to numpy array for faster access
    points_array = np.array([(p['x'], p['y']) for p in points])
    
    for i in range(n):
        for j in range(i + 1, n):
            if euclidean_distance_numba(points_array[i, 0], points_array[i, 1], 
                                       points_array[j, 0], points_array[j, 1]) <= threshold:
                uf.union(i, j)
    
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(points[i])
    return list(groups.values())


def find_active_locations_smart(array_3d_zero, neighbor_distance=3):
    """
    Much smarter approach: only look at locations with activity over time
    and filter out isolated points
    """
    activity_map = array_3d_zero.sum(axis=2)
    active_coords = np.argwhere(activity_map > 0)
    
    if len(active_coords) == 0:
        return []
    
    from scipy.spatial import KDTree
    tree = KDTree(active_coords)
    non_isolated = []
    for i, coord in enumerate(active_coords):
        distances, indices = tree.query(coord, k=min(10, len(active_coords)), distance_upper_bound=neighbor_distance)
        if len(distances[distances < neighbor_distance]) > 1:
            non_isolated.append(tuple(coord))
    
    return non_isolated

@jit(nopython=True)
def create_3d_array_optimized(df_prio_xy, df_input_values, max_row, max_col, max_time):
    """Numba-optimized 3D array creation"""
    array_3d = np.full((max_row, max_col, max_time), np.nan, dtype=np.float32)
    
    for i in range(len(df_prio_xy)):
        pg_id = df_prio_xy[i, 0]
        row = df_prio_xy[i, 1]
        col = df_prio_xy[i, 2]
        values = df_input_values[i]
        
        for time_idx in range(max_time):
            array_3d[row-1, col-1, time_idx] = values[time_idx]
    
    return array_3d

def create_non_overlapping_tasks(array_3d_zero, sub_array, active_locations, 
                                x_step, y_step, x_r_range, y_r_range, z_r_range):
    """
    Create tasks that cover all active regions without overlap
    Returns only the coordinates, not the full arrays
    """
    tasks = []
    
    if not active_locations:
        return tasks
    
    # Convert active locations to numpy array for efficient processing
    active_locs_array = np.array(active_locations)
    
    # Calculate the actual search window size including variations
    actual_window_x = sub_array.shape[0] + max(x_r_range) - min(x_r_range)
    actual_window_y = sub_array.shape[1] + max(y_r_range) - min(y_r_range)
    actual_window_z = sub_array.shape[2] + max(z_r_range) - min(z_r_range)
    
    # Find bounding box of all active locations with some padding
    x_min_bound = max(0, active_locs_array[:, 0].min() - actual_window_x)
    x_max_bound = min(array_3d_zero.shape[0] - sub_array.shape[0] + 1, 
                      active_locs_array[:, 0].max() + actual_window_x)
    y_min_bound = max(0, active_locs_array[:, 1].min() - actual_window_y)
    y_max_bound = min(array_3d_zero.shape[1] - sub_array.shape[1] + 1, 
                      active_locs_array[:, 1].max() + actual_window_y)
    
    # Ensure we have valid bounds even for single-point trajectories
    if x_max_bound <= x_min_bound:
        x_max_bound = x_min_bound + 1
    if y_max_bound <= y_min_bound:
        y_max_bound = y_min_bound + 1
    
    # Create regular grid within bounds
    x_positions = list(range(x_min_bound, x_max_bound, x_step))
    y_positions = list(range(y_min_bound, y_max_bound, y_step))
    z_positions = list(range(0, max(1, array_3d_zero.shape[2] - actual_window_z + 1), 6))
    
    # Pre-compute activity mask for efficiency
    activity_map = array_3d_zero.sum(axis=2) > 0
    
    # Group tasks by z for better cache efficiency
    for z in z_positions:
        z_end = min(z + actual_window_z, array_3d_zero.shape[2])
        z_tasks = []
        
        for x in x_positions:
            x_end = min(x + actual_window_x, array_3d_zero.shape[0])
            for y in y_positions:
                y_end = min(y + actual_window_y, array_3d_zero.shape[1])
                
                # Quick check using pre-computed activity mask
                if np.any(activity_map[x:x_end, y:y_end]):
                    # Detailed check for temporal activity
                    region = array_3d_zero[x:x_end, y:y_end, z:z_end]
                    if (region.sum(axis=2) > 0).sum() >= 2:
                        z_tasks.append((x, y))
        
        # Create chunks of tasks for this z level
        if z_tasks:
            chunk_size = max(1, len(z_tasks) // 8)
            for i in range(0, len(z_tasks), chunk_size):
                chunk = z_tasks[i:i + chunk_size]
                
                # Group by y values for better locality
                y_groups = {}
                for x, y in chunk:
                    if y not in y_groups:
                        y_groups[y] = []
                    y_groups[y].append(x)
                
                # Create one task per y group - only store coordinates!
                for y, x_list in y_groups.items():
                    tasks.append((x_list, y, z))
    
    return tasks

# =============================================================================
# Parallel Matching Worker Function
# =============================================================================

def parallel_matching_worker(args):
    """Worker function for parallel processing of matching"""
    (x_range, y, z, array_3d_zero, sub_array, x_r_range, y_r_range, z_r_range,
     non_zero_indices, coordinates_1, weights_1, mass_w1) = args
    
    results = []
    
    # Remove duplicates from range arrays for efficiency
    x_r_unique = sorted(set(x_r_range))
    y_r_unique = sorted(set(y_r_range))
    z_r_unique = sorted(set(z_r_range))
    
    for x in x_range:
        for x_r in x_r_unique:
            for y_r in y_r_unique:
                for z_r in z_r_unique:
                    # Extract sub-array
                    x_end = x + x_r + sub_array.shape[0]
                    y_end = y + y_r + sub_array.shape[1]
                    z_end = z + z_r + sub_array.shape[2]
                    
                    if x_end > array_3d_zero.shape[0] or y_end > array_3d_zero.shape[1] or z_end > array_3d_zero.shape[2]:
                        continue
                    
                    sub_array_2 = array_3d_zero[x:x_end, y:y_end, z:z_end]
                    
                    # Quick checks - skip if less than 2 non-zero points
                    if (sub_array_2.sum(axis=2)>0).sum() < 2:
                        continue
                    
                    zero_indices_2 = np.argwhere(sub_array_2 != 0)
                    ratio = abs(tanh(np.log(len(zero_indices_2) / len(non_zero_indices))))
                    
                    if ratio > 0.6:
                        continue
                    
                    # Process coordinates
                    bound = np.array([[0, 0, 0], list(sub_array_2.shape)])
                    coordinates_2 = zero_indices_2.astype(np.float32)
                    coordinates_2 = (coordinates_2 - bound.min(axis=0)) / (bound.max(axis=0) - 1 - bound.min(axis=0))
                    coordinates_2 = np.nan_to_num(coordinates_2, nan=0.5)
                    
                    # Process weights
                    weights_2 = sub_array_2[sub_array_2 != 0]
                    if len(zero_indices_2) > 1:
                        weight_range = weights_2.flatten().max(axis=0) - weights_2.flatten().min(axis=0)
                        weights_2 = weights_2 / np.sum(weights_2)
                    else:
                        weights_2 = np.array([1])
                    
                    # Calculate distance
                    d_met = ot.dist(coordinates_1, coordinates_2, metric='euclidean')
                    d_min = ot.emd2(weights_1, weights_2, d_met)
                    best_rota = 0
                    
                    # Check rotations only if needed
                    if len(zero_indices_2) > 1:
                        for i in range(3):
                            sub_array_3 = np.rot90(sub_array_2, k=i+1, axes=(0, 1))
                            non_zero_indices_3 = np.argwhere(sub_array_3 != 0)
                            bound = np.array([[0, 0, 0], list(sub_array_3.shape)])
                            coordinates_3 = non_zero_indices_3.astype(np.float32)
                            coordinates_3 = (coordinates_3 - bound.min(axis=0)) / (bound.max(axis=0) - 1 - bound.min(axis=0))
                            coordinates_3 = np.nan_to_num(coordinates_3, nan=0.5)
                            weights_3 = sub_array_3[sub_array_3 != 0]
                            weight_range = weights_3.flatten().max(axis=0) - weights_3.flatten().min(axis=0)
                            if weight_range > 0:
                                mass_w3 = sum((weights_3.flatten() - weights_3.flatten().min(axis=0)) / weight_range)
                            else:
                                mass_w3 = len(weights_3)  # All weights are equal
                            weights_3 = weights_3 / np.sum(weights_3)
                            d_met = ot.dist(coordinates_1, coordinates_3, metric='euclidean')
                            d_sub = ot.emd2(weights_1, weights_3, d_met)
                            if d_min > d_sub:
                                d_min = d_sub
                                best_rota = i + 1
                    
                    results.append([x, x_end, y, y_end, z, z_end, d_min, best_rota, ratio])
    
    return results

# =============================================================================
# Shared Memory Support for Large Arrays
# =============================================================================

from multiprocessing import shared_memory
import multiprocessing as mp

def create_shared_array(array):
    """Create a shared memory array from numpy array"""
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shared_array[:] = array[:]
    return shm, shared_array

def parallel_matching_worker_shared(args):
    """Worker function that uses shared memory arrays"""
    (x_range, y, z, array_3d_shape, array_3d_dtype, array_3d_shm_name,
     sub_array_shape, sub_array_dtype, sub_array_shm_name,
     x_r_range, y_r_range, z_r_range,
     non_zero_indices, coordinates_1, weights_1, mass_w1) = args
    
    # Reconnect to shared memory
    array_3d_shm = shared_memory.SharedMemory(name=array_3d_shm_name)
    array_3d_zero = np.ndarray(array_3d_shape, dtype=array_3d_dtype, buffer=array_3d_shm.buf)
    
    sub_array_shm = shared_memory.SharedMemory(name=sub_array_shm_name)
    sub_array = np.ndarray(sub_array_shape, dtype=sub_array_dtype, buffer=sub_array_shm.buf)
    
    results = []
    
    # Remove duplicates from range arrays for efficiency
    x_r_unique = sorted(set(x_r_range))
    y_r_unique = sorted(set(y_r_range))
    z_r_unique = sorted(set(z_r_range))
    
    try:
        for x in x_range:
            for x_r in x_r_unique:
                for y_r in y_r_unique:
                    for z_r in z_r_unique:
                        # Extract sub-array
                        x_end = x + x_r + sub_array.shape[0]
                        y_end = y + y_r + sub_array.shape[1]
                        z_end = z + z_r + sub_array.shape[2]
                        
                        if x_end > array_3d_zero.shape[0] or y_end > array_3d_zero.shape[1] or z_end > array_3d_zero.shape[2]:
                            continue
                        
                        sub_array_2 = array_3d_zero[x:x_end, y:y_end, z:z_end].copy()  # Copy to avoid shared memory issues
                        
                        # Quick checks - skip if less than 2 non-zero points
                        if (sub_array_2.sum(axis=2)>0).sum() < 2:
                            continue
                        
                        zero_indices_2 = np.argwhere(sub_array_2 != 0)
                        ratio = abs(tanh(np.log(len(zero_indices_2) / len(non_zero_indices))))
                        
                        if ratio > 0.6:
                            continue
                        
                        # Process coordinates
                        bound = np.array([[0, 0, 0], list(sub_array_2.shape)])
                        coordinates_2 = zero_indices_2.astype(np.float32)
                        coordinates_2 = (coordinates_2 - bound.min(axis=0)) / (bound.max(axis=0) - 1 - bound.min(axis=0))
                        coordinates_2 = np.nan_to_num(coordinates_2, nan=0.5)
                        
                        # Process weights
                        weights_2 = sub_array_2[sub_array_2 != 0]
                        if len(zero_indices_2) > 1:
                            weight_range = weights_2.flatten().max(axis=0) - weights_2.flatten().min(axis=0)
                            weights_2 = weights_2 / np.sum(weights_2)
                        else:
                            weights_2 = np.array([1])
                        
                        # Calculate distance
                        d_met = ot.dist(coordinates_1, coordinates_2, metric='euclidean')
                        d_min = ot.emd2(weights_1, weights_2, d_met)
                        best_rota = 0
                        
                        # Check rotations only if needed
                        if len(zero_indices_2) > 1:
                            for i in range(3):
                                sub_array_3 = np.rot90(sub_array_2, k=i+1, axes=(0, 1))
                                non_zero_indices_3 = np.argwhere(sub_array_3 != 0)
                                bound = np.array([[0, 0, 0], list(sub_array_3.shape)])
                                coordinates_3 = non_zero_indices_3.astype(np.float32)
                                coordinates_3 = (coordinates_3 - bound.min(axis=0)) / (bound.max(axis=0) - 1 - bound.min(axis=0))
                                coordinates_3 = np.nan_to_num(coordinates_3, nan=0.5)
                                weights_3 = sub_array_3[sub_array_3 != 0]
                                weight_range = weights_3.flatten().max(axis=0) - weights_3.flatten().min(axis=0)
                                if weight_range > 0:
                                    mass_w3 = sum((weights_3.flatten() - weights_3.flatten().min(axis=0)) / weight_range)
                                else:
                                    mass_w3 = len(weights_3)
                                weights_3 = weights_3 / np.sum(weights_3)
                                d_met = ot.dist(coordinates_1, coordinates_3, metric='euclidean')
                                d_sub = ot.emd2(weights_1, weights_3, d_met)
                                if d_min > d_sub:
                                    d_min = d_sub
                                    best_rota = i + 1
                        
                        results.append([x, x_end, y, y_end, z, z_end, d_min, best_rota, ratio])
    finally:
        # Clean up shared memory references
        array_3d_shm.close()
        sub_array_shm.close()
    
    return results

# =============================================================================
# Main Execution
# =============================================================================

print("Starting optimized parallel version...")
start_time = datetime.now()

# =============================================================================
# Load and Process Conflict Data
# =============================================================================

print("Loading conflict data...")
df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged251-csv.zip",
                     parse_dates=['date_start','date_end'], low_memory=False)
    
# Load monthly candidates
month = datetime.now().strftime("%m")
if month == '01':
    month = '13'

# Vectorized concatenation
dfs_to_concat = [df]
for i in range(1, int(month)):
    df_can = pd.read_csv(f'https://ucdp.uu.se/downloads/candidateged/GEDEvent_v25_0_{i}.csv')
    df_can.columns = df.columns
    df_can['date_start'] = pd.to_datetime(df_can['date_start'])
    df_can['date_end'] = pd.to_datetime(df_can['date_end'])
    dfs_to_concat.append(df_can)

df = pd.concat(dfs_to_concat, axis=0).drop_duplicates()

# Process conflict data efficiently
print("Processing conflict data...")
unique_grids = df.priogrid_gid.unique()
date_range = pd.date_range(df.date_start.min(), df.date_end.max())

# Use sparse matrix for efficiency
from scipy.sparse import lil_matrix
sparse_data = lil_matrix((len(date_range), len(unique_grids)), dtype=np.float32)

# Create mapping for faster lookup
grid_to_idx = {grid: idx for idx, grid in enumerate(unique_grids)}
date_to_idx = {date: idx for idx, date in enumerate(date_range)}

# Vectorized processing
for grid in unique_grids:
    df_sub = df[df.priogrid_gid == grid]
    # Only process same-month events
    same_month_mask = df_sub.date_start.dt.month == df_sub.date_end.dt.month
    df_sub_filtered = df_sub[same_month_mask]
    
    for _, row in df_sub_filtered.iterrows():
        date_idx = date_to_idx.get(row.date_start.normalize())
        if date_idx is not None:
            sparse_data[date_idx, grid_to_idx[grid]] += row.best

# Convert to dense DataFrame
df_tot = pd.DataFrame(sparse_data.toarray(), index=date_range, columns=unique_grids)

# Resample monthly
df_tot_m = df_tot.resample('M').sum()
last_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
df_input = df_tot_m.loc[:last_month, :]

# Save to CSV
df_input.to_csv('df_prio.csv')
del df  # Free memory
del df_tot

# =============================================================================
# Search for patterns and similar cases
# =============================================================================

print("Loading PRIO grid data...")
df_prio = pd.read_csv('prio.csv', index_col=0, parse_dates=True)
common_cols = df_prio.index.intersection(df_input.columns)
df_prio = df_prio.loc[common_cols]
df_input = df_input.loc[:,common_cols]
df_prio_xy = df_prio.iloc[:, :2]

# Create 3D array efficiently
print("Creating 3D array...")
max_row = df_prio_xy['row'].max()
max_col = df_prio_xy['col'].max()
max_time = len(df_input.index)

# Prepare data for numba function
df_prio_xy_array = np.column_stack([
    df_prio_xy.index.values,
    df_prio_xy['row'].values,
    df_prio_xy['col'].values
])
df_input_values = df_input.values.T

array_3d = create_3d_array_optimized(df_prio_xy_array, df_input_values, 
                                     max_row, max_col, max_time)

# Adjust array dimensions
array_3d = array_3d[df_prio_xy['row'].min()-1:, df_prio_xy['col'].min()-1:, :]
array_3d_na = array_3d.copy()
array_3d_na[array_3d_na == 0] = np.nan
array_3d_zero = np.nan_to_num(array_3d, nan=0)

# Find trajectories
print("Finding trajectories...")
thres_gr = 2
traj_bo = []

# Vectorized trajectory search
last_12_months = array_3d_zero[:, :, -12:]
active_mask = (last_12_months != 0).any(axis=2)
active_coords = np.argwhere(active_mask)

for coord in active_coords:
    traj_bo.append({'x': coord[0], 'y': coord[1]})

grouped_points = group_points(traj_bo, thres_gr)
grouped_points = [i for i in grouped_points if len(i)>1]
sum_t=[]
for t in grouped_points:
    sum_tot=0
    for co in t:
        sum_tot+=array_3d_zero[co['x'],co['y'],-12:].sum()
    sum_t.append(sum_tot)
grouped_points = [j for i,j in enumerate(grouped_points) if sum_t[i]>15]    

    
# Process final trajectories
final_traj = []
for gr_li in grouped_points:
    df_p = pd.DataFrame(gr_li)
    x_coords = df_p['x'].values
    y_coords = df_p['y'].values
    
    if len(np.unique(x_coords)) == 1:
        x_min = x_max = x_coords[0]
    else:
        x_min, x_max = x_coords.min(), x_coords.max()
    
    if len(np.unique(y_coords)) == 1:
        y_min = y_max = y_coords[0]
    else:
        y_min, y_max = y_coords.min(), y_coords.max()
    
    final_traj.append({'xmin': x_min, 'xmax': x_max, 'ymin': y_min, 'ymax': y_max})

# Save trajectories
dict_final_traj = {i: final_traj[i] for i in range(len(final_traj))}
with open('Results/final_traj.pkl', 'wb') as f:
    pickle.dump(dict_final_traj, f)

# =============================================================================
# Matching process with parallel processing
# =============================================================================

print("Starting parallel matching process...")
dict_inp = []
dict_mat = []

for idx, coor in enumerate(final_traj):
    print(f"Processing trajectory {idx+1}/{len(final_traj)}...")
    
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, 
                              coor['ymin']:coor['ymax']+1, -12:]
    dict_inp.append(sub_array)
    
    # Pre-compute values for this sub-array
    bound_1 = np.array([[0, 0, 0], list(sub_array.shape)])
    non_zero_indices = np.argwhere(sub_array != 0)
    coordinates_1 = non_zero_indices.astype(np.float32)
    coordinates_1 = (coordinates_1 - bound_1.min(axis=0)) / (bound_1.max(axis=0) - 1 - bound_1.min(axis=0))
    coordinates_1 = np.nan_to_num(coordinates_1,0.5)
    
    if len(non_zero_indices) > 1:
        weights_1 = sub_array[sub_array != 0]
        weight_range = weights_1.flatten().max(axis=0) - weights_1.flatten().min(axis=0)
        if weight_range > 0:
            mass_w1 = sum((weights_1.flatten() - weights_1.flatten().min(axis=0)) / weight_range)
        else:
            mass_w1 = len(weights_1)  # All weights are equal
        weights_1 = weights_1 / np.sum(weights_1)
    else:
        mass_w1 = 1
        weights_1 = np.array([1])
    
    # Generate parameter ranges - handle edge cases where dimensions are 1
    x_step = max(1, int(sub_array.shape[0] / 2)) if sub_array.shape[0] > 1 else 1
    y_step = max(1, int(sub_array.shape[1] / 2)) if sub_array.shape[1] > 1 else 1
    
    # Ensure we have meaningful search ranges even for small sub-arrays
    x_r_range = [-int(sub_array.shape[0] / 4), 0, int(sub_array.shape[0] / 4)]
    y_r_range = [-int(sub_array.shape[1] / 4), 0, int(sub_array.shape[1] / 4)]
    
    z_r_range = [-3, 0, 3]
    
    # Smart filtering: only look at active, non-isolated locations
    print(f"  Finding active locations for trajectory {idx+1}...")
    
    # Get locations with activity (summed across time)
    activity_map = array_3d_zero.sum(axis=2)
    active_locations = find_active_locations_smart(array_3d_zero, neighbor_distance=3)
    
    print(f"  Found {len(active_locations)} non-isolated active locations out of {(activity_map > 0).sum()} total active locations")
    
    # Create non-overlapping search tasks
    tasks = create_non_overlapping_tasks(
        array_3d_zero, sub_array, active_locations,
        x_step, y_step, x_r_range, y_r_range, z_r_range
    )
    
    print(f"  Created {len(tasks)} non-overlapping parallel tasks")
    
    # Execute in parallel using all available cores
    if tasks:
        
        print(f"  Using batched processing")
        
        # Process in smaller batches to avoid memory issues
        batch_size = max(1, len(tasks) // 4)  # Process in 4 batches
        all_results = []
        
        for batch_start in range(0, len(tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            
            print(f"    Processing batch {batch_start//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
            
            # Create the full arguments for this batch
            full_tasks = []
            for x_list, y, z in batch_tasks:
                full_tasks.append((
                    x_list, y, z, array_3d_zero, sub_array,
                    x_r_range, y_r_range, z_r_range,
                    non_zero_indices, coordinates_1, weights_1, mass_w1
                ))
            
            # Use backend='threading' to share memory instead of copying
            batch_results = Parallel(n_jobs=-1, backend='threading')(
                delayed(parallel_matching_worker)(task) for task in full_tasks
            )
            all_results.extend(batch_results)
            
            # Force garbage collection between batches
            del full_tasks
            del batch_results
            import gc
            gc.collect()
        
        results = all_results
        
        # Flatten results
        dist_arr = []
        for result_batch in results:
            dist_arr.extend(result_batch)
    else:
        dist_arr = []
    
    # Process results
    if dist_arr:
        dist_arr = pd.DataFrame(dist_arr)
        dist_arr['Sum'] = dist_arr[6] + dist_arr[8]
        dist_arr = dist_arr.sort_values(['Sum'])
        dist_arr = dist_arr.iloc[:1000, :]
        dist_arr = filter_overlaps(dist_arr)
    else:
        dist_arr = pd.DataFrame()
    
    dict_mat.append(dist_arr)

# Save results
dict_save_mat = {i: dict_mat[i] for i in range(len(dict_mat))}
with open('Results/matches.pkl', 'wb') as f:
    pickle.dump(dict_save_mat, f)

dict_save_inp = {i: dict_inp[i] for i in range(len(dict_inp))}
with open('Results/input.pkl', 'wb') as f:
    pickle.dump(dict_save_inp, f)

# =============================================================================
# Create the forecasts
# =============================================================================

print("Creating forecasts...")
h = 6
thres_emd, thres_np, thres_clu, mini_mat = [0.15, 0.07, 1100, 11]

df_output_2023 = pd.DataFrame(index=range(h), columns=df_prio.index)
gr_b_tot_2023 = []

for enu, df_match in enumerate(dict_mat):
    if len(df_match[(df_match.iloc[:, 6] < thres_emd) & (df_match.iloc[:, 8] < thres_np)]) > mini_mat:
        df_match = df_match[(df_match.iloc[:, 6] < thres_emd) & (df_match.iloc[:, 8] < thres_np)]
    else:
        df_match = df_match.iloc[:mini_mat, :]
    
    source = dict_inp[enu]
    source_coor = np.meshgrid(np.arange(source.shape[0]), np.arange(source.shape[1]), 
                              np.arange(h), indexing='ij')
    source_np = np.column_stack((source_coor[0].ravel(), source_coor[1].ravel(), 
                                 source_coor[2].ravel()))
    
    l_mat = []
    for i in range(len(df_match)):
        sub_a = array_3d_zero[df_match.iloc[i, 0]:df_match.iloc[i, 1],
                              df_match.iloc[i, 2]:df_match.iloc[i, 3],
                              df_match.iloc[i, 5]:df_match.iloc[i, 5] + h]
        sub_a = np.rot90(sub_a, k=df_match.iloc[i, 7], axes=(0, 1))
        matc = array_3d_zero[df_match.iloc[i, 0]:df_match.iloc[i, 1],
                             df_match.iloc[i, 2]:df_match.iloc[i, 3],
                             df_match.iloc[i, 4]:df_match.iloc[i, 5]]
        sub_a = (sub_a - matc.min()) / (matc.max() - matc.min()) if matc.max() > matc.min() else sub_a * 0
        x_coords, y_coords, z_coords = np.meshgrid(np.arange(sub_a.shape[0]), 
                                                    np.arange(sub_a.shape[1]), 
                                                    np.arange(sub_a.shape[2]), indexing='ij')
        x_norm = x_coords.ravel() / (x_coords.max())
        y_norm = y_coords.ravel() / (y_coords.max())
        z_norm = z_coords.ravel() / (z_coords.max())
        sub_a = sub_a.reshape(-1, 1)
        sub_a = np.column_stack((x_norm, y_norm, z_norm, sub_a.ravel()))
        sub_a = np.nan_to_num(sub_a,0)
        sub_a[:, :3] *= tuple(np.array(list(source.shape[:2]) + [h]) - 1)
        
        # Handle size mismatches
        if len(sub_a) < source.shape[0] * source.shape[1] * h:
            sub_a[:, :3] = np.round(sub_a[:, :3]).astype(int)
            for row in source_np:
                if not np.any(np.all(sub_a[:, :3] == row, axis=1)):
                    new_row = np.append(row, 0)
                    sub_a = np.vstack([sub_a, new_row])
            reordered_sub_a = np.zeros_like(sub_a)
            for ki, row in enumerate(source_np):
                index = np.where(np.all(sub_a[:, :3] == row, axis=1))[0][0]
                reordered_sub_a[ki] = sub_a[index]
            sub_a = reordered_sub_a.copy()
            
            if len(sub_a) > source.shape[0] * source.shape[1] * h:
                sub_a[:, :3] = np.round(sub_a[:, :3]).astype(int)
                unique_rows, indices, inverse_indices = np.unique(sub_a[:, :3], axis=0, 
                                                                  return_index=True, 
                                                                  return_inverse=True)
                means_sub = np.zeros((unique_rows.shape[0], sub_a.shape[1]))
                means_sub[:, :3] = unique_rows
                means_sub[:, 3] = np.bincount(inverse_indices, weights=sub_a[:, 3])
                sub_a = means_sub.copy()
                
        elif len(sub_a) > source.shape[0] * source.shape[1] * h:
            sub_a[:, :3] = np.round(sub_a[:, :3]).astype(int)
            unique_rows, indices, inverse_indices = np.unique(sub_a[:, :3], axis=0, 
                                                              return_index=True, 
                                                              return_inverse=True)
            means_sub = np.zeros((unique_rows.shape[0], sub_a.shape[1]))
            means_sub[:, :3] = unique_rows
            means_sub[:, 3] = np.bincount(inverse_indices, weights=sub_a[:, 3])
            sub_a = means_sub.copy()
            
            if len(sub_a) < source.shape[0] * source.shape[1] * h:
                for row in source_np:
                    if not np.any(np.all(sub_a[:, :3] == row, axis=1)):
                        new_row = np.append(row, 0)
                        sub_a = np.vstack([sub_a, new_row])
                reordered_sub_a = np.zeros_like(sub_a)
                for ki, row in enumerate(source_np):
                    index = np.where(np.all(sub_a[:, :3] == row, axis=1))[0][0]
                    reordered_sub_a[ki] = sub_a[index]
                sub_a = reordered_sub_a.copy()
        l_mat.append(sub_a)
    
    # Clustering and averaging
    cluster_labels = cluster_based_on_threshold_dtw(l_mat, (source.shape[0] * source.shape[1] * h / thres_clu))
    l_mat_sub = [l_mat[i] for i in range(len(cluster_labels)) 
                 if pd.Series(cluster_labels).value_counts().loc[cluster_labels[i]] == 
                 pd.Series(cluster_labels).value_counts().max()]
    last_sub_val = np.array([arr[:, 3] for arr in l_mat_sub])
    last_sub_val = last_sub_val.mean(axis=0)
    last_sub_val = last_sub_val * (source.max() - source.min()) + source.min()
    df_g_mean = np.column_stack((source_np, last_sub_val))
    df_g_mean[:, 0] = df_g_mean[:, 0] + final_traj[enu]['xmin']
    df_g_mean[:, 1] = df_g_mean[:, 1] + final_traj[enu]['ymin']
    df_g_mean = pd.DataFrame(df_g_mean)
    
    gr_b = []
    for i in range(len(df_g_mean)):
        try:
            grid_mask = (df_prio.iloc[:, 0] == int(df_g_mean.iloc[i, 0]) + df_prio_xy['row'].min()-1) & \
                       (df_prio.iloc[:, 1] == int(df_g_mean.iloc[i, 1]) + df_prio_xy['col'].min()-1)
            if grid_mask.any():
                grid_id = df_prio[grid_mask].index[0]
                df_output_2023.loc[int(df_g_mean.iloc[i, 2]), grid_id] = df_g_mean.iloc[i, 3]
                gr_b.append(grid_id)
        except:
            pass  # if prio grid does not exist
    
    gr_b_tot_2023.append(pd.Series(gr_b).unique().tolist())

# Save final output
df_output_2023.to_csv('df_output_2023.csv')
with open('Results/gr_b_tot_2023.pkl', 'wb') as f:
    pickle.dump(gr_b_tot_2023, f)

end_time = datetime.now()
print(f"\nTotal execution time: {end_time - start_time}")
print("Optimization complete!")