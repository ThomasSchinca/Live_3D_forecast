# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 16:13:10 2025

@author: thoma
"""

import pandas as pd
from datetime import datetime
import numpy as np
import pickle
from numba import jit
import ot
from tslearn.metrics import dtw
import warnings
import os
from math import tanh
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

# =============================================================================
# DTW Clustering Function
# =============================================================================

def cluster_based_on_threshold_dtw(array_list, threshold):
    """
    Compute a pairwise DTW distance matrix for a list of 4D trajectories using tslearn.
    """
    n = len(array_list)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist = dtw(array_list[i], array_list[j])  # tslearn handles multi-dimensional DTW
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
                            weights_3 = weights_3 / np.sum(weights_3)
                            d_met = ot.dist(coordinates_1, coordinates_3, metric='euclidean')
                            d_sub = ot.emd2(weights_1, weights_3, d_met)
                            if d_min > d_sub:
                                d_min = d_sub
                                best_rota = i + 1
                    
                    results.append([x, x_end, y, y_end, z, z_end, d_min, best_rota, ratio])
    
    return results

# =============================================================================
# Load/Save Progress Functions
# =============================================================================

def save_progress(idx, dict_inp, dict_mat):
    """Save current progress"""
    progress_data = {
        'current_idx': idx,
        'dict_inp': {i: dict_inp[i] for i in range(len(dict_inp))},
        'dict_mat': {i: dict_mat[i] for i in range(len(dict_mat))},
        'timestamp': datetime.now().isoformat()
    }
    with open('Results/progress.pkl', 'wb') as f:
        pickle.dump(progress_data, f)

def load_progress():
    """Load previous progress if exists"""
    try:
        with open('Results/progress.pkl', 'rb') as f:
            progress_data = pickle.load(f)
        return progress_data['current_idx'], progress_data['dict_inp'], progress_data['dict_mat']
    except FileNotFoundError:
        return 0, [], []

# =============================================================================
# Main Execution
# =============================================================================

def main():
    start_time = datetime.now()

    # Load intermediate data from part 1
    try:
        with open('Results/final_traj.pkl', 'rb') as f:
            dict_final_traj = pickle.load(f)
        final_traj = [dict_final_traj[i] for i in sorted(dict_final_traj.keys())]
        
        array_3d_zero = np.load('Results/array_3d_zero.npy')
        
        with open('Results/df_prio_xy.pkl', 'rb') as f:
            df_prio_xy = pickle.load(f)
            
        df_prio = pd.read_csv('prio.csv', index_col=0, parse_dates=True)
        
    except FileNotFoundError as e:
        raise Exception(f"Required data from Part 1 not found: {e}")

    # Check for existing progress
    start_idx, dict_inp, dict_mat = load_progress()
    dict_inp = [dict_inp[i] for i in dict_inp.keys()]
    dict_mat = [dict_mat[i] for i in dict_mat.keys()]
    
    # =============================================================================
    # Matching process with parallel processing and resume capability
    # =============================================================================

    for idx in range(start_idx, len(final_traj)):
        coor = final_traj[idx]
        
        sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, 
                                  coor['ymin']:coor['ymax']+1, -12:]
        
        # Only append if we're processing new trajectories
        if idx >= len(dict_inp):
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
        active_locations = find_active_locations_smart(array_3d_zero, neighbor_distance=3)
        
        # Create non-overlapping search tasks
        tasks = create_non_overlapping_tasks(
            array_3d_zero, sub_array, active_locations,
            x_step, y_step, x_r_range, y_r_range, z_r_range
        )
        
        # Execute in parallel using all available cores
        if tasks:
            # Process in smaller batches to avoid memory issues
            batch_size = max(1, len(tasks) // 4)  # Process in 4 batches
            all_results = []
            
            for batch_start in range(0, len(tasks), batch_size):
                batch_end = min(batch_start + batch_size, len(tasks))
                batch_tasks = tasks[batch_start:batch_end]
                
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
        
        # Only append if we're processing new trajectories
        if idx >= len(dict_mat):
            dict_mat.append(dist_arr)
        
        # Save progress every 5 trajectories
        if (idx + 1) % 5 == 0:
            save_progress(idx + 1, dict_inp, dict_mat)

    # Save final results
    dict_save_mat = {i: dict_mat[i] for i in range(len(dict_mat))}
    with open('Results/matches.pkl', 'wb') as f:
        pickle.dump(dict_save_mat, f)

    dict_save_inp = {i: dict_inp[i] for i in range(len(dict_inp))}
    with open('Results/input.pkl', 'wb') as f:
        pickle.dump(dict_save_inp, f)

    # =============================================================================
    # Create the forecasts
    # =============================================================================

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
    df_output_2023.to_csv('df_output.csv')
    with open('Results/gr_b_tot.pkl', 'wb') as f:
        pickle.dump(gr_b_tot_2023, f)

    # Clean up progress file
    if os.path.exists('Results/progress.pkl'):
        os.remove('Results/progress.pkl')

    end_time = datetime.now()
    
    # Log completion
    with open('Results/part2_log.txt', 'w') as f:
        f.write(f"Part 2 completed at: {end_time}\n")
        f.write(f"Execution time: {end_time - start_time}\n")
        f.write(f"Total forecasts created: {len(gr_b_tot_2023)}\n")

if __name__ == "__main__":
    main()