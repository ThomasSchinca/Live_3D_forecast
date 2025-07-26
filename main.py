# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:39:53 2025

@author: thoma
"""

import pandas as pd
from datetime import datetime,timedelta
import numpy as np
import math
import pickle
import ot
from math import tanh
from scipy.spatial.distance import pdist, squareform

# =============================================================================
# Functions
# =============================================================================

def cluster_based_on_threshold(array_list, threshold):
    last_column = np.array([arr[:, 3] for arr in array_list])
    distances = squareform(pdist(last_column, metric='euclidean'))
    cluster_labels = np.full(len(array_list), -1)
    current_cluster = 0
    for i in range(len(array_list)):
        if cluster_labels[i] == -1:
            cluster_labels[i] = current_cluster
            for j in range(i + 1, len(array_list)):
                if distances[i, j] <= threshold:
                    cluster_labels[j] = current_cluster
            current_cluster += 1
    return cluster_labels

def compute_overlap(row1, row2):
    overlap = 1
    union = 1
    for i in range(3):  # There are 3 dimensions (x, y, z)
        low1, high1 = row1[2*i], row1[2*i+1]
        low2, high2 = row2[2*i], row2[2*i+1]
        intersect_low = max(low1, low2)
        intersect_high = min(high1, high2)
        union_low = min(low1, low2)
        union_high = max(high1, high2)

        if intersect_high > intersect_low:
            overlap *= (intersect_high - intersect_low)
        else:
            overlap = 0
            break
        union *= (union_high - union_low)
    return overlap / union if union > 0 else 0

def filter_overlaps(df):
    while True:
        to_drop = set()
        for i, row1 in df.iterrows():
            for j, row2 in df.iterrows():
                if i < j:
                    overlap_ratio = compute_overlap(row1[:6], row2[:6])
                    if overlap_ratio > 0.25:
                        to_drop.add(i if row1[6] > row2[6] else j)
        if not to_drop:
            break
        df = df.drop(index=to_drop)
    return df

def euclidean_distance(p1, p2):
    return math.sqrt((p2['x'] - p1['x']) ** 2 + (p2['y'] - p1['y']) ** 2)

class UnionFind:
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
    n = len(points)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if euclidean_distance(points[i], points[j]) <= threshold:
                uf.union(i, j)
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(points[i])
    return list(groups.values())


# =============================================================================
# Extract the 3d data
# =============================================================================

df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged251-csv.zip",
                  parse_dates=['date_start','date_end'],low_memory=False)
month = datetime.now().strftime("%m")

if month=='01':
   month='13'
for i in range(1,int(month)):
    df_can = pd.read_csv(f'https://ucdp.uu.se/downloads/candidateged/GEDEvent_v25_0_{i}.csv')
    df_can.columns = df.columns
    df_can['date_start'] = pd.to_datetime(df_can['date_start'])
    df_can['date_end'] = pd.to_datetime(df_can['date_end'])
    df_can = df_can.drop_duplicates()
    df= pd.concat([df,df_can],axis=0)

df_tot = pd.DataFrame(columns=df.priogrid_gid.unique(),index=pd.date_range(df.date_start.min(),
                                          df.date_end.max()))
df_tot=df_tot.fillna(0)
for i in df.priogrid_gid.unique():
    df_sub=df[df.priogrid_gid==i]
    for j in range(len(df_sub)):
        if df_sub.date_start.iloc[j].month == df_sub.date_end.iloc[j].month:
            df_tot.loc[df_sub.date_start.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j],i]+df_sub.best.iloc[j]
        else:
            pass                                                    
                                                     
df_tot_m=df_tot.resample('M').sum()
last_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
df_input= df_tot_m.loc[:last_month,:]
df_input.to_csv('df_prio.csv')
del df
del df_tot


# =============================================================================
# Search for patterns and similar cases 
# =============================================================================


df_prio = pd.read_csv('prio.csv',index_col=0,parse_dates=True)
common_cols = df_prio.index.intersection(df_input.columns)
df_prio = df_prio.loc[common_cols]
df_input = df_input.loc[:,common_cols]
df_prio_xy = df_prio.iloc[:,:2]
max_row = df_prio_xy['row'].max()
max_col = df_prio_xy['col'].max()
max_time = len(df_input.index)
array_3d = np.full((max_row, max_col, max_time), np.nan)
for pg_id, (row, col) in df_prio_xy.iterrows():
    input_column = df_input[pg_id]
    for time_index, value in enumerate(input_column):
        array_3d[row-1, col-1, time_index] = value

array_3d=array_3d[df_prio_xy['row'].min():,df_prio_xy['col'].min():,:]
array_3d_na = array_3d.copy()
array_3d_na[array_3d_na==0]=np.nan
array_3d_zero = np.nan_to_num(array_3d, nan=0)

thres_gr = 2

traj_bo=[]
for x_b in range(0,array_3d_zero.shape[0]):
    for y_b in range(0,array_3d_zero.shape[1]):
        if (array_3d_zero[x_b,y_b,-12:]==0).all()==False :
            traj_bo.append({'x': x_b,'y': y_b})
grouped_points = group_points(traj_bo,thres_gr)
grouped_points = [i for i in grouped_points if len(i)>1]

final_traj=[]
for gr_li in grouped_points:
    df_p = pd.DataFrame(gr_li)
    if len(df_p.iloc[:,0].unique())==1:
        x_min = df_p.iloc[:,0].min()
        x_max = df_p.iloc[:,0].max()
    else:
        x_min = df_p.iloc[:,0].min()
        x_max = df_p.iloc[:,0].max()
    
    if len(df_p.iloc[:,1].unique())==1:
        y_min = df_p.iloc[:,1].min()
        y_max = df_p.iloc[:,1].max()
    else:
        y_min = df_p.iloc[:,1].min()
        y_max = df_p.iloc[:,1].max()
    final_traj.append({'xmin':x_min,'xmax':x_max,'ymin': y_min,'ymax':y_max})

dict_final_traj = {i:final_traj[i] for i in range(len(final_traj))}    
with open('Results/final_traj.pkl', 'wb') as f:
    pickle.dump(dict_final_traj, f)

# =============================================================================
# Matching process
# =============================================================================
      
dict_inp= []
dict_mat= []
for coor in final_traj:
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-12:]
    dict_inp.append(sub_array)
    bound_1=np.array([[0,0,0],list(sub_array.shape)])
    non_zero_indices = np.argwhere(sub_array!=0)
    coordinates_1 = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates_1 = np.array(coordinates_1)
    coordinates_1 = (coordinates_1 - bound_1.min(axis=0)) / (bound_1.max(axis=0)-1 - bound_1.min(axis=0))
    if len(non_zero_indices)>1:
        weights_1 = sub_array[sub_array != 0]
        mass_w1 = sum((weights_1.flatten() - weights_1.flatten().min(axis=0)) / (weights_1.flatten().max(axis=0) - weights_1.flatten().min(axis=0)))
        weights_1 = weights_1 / np.sum(weights_1)
    else:
        mass_w1=1
        weights_1=np.array([1])
    dist_arr=[]
    for x in range(0,array_3d_zero.shape[0]-2,int(sub_array.shape[0]/2)):
        for y in range(0,array_3d_zero.shape[1]-2,int(sub_array.shape[1]/2)):
            for z in range(0,array_3d_zero.shape[2]-36,6):
                for x_r in [-int(sub_array.shape[0]/4),0,int(sub_array.shape[0]/4)]:
                    for y_r in [-int(sub_array.shape[1]/4),0,int(sub_array.shape[1]/4)]:
                        for z_r in [-3,0,3]:
                            sub_array_2 = array_3d_zero[x:x+x_r+sub_array.shape[0],y:y+y_r+sub_array.shape[1],z:z+z_r+sub_array.shape[2]]
                            bound=np.array([[0,0,0],list(sub_array_2.shape)])
                            zero_indices_2 = np.argwhere(sub_array_2!=0)
                            if ((sub_array_2.flatten()==0).all()==False) & (abs(tanh(np.log(len(zero_indices_2)/len(non_zero_indices)))) <= 0.6):
                                coordinates_2 = [(idx[0],idx[1],idx[2]) for idx in zero_indices_2]
                                coordinates_2 = np.array(coordinates_2)
                                coordinates_2 = (coordinates_2 - bound.min(axis=0)) / (bound.max(axis=0)-1 - bound.min(axis=0))
                                coordinates_2 = np.nan_to_num(coordinates_2, nan=0.5)
                                if len(zero_indices_2)>1:
                                    weights_2 = sub_array_2[sub_array_2!=0]
                                    mass_w2 = sum((weights_2.flatten() - weights_2.flatten().min(axis=0)) / (weights_2.flatten().max(axis=0) - weights_2.flatten().min(axis=0)))
                                    weights_2 = weights_2 / np.sum(weights_2)
                                else:
                                    mass_w2=1
                                    weights_2=np.array([1])
                                d_met=ot.dist(coordinates_1,coordinates_2,metric='euclidean')
                                d_min=ot.emd2(weights_1,weights_2,d_met)
                                best_rota=0
                                if len(zero_indices_2)>1:
                                    for i in range(3):
                                        sub_array_3 = np.rot90(sub_array_2, k=i+1, axes=(0, 1))
                                        non_zero_indices_3 = np.argwhere(sub_array_3 != 0)
                                        bound=np.array([[0,0,0],list(sub_array_3.shape)])
                                        coordinates_3 = [(idx[0], idx[1], idx[2]) for idx in non_zero_indices_3]
                                        coordinates_3 = np.array(coordinates_3)
                                        coordinates_3 = (coordinates_3 - bound.min(axis=0)) / (bound.max(axis=0)-1 - bound.min(axis=0))
                                        coordinates_3 = np.nan_to_num(coordinates_3, nan=0.5)
                                        weights_3 = sub_array_3[sub_array_3!=0]
                                        mass_w3 = sum((weights_3.flatten() - weights_3.flatten().min(axis=0)) / (weights_3.flatten().max(axis=0) - weights_3.flatten().min(axis=0)))
                                        weights_3 = weights_3 / np.sum(weights_3)
                                        d_met=ot.dist(coordinates_1,coordinates_3,metric='euclidean')
                                        d_sub=ot.emd2(weights_1,weights_3,d_met) 
                                        if d_min>d_sub:
                                            d_min=d_sub
                                            best_rota=i+1
                                else:
                                    best_rota=0
                                dist_arr.append([x,x+x_r+sub_array.shape[0],y,y+y_r+sub_array.shape[1],z,z+z_r+sub_array.shape[2],d_min,best_rota,abs(tanh(np.log(len(zero_indices_2)/len(non_zero_indices))))])
                            else:
                                pass
    dist_arr=pd.DataFrame(dist_arr)
    dist_arr['Sum'] = dist_arr[6] + dist_arr[8]
    dist_arr=dist_arr.sort_values(['Sum'])
    dist_arr = dist_arr.iloc[:500,:]
    dist_arr = filter_overlaps(dist_arr)
    dict_mat.append(dist_arr)

dict_save_mat = {i:dict_mat[i] for i in range(len(dict_mat))}    
with open('Results/matches.pkl', 'wb') as f:
    pickle.dump(dict_save_mat, f)
dict_save_inp = {i:dict_inp[i] for i in range(len(dict_inp))}    
with open('Results/input.pkl', 'wb') as f:
    pickle.dump(dict_save_inp, f)
    
    
# =============================================================================
# Create the forecasts    
# =============================================================================

h=6
final_traj = [dict_final_traj[key] for key in dict_final_traj.keys()]
thres_emd, thres_np, thres_clu, mini_mat = [0.15,0.07,1100,11]

df_output_2023 = pd.DataFrame(index=range(h),columns=df_prio.index)
gr_b_tot_2023=[]
for enu,df_match in enumerate(dict_mat):
    if len(df_match[(df_match.iloc[:,6]<thres_emd) & (df_match.iloc[:,8]<thres_np)])>mini_mat:
        df_match= df_match[(df_match.iloc[:,6]<thres_emd) & (df_match.iloc[:,8]<thres_np)]
    else:
        df_match= df_match.iloc[:mini_mat,:]
    source = dict_inp[enu]
    source_coor = np.meshgrid(np.arange(source.shape[0]), np.arange(source.shape[1]), np.arange(h), indexing='ij')
    source_np= np.column_stack((source_coor[0].ravel(), source_coor[1].ravel(), source_coor[2].ravel()))
    l_mat=[]
    for i in range(len(df_match)):
        sub_a = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,5]:df_match.iloc[i,5]+h]
        sub_a = np.rot90(sub_a, k=df_match.iloc[i,7], axes=(0, 1))
        matc = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,4]:df_match.iloc[i,5]]
        sub_a = (sub_a - matc.min()) / (matc.max()-matc.min())
        x_coords, y_coords, z_coords = np.meshgrid(np.arange(sub_a.shape[0]), np.arange(sub_a.shape[1]), np.arange(sub_a.shape[2]), indexing='ij')
        x_norm = x_coords.ravel() / (x_coords.max())
        y_norm = y_coords.ravel() / (y_coords.max())
        z_norm = z_coords.ravel() / (z_coords.max())
        sub_a = sub_a.reshape(-1, 1)
        sub_a = np.column_stack((x_norm, y_norm, z_norm, sub_a.ravel()))
        sub_a[:, :3] *= tuple(np.array(list(source.shape[:2])+[h]) - 1)
        if len(sub_a) < source.shape[0]*source.shape[1]*h:
            sub_a[:,:3]=np.round(sub_a[:, :3]).astype(int)
            for row in source_np:
                if not np.any(np.all(sub_a[:, :3] == row, axis=1)):
                    new_row = np.append(row, 0)
                    sub_a = np.vstack([sub_a, new_row])
            reordered_sub_a = np.zeros_like(sub_a)
            for ki, row in enumerate(source_np):
                index = np.where(np.all(sub_a[:, :3] == row, axis=1))[0][0]
                reordered_sub_a[ki] = sub_a[index]
            sub_a = reordered_sub_a.copy()
            if len(sub_a) > source.shape[0]*source.shape[1]*h:
                sub_a[:,:3]=np.round(sub_a[:, :3]).astype(int)
                unique_rows, indices, inverse_indices = np.unique(sub_a[:, :3], axis=0, return_index=True, return_inverse=True)
                means_sub = np.zeros((unique_rows.shape[0], sub_a.shape[1]))
                means_sub[:, :3] = unique_rows
                means_sub[:, 3] = np.bincount(inverse_indices, weights=sub_a[:, 3])
                sub_a = means_sub.copy()
        elif len(sub_a) > source.shape[0]*source.shape[1]*h:  
            sub_a[:,:3]=np.round(sub_a[:, :3]).astype(int)
            unique_rows, indices, inverse_indices = np.unique(sub_a[:, :3], axis=0, return_index=True, return_inverse=True)
            means_sub = np.zeros((unique_rows.shape[0], sub_a.shape[1]))
            means_sub[:, :3] = unique_rows
            means_sub[:, 3] = np.bincount(inverse_indices, weights=sub_a[:, 3])
            sub_a = means_sub.copy()
            if len(sub_a) < source.shape[0]*source.shape[1]*h:
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
        
    cluster_labels = cluster_based_on_threshold(l_mat, (source.shape[0]*source.shape[1]*h/thres_clu))
    l_mat_sub = [l_mat[i] for i in range(len(cluster_labels)) if pd.Series(cluster_labels).value_counts().loc[cluster_labels[i]] == pd.Series(cluster_labels).value_counts().max()]
    last_sub_val = np.array([arr[:, 3] for arr in l_mat_sub])
    last_sub_val = last_sub_val.mean(axis=0)
    last_sub_val = last_sub_val*(source.max()-source.min()) + source.min()
    df_g_mean = np.column_stack((source_np,last_sub_val))
    df_g_mean[:,0] = df_g_mean[:,0]+final_traj[enu]['xmin']
    df_g_mean[:,1] = df_g_mean[:,1]+final_traj[enu]['ymin']
    df_g_mean=pd.DataFrame(df_g_mean)
    gr_b=[]
    for i in range(len(df_g_mean)):
        try:
            df_output_2023.loc[int(df_g_mean.iloc[i,2]),df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index]=df_g_mean.iloc[i,3]
            gr_b.append(df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index[0])
        except:
            pass #if prio grid does not exist 
    gr_b_tot_2023.append(pd.Series(gr_b).unique().tolist())






