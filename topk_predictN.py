import heapq
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import time
import scipy
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def topk_predict(model, num_entities, block_size, topk, train_data_array, epsilon, stop=True):
    DEVICE = train_data_array.device
    
    with torch.no_grad():
        embeddings = model.embeddings() 

    n = len(num_entities)
    sorted_indices_list = []
    norm_list = []
    norm_arrays = []
    
    for d in range(n):
        sorted_idx = torch.argsort(embeddings[d], dim=-1, descending=True)
        emb = embeddings[d]
        
        sorted_indices_list.append(sorted_idx)
        norm_list.append(emb)

        num_blocks = math.ceil(sorted_idx.shape[0] / block_size)
        
        pad_size = num_blocks * block_size - sorted_idx.shape[0]
        if pad_size > 0:
            sorted_vals = emb[sorted_idx]
            padded_vals = torch.nn.functional.pad(sorted_vals, (0, pad_size), value=0.0)
            reshaped_vals = padded_vals.view(num_blocks, block_size)
        else:
            reshaped_vals = emb[sorted_idx].view(num_blocks, block_size)
            
        norm_arrays.append(reshaped_vals.max(dim=1).values)

    block_ranges = [torch.arange(0, a.shape[0], device=DEVICE) for a in norm_arrays]
    prod_tuple = torch.cartesian_prod(*block_ranges)
    
    tuples_norm_array = torch.ones(prod_tuple.shape[0], device=DEVICE)
    for d in range(n):
        tuples_norm_array *= norm_arrays[d][prod_tuple[:, d]]

    sorted_tuples_norm_array, sorted_indices = torch.sort(tuples_norm_array, descending=True)
    sorted_tuples = prod_tuple[sorted_indices]

    current_min_score = -float('inf')
    
    global_top_scores = torch.full((topk,), -float('inf'), device=DEVICE)
    global_top_indices = torch.zeros((topk, n), dtype=torch.long, device=DEVICE)
    
    res_div_max = torch.zeros((sorted_tuples.shape[0],), device=DEVICE)
    
    num_exam = 0
    start_time = time.time()

    MAX_ANGLE_VAL = 1.0 

    for i in tqdm(range(sorted_tuples.shape[0])):
        max_possible_block_score = sorted_tuples_norm_array[i] * MAX_ANGLE_VAL
        
        if stop and (i > 0):
            current_max_angle = res_div_max[:i].max().item() if i > 0 else MAX_ANGLE_VAL
            bound_score = sorted_tuples_norm_array[i] * (epsilon ** i) * current_max_angle
            if bound_score < current_min_score:
                break
        else:
            if max_possible_block_score < current_min_score:
                break

        block_indices = sorted_tuples[i]
        candidate_indices_list = []
        
        for d in range(n):
            b_idx = block_indices[d].item()
            start = b_idx * block_size
            end = min(start + block_size, sorted_indices_list[d].shape[0])
            candidate_indices_list.append(sorted_indices_list[d][start:end])

        testa = torch.cartesian_prod(*candidate_indices_list)
        
        prod = torch.ones(testa.shape[0], device=DEVICE)
        for d in range(n):
            prod *= embeddings[d][testa[:, d]]

        potential_mask = (prod * MAX_ANGLE_VAL) > current_min_score
        
        if not potential_mask.any():
            continue
            
        testa_pruned = testa[potential_mask]
        prod_pruned = prod[potential_mask]
        
        idxs = [testa_pruned[:, d] for d in range(n)]
        angle_i = model.angle(*idxs)
        
        current_block_max_angle = angle_i.max().item()
        res_div_max[i] = current_block_max_angle
        
        res = prod_pruned * angle_i
        
        better_mask = res > current_min_score
        if not better_mask.any():
            continue

        valid_res = res[better_mask]
        valid_indices = testa_pruned[better_mask]
        mask_dup = torch.ones(valid_indices.shape[0], dtype=torch.bool, device=DEVICE)
        
        combined_check = torch.cat((train_data_array, valid_indices), dim=0)

        sub_mask = torch.ones(train_data_array.shape[0], dtype=torch.bool, device=DEVICE)
        for d in range(n):
             min_v = valid_indices[:, d].min()
             max_v = valid_indices[:, d].max()
             sub_mask &= (train_data_array[:, d] >= min_v) & (train_data_array[:, d] <= max_v)
        
        relevant_train = train_data_array[sub_mask]
        
        if relevant_train.shape[0] > 0:

            check_cat = torch.cat((relevant_train, valid_indices), dim=0)
            _, inv, counts = check_cat.unique(dim=0, return_inverse=True, return_counts=True)
            is_dup = counts[inv] > 1
            valid_is_dup = is_dup[relevant_train.shape[0]:]
            
            valid_res = valid_res[~valid_is_dup]
            valid_indices = valid_indices[~valid_is_dup]
        
        if valid_res.shape[0] == 0:
            continue

        num_exam += valid_indices.shape[0]

        cat_scores = torch.cat((global_top_scores, valid_res))
        cat_indices = torch.cat((global_top_indices, valid_indices))
        
        sorted_score, sorted_idx_k = torch.topk(cat_scores, k=topk)
        
        global_top_scores = sorted_score
        global_top_indices = cat_indices[sorted_idx_k]
        
        current_min_score = global_top_scores[-1].item()

    final_res = torch.cat((global_top_scores.unsqueeze(1), global_top_indices), dim=1)
    queue = tuple(map(tuple, final_res.detach().cpu().numpy()))
    queue = list(queue) 
    import heapq
    heapq.heapify(queue)

    print("Max res_div_max:", res_div_max.max().item())
    print("Lastscore:", queue[0])
    
    return queue, num_exam

def precision_recall2(queue, answer, k):
    res_indices = torch.FloatTensor(queue).to(DEVICE)

    _, indices = torch.sort(res_indices[:, 0], descending=True)
    res_indices = res_indices[indices, 1:].long()

    _, idx, counts = torch.cat((answer, res_indices), dim=0).unique(
        dim=0, return_inverse=True, return_counts=True
    )

    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    mask2 = mask[len(answer):]
    precision = mask2[:k].sum().item() / k
    recall = mask2.sum().item() / answer.shape[0]
    return precision, recall


def apk_function2(queue, answer, k):
    res_indices = torch.FloatTensor(queue).to(DEVICE)
    _, indices = torch.sort(res_indices[:, 0], descending=True)
    res_indices = res_indices[indices, 1:].long()

    _, idx, counts = torch.cat((answer, res_indices), dim=0).unique(
        dim=0, return_inverse=True, return_counts=True
    )

    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    mask2 = mask[len(answer):]

    hit = 0
    apk = 0
    for i in range(k):
        if mask2[i]:
            hit += 1
            apk += hit / (i + 1)

    if hit == 0:
        return 0.0
    return apk / min(k, answer.shape[0])