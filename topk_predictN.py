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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def topk_predict(model, num_entities, block_size, topk, train_data_array, alpha, stop=True):
    num_exam = 0
    with torch.no_grad():
        embeddings = model.embeddings()  

    n = len(num_entities)
    indices_list = [torch.arange(0, num_entities[d]).to(DEVICE) for d in range(n)]
    sorted_indices_list = [torch.argsort(embeddings[d], dim=-1, descending=True) for d in range(n)]
    
    norm_list = [embeddings[d] for d in range(n)]  # 각 임베딩의 값 (또는 norm 값)

    num_indices = block_size
    blocks_list = [math.ceil(sorted_indices_list[d].shape[0] / num_indices) for d in range(n)]  
    
    norm_arrays = []
    for d in range(n):
        sorted_idx = sorted_indices_list[d]
        emb = norm_list[d]
        num_blocks = blocks_list[d]
        norm_arr = torch.zeros((num_blocks,))
        for i in range(num_blocks):
            start_idx = i * num_indices
            end_idx = (i + 1) * num_indices if i != num_blocks - 1 else sorted_idx.shape[0]
            block_indices = sorted_idx[start_idx:end_idx]
            max_norm = torch.max(emb[block_indices])
            norm_arr[i] = max_norm.item()
        norm_arrays.append(norm_arr)

    block_ranges = [torch.arange(0, norm_arrays[d].shape[0]) for d in range(n)]
    prod_tuple = torch.cartesian_prod(*block_ranges)
    tuples_norm_array = torch.ones(prod_tuple.shape[0])
    for d in range(n):
        tuples_norm_array *= norm_arrays[d][prod_tuple[:, d]]
    
    sorted_tuples_norm_array, sorted_indices = torch.sort(tuples_norm_array, descending=True)
    sorted_tuples = prod_tuple[sorted_indices, :]

    queue = []
    start_time = time.time()
    lastscore = -torch.inf
    count_res = torch.zeros((sorted_tuples.shape[0],))
    res_div_max = torch.zeros((sorted_tuples.shape[0],))
    break_flag = 0


    
    for i in tqdm(range(sorted_tuples.shape[0])):
        block_indices = sorted_tuples[i, :] 
        candidate_indices_list = []
        for d in range(n):
            b_idx = block_indices[d].item()
            sorted_idx = sorted_indices_list[d]
            num_blocks = blocks_list[d]
            start_idx = b_idx * num_indices
            end_idx = (b_idx + 1) * num_indices if b_idx != num_blocks - 1 else sorted_idx.shape[0]
            candidate_indices_list.append(sorted_idx[start_idx:end_idx].squeeze(-1))
        
        testa = torch.cartesian_prod(*candidate_indices_list)  
        
        mask = torch.ones(train_data_array.shape[0], dtype=torch.bool, device=train_data_array.device)
        for d in range(n):
            col = train_data_array[:, d]
            block_min = candidate_indices_list[d].min().item()
            block_max = candidate_indices_list[d].max().item()
            mask = mask & (col >= block_min) & (col <= block_max)
        combined = torch.cat((train_data_array[mask, :], testa), dim=0)
        _, idx, counts = combined.unique(dim=0, return_inverse=True, return_counts=True)
        dup_mask = torch.isin(idx, torch.where(counts.gt(1))[0])
        mask_train = dup_mask[:train_data_array[mask, :].shape[0]]
        mask_testa = dup_mask[train_data_array[mask, :].shape[0]:]
        testa = testa[~mask_testa]
        num_exam += testa.shape[0]


        if testa.shape[0] == 0:
            continue

        prod = torch.ones(testa.shape[0], device=DEVICE)
        for d in range(n):
            prod = prod * embeddings[d][testa[:, d]]
        
        angle_i = model.angle([testa[:, d] for d in range(n)])
        res = prod * angle_i

        if i == 0:
            res_div_max[i] = angle_i.max().item()
            sorted_values, s_indices = torch.sort(res, descending=True)
            res_sorted = torch.cat((sorted_values.unsqueeze(1), testa[s_indices, :]), dim=1)
            res_sorted = res_sorted[:topk, :]
            res_sorted_tuple = tuple(map(tuple, res_sorted.detach().cpu().numpy()))
            queue = list(res_sorted_tuple)
            heapq.heapify(queue)
            lastscore = heapq.heappop(queue)
            heapq.heappush(queue, lastscore)
        else:
            lastscore = heapq.heappop(queue)
            heapq.heappush(queue, lastscore)
            check_index = torch.where(prod > lastscore[0])[0]
            prod = prod[check_index]
            testa = testa[check_index]
            count_res[i] = prod.shape[0]
            if testa.shape[0] == 0:
                continue
            angle_i = model.angle([testa[:, d] for d in range(n)])
            res = prod * angle_i
            res_div_max[i] = angle_i.max().item()
            remove_index = torch.where(res > lastscore[0])[0]
            res = res[remove_index]
            testa = testa[remove_index]
            sorted_values, s_indices = torch.sort(res, descending=True)
            res_sorted = torch.cat((sorted_values.unsqueeze(1), testa[s_indices, :]), dim=1)
            res_sorted_tuple = tuple(map(tuple, res_sorted.detach().cpu().numpy()))
            testp = 0
            for j in range(len(res_sorted_tuple)):
                lastscore = heapq.heappop(queue)
                if lastscore[0] < res_sorted_tuple[j][0]:
                    heapq.heappush(queue, res_sorted_tuple[j])
                    testp += 1
                else:
                    heapq.heappush(queue, lastscore)
                    break
            if testp == 0:
                break_flag += 1
            else:
                break_flag = 0

        if stop and (i > 0):
            if i + 1 < sorted_tuples.shape[0]:
                if sorted_tuples_norm_array[i + 1] * (res_div_max[:i].max()) * (alpha ** i) < lastscore[0]:
                    break
        else:
            if i + 1 < sorted_tuples.shape[0]:
                if sorted_tuples_norm_array[i + 1] < lastscore[0]:
                    break

    end_time = time.time()
    return queue, num_exam

def precision_recall2(queue, answer, k):
    res_indices = torch.FloatTensor(queue).to(DEVICE)

    sorted, indices = torch.sort(res_indices[:,0], descending=True)
    res_indices = res_indices[indices,1:].long()


    _, idx, counts = torch.cat((answer, res_indices), dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)

    
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    mask1 = mask[:len(answer)]  # tensor([ True, False,  True], device='cuda:0')
    mask2 = mask[len(answer):]  # tensor([ True, False, False,  True], device='cuda:0')
    precision = mask2[:k].sum().item()/k
    recall = mask2.sum().item()/answer.shape[0]

    return precision, recall

def apk_function2(queue, answer, k):
    res_indices = torch.FloatTensor(queue).to(DEVICE)
    sorted, indices = torch.sort(res_indices[:,0], descending=True)
    res_indices = res_indices[indices,1:].long()

    _, idx, counts = torch.cat((answer, res_indices), dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    mask1 = mask[:len(answer)]  # tensor([ True, False,  True], device='cuda:0')
    mask2 = mask[len(answer):]  # tensor([ True, False, False,  True], device='cuda:0')
    
    hit = 0
    apk = 0
    hit_position = []
    for i in range(k):
        if mask2[i]:
            hit += 1
            apk += hit/(i+1)
            hit_position.append(i)
    if hit == 0:
        apk = 0
    else:
        apk = apk/min(k, answer.shape[0])

    return apk
