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
        u_final, i_final, t_final = model.embeddings()
    user_idx = torch.arange(start=0, end=num_entities[0]).to(DEVICE)
    item_idx = torch.arange(start=0, end=num_entities[1]).to(DEVICE)
    time_idx = torch.arange(start=0, end=num_entities[2]).to(DEVICE)


    u_norm = u_final
    i_norm = i_final
    t_norm = t_final
    
    u_indices = torch.argsort(u_final, dim=-1, descending=True)
    i_indices = torch.argsort(i_final, dim=-1, descending=True)
    t_indices = torch.argsort(t_final, dim=-1, descending=True)

    num_indices = block_size
    
    u_blocks = math.ceil(u_indices.shape[0] / num_indices) if u_indices.shape[0] > 0 else 1
    i_blocks = math.ceil(i_indices.shape[0] / num_indices) if i_indices.shape[0] > 0 else 1
    t_blocks = math.ceil(t_indices.shape[0] / num_indices) if t_indices.shape[0] > 0 else 1

    u_norm_array = torch.zeros((u_blocks,))
    i_norm_array = torch.zeros((i_blocks,))
    t_norm_array = torch.zeros((t_blocks,))

    for i in range(u_blocks):
        if i != u_blocks-1:
            max_norm = torch.max(u_norm[u_indices[i*num_indices:(i+1)*num_indices]])
        else:
            max_norm = torch.max(u_norm[u_indices[i*num_indices:]])
        u_norm_array[i] = max_norm.item()
    
    for i in range(i_blocks):
        if i != i_blocks-1:
            max_norm = torch.max(i_norm[i_indices[i*num_indices:(i+1)*num_indices]])
        else:
            max_norm = torch.max(i_norm[i_indices[i*num_indices:]])
        i_norm_array[i] = max_norm.item()
    
    for i in range(t_blocks):
        if i != t_blocks-1:
            max_norm = torch.max(t_norm[t_indices[i*num_indices:(i+1)*num_indices]])
        else:
            max_norm = torch.max(t_norm[t_indices[i*num_indices:]])
        t_norm_array[i] = max_norm.item()

    tuples = torch.cartesian_prod(torch.arange(start=0, end=u_norm_array.shape[0]), torch.arange(start=0, end=i_norm_array.shape[0]), torch.arange(start=0, end=t_norm_array.shape[0]))
    tuples_norm_array = torch.zeros((tuples.shape[0],))

    for i in range(tuples.shape[0]):
        tuples_norm_array[i] = (u_norm_array[tuples[i,0]] * i_norm_array[tuples[i,1]] * t_norm_array[tuples[i,2]])

    sorted_tuples_norm_array, sorted_indices = torch.sort(tuples_norm_array, descending=True)
    sorted_tuples = tuples[sorted_indices,:]

    queue = []

    start = time.time()
    lastscore = -torch.inf
    count_res = np.zeros((sorted_tuples.shape[0],))
    res_div_max = np.zeros((sorted_tuples.shape[0],))
    break_flag = 0
    for i in tqdm(range(sorted_tuples.shape[0])):
        u_block, i_block, t_block = sorted_tuples[i,:]
        ub_ind, ib_ind, tb_ind = u_indices[u_block*num_indices:(u_block+1)*num_indices],\
                                     i_indices[i_block*num_indices:(i_block+1)*num_indices],\
                                     t_indices[t_block*num_indices:(t_block+1)*num_indices]
        testa = torch.cartesian_prod(ub_ind, ib_ind, tb_ind)

        mask_init = (train_data_array[:,0] >= ub_ind.min().item()) & (train_data_array[:,0] <= ub_ind.max().item()) &\
        (train_data_array[:,1] >= ib_ind.min().item()) & (train_data_array[:,1] <= ib_ind.max().item()) &\
        (train_data_array[:,2] >= tb_ind.min().item()) & (train_data_array[:,2] <= tb_ind.max().item())
        _, idx, counts = torch.cat((train_data_array[mask_init,:], testa), dim=0).unique(
            dim=0, return_inverse=True, return_counts=True)
        mask = torch.isin(idx, torch.where(counts.gt(1))[0])
        mask1 = mask[:len(train_data_array[mask_init,:])]  # tensor([ True, False,  True], device='cuda:0')
        mask2 = mask[len(train_data_array[mask_init,:]):]  # tensor([ True, False, False,  True], device='cuda:0')
        testa = testa[~mask2]
        num_exam += testa.shape[0]
        
        if i == 0:
            angle_i = model.angle(testa[:,0], testa[:,1], testa[:,2])
            res = u_final[testa[:,0]] * i_final[testa[:,1]] * t_final[testa[:,2]] * angle_i
            res_div_max[i] = angle_i.max().item()
            sorted_values, s_indices = torch.sort(res, descending=True)
            res = torch.cat((res[s_indices].unsqueeze(1), testa[s_indices,:]), dim=1)
            res = res[:topk,:]
            res = tuple(map(tuple, res.detach().cpu().numpy()))
            queue = list(res)
            heapq.heapify(queue)    
            lastscore = heapq.heappop(queue)
            heapq.heappush(queue, lastscore)
   
        else:
            lastscore = heapq.heappop(queue)
            heapq.heappush(queue, lastscore)
            
            product = u_final[testa[:,0]] * i_final[testa[:,1]] * t_final[testa[:,2]]
            check_index = torch.where(product>lastscore[0])[0]
            check_index = check_index
            product = product[check_index]
            testa = testa[check_index]     
            count_res[i] = product.shape[0]            
            if testa.shape[0] == 0:
                continue
            angle_i = model.angle(testa[:,0], testa[:,1], testa[:,2])
            res = product * angle_i

            res_div_max[i] = angle_i.max().item()
            

            remove_index = torch.where(res>lastscore[0])[0]
            res = res[remove_index]
            testa = testa[remove_index]
            sorted_values, s_indices = torch.sort(res, descending=True)
            res = torch.cat((res[s_indices].unsqueeze(1), testa[s_indices,:]), dim=1)
            res = tuple(map(tuple, res.detach().cpu().numpy()))

            testp = 0
            for j in (range(len(res))):
                lastscore = heapq.heappop(queue)
                # heapq.heappush(queue, res[j])
                if lastscore[0] < res[j][0]:
                    heapq.heappush(queue, res[j])
                    testp += 1
                else:
                    heapq.heappush(queue, lastscore)
                    break

            if testp == 0:
                break_flag += 1
            else:
                break_flag = 0

        if stop and (i>0):
            if i+1 < sorted_tuples.shape[0]:
                if sorted_tuples_norm_array[i+1] * (res_div_max[:i].max()) * (alpha**(i))  < lastscore[0]:
                    break

        else:
            if i+1 < sorted_tuples.shape[0]:
                if sorted_tuples_norm_array[i+1] < lastscore[0]:
                    break

    end = time.time()
    
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
