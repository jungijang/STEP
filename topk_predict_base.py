import heapq
import torch
from tqdm import tqdm

def topk_predict_lazy_best_first(model, num_entities, topk, train_data_array, alpha, stop=True):
    num_exam = 0
    with torch.no_grad():
        u_final, i_final, t_final = model.embeddings()

    u_norm = u_final
    i_norm = i_final
    t_norm = t_final

    u_sorted_indices = torch.argsort(u_norm, descending=True)
    i_sorted_indices = torch.argsort(i_norm, descending=True)
    t_sorted_indices = torch.argsort(t_norm, descending=True)    
    
    queue = []

    heapq.heappush(queue, (-(u_norm[u_sorted_indices[0]] * i_norm[i_sorted_indices[0]] * t_norm[t_sorted_indices[0]]), 
                          u_sorted_indices[0], i_sorted_indices[0], t_sorted_indices[0]))  # Max-heap by negative value
    
    result = []
    lastscore = -float('inf')
    max_int_score = -float('inf')
    num_iter = 0
    
    while queue:
        num_exam += 1
        num_iter += 1
        if num_iter % 10000 == 0:
            print(num_iter)
        score, u_block, i_block, t_block = heapq.heappop(queue)

        norm_prod = u_norm[u_block] * i_norm[i_block] * t_norm[t_block]

        angle_score = model.angle(u_block.unsqueeze(0), i_block.unsqueeze(0), t_block.unsqueeze(0))
        score = angle_score * norm_prod

        if len(result) < topk or score > result[-1][0]:
            result.append((score, u_block, i_block, t_block))
            result.sort(reverse=True, key=lambda x: x[0])  # Sort by score
            
            if len(result) > topk:
                result.pop()

        if max_int_score < angle_score:
            max_int_score = angle_score
        if stop and queue:
            next_u_block, next_i_block, next_t_block = queue[0][1], queue[0][2], queue[0][3]
            next_norm_prod = u_norm[next_u_block] * i_norm[next_i_block] * t_norm[next_t_block]

            if next_norm_prod * (alpha)**(num_iter) * max_int_score  < result[-1][0] and len(result) >= topk:
                break

        if u_block + 1 < num_entities[0]:
            heapq.heappush(queue, (-(u_norm[u_sorted_indices[u_block+1]] * i_norm[i_sorted_indices[i_block]] * t_norm[t_sorted_indices[t_block]]),
                                  u_sorted_indices[u_block+1], i_sorted_indices[i_block], t_sorted_indices[t_block]))
        if i_block + 1 < num_entities[1]:
            heapq.heappush(queue, (-(u_norm[u_sorted_indices[u_block]] * i_norm[i_sorted_indices[i_block+1]] * t_norm[t_sorted_indices[t_block]]),
                                  u_sorted_indices[u_block], i_sorted_indices[i_block+1], t_sorted_indices[t_block]))
        if t_block + 1 < num_entities[2]:
            heapq.heappush(queue, (-(u_norm[u_sorted_indices[u_block]] * i_norm[i_sorted_indices[i_block]] * t_norm[t_sorted_indices[t_block+1]]),
                                  u_sorted_indices[u_block], i_sorted_indices[i_block], t_sorted_indices[t_block+1]))

    return result, num_exam


def precision_recall(answer_list, sorted_list, k):
    hit = 0
    for i in range(k):
        if sorted_list[i] in answer_list:
            hit += 1

    precision = hit/k
    recall = hit/answer_list.shape[0]
    return precision, recall

def apk_function(answer_list, sorted_list, k):
    hit = 0
    apk = 0
    hit_position = []
    for i in range(k):
        if sorted_list[i] in answer_list:
            hit += 1
            apk += hit/(i+1)
            hit_position.append(i)
    if hit == 0:
        apk = 0
    else:
        apk = apk/min(k, answer_list.shape[0])

    return apk

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
