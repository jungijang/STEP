import numpy as np

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
