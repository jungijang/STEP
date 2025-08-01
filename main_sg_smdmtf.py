import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import time
import scipy
import bottleneck
import torch.nn.functional as F
from model import SMDMTF
from topk_predictN import topk_predict, apk_function2, precision_recall2
import torch.optim.lr_scheduler as lr_scheduler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')

class TensorDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.x[idx, :])
        return x

def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-8))

def main():
    path = './data/sg'
    df_train = pd.read_csv(os.path.join(path,'train_indices.tsv'), header=None, sep='\t')
    df_valid = pd.read_csv(os.path.join(path,'valid_indices.tsv'), header=None, sep='\t')
    df_test = pd.read_csv(os.path.join(path,'test_indices.tsv'), header=None, sep='\t')

    train_data = df_train.to_numpy()
    val_data = df_valid.to_numpy()
    test_data = df_test.to_numpy()

    train_data_array = torch.LongTensor(train_data)
    train_data_array = train_data_array.to(DEVICE)

    val_data_array = torch.LongTensor(val_data)
    val_data_array = val_data_array.to(DEVICE)

    test_data_array = torch.LongTensor(test_data)
    test_data_array = test_data_array.to(DEVICE)        

    num_users = int(max(max(train_data[:, 0]), max(test_data[:, 0])) + 1)
    num_items = int(max(max(train_data[:, 1]), max(test_data[:, 1])) + 1)
    num_times = int(max(max(train_data[:, 2]), max(test_data[:, 2])) + 1)

    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_answer_indices = np.zeros((test_data.shape[0],), dtype=np.int64)
    for i in range(test_answer_indices.shape[0]):
        test_answer_indices[i] = (test_data[i,2]*num_users*num_items + test_data[i,0]*num_items + test_data[i,1]).astype(np.int64)
        
    train_answer_indices = np.zeros(train_data.shape[0], dtype=np.int64)
    for i in range(train_answer_indices.shape[0]):
        train_answer_indices[i] = (train_data[i,2]*num_users*num_items + train_data[i,0]*num_items + train_data[i,1]).astype(np.int64)
        
    val_answer_indices = np.zeros(val_data.shape[0], dtype=np.int64)
    for i in range(val_answer_indices.shape[0]):
        val_answer_indices[i] = (val_data[i,2]*num_users*num_items + val_data[i,0]*num_items + val_data[i,1]).astype(np.int64)    

    u_indices = torch.LongTensor([i for i in range(num_users)])
    i_indices = torch.LongTensor([i for i in range(num_items)])

    # testa = torch.cartesian_prod(u_indices, i_indices, (torch.zeros(1, dtype=torch.long))).to(DEVICE)

    MSEloss = torch.nn.MSELoss()
    test_num = 1
    K = 10
    lr = 1e-2
    decay = 0
    epochs = 20
    alpha = 0.999
    k_list = [100, 500, 1000, 5000, 10000]
    prec_list = np.zeros((test_num, len(k_list)))
    apk_list = np.zeros((test_num, len(k_list)))
    run_times = np.zeros((test_num, epochs))
    num_exams = np.zeros((test_num, epochs+1))
    for test_idx in range(test_num):
        # Hyperparameters

        model_name = 'SMDMTF'

        model = SMDMTF(3, [num_users, num_items, num_times], K)
        model.to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        
        batch_it = num_items*num_times
        n_negs = 1
        best_acc = 0

        for epoch in tqdm(range(epochs)):
            start_time = time.time()
            train_mse = 0.
            test_mse = 0.

            # train the model
            model.train()
            for batch_idx, x in (enumerate(train_dataloader)):
                # get data
                x = x.to(DEVICE)
                pred_res = torch.zeros(x.shape[0], 2).to(DEVICE)

                i, j, k = x[:, 0], x[:, 1], x[:, 2]
                optimizer.zero_grad()
                

                neg_i = torch.randint(0, num_users, (i.shape[0],1))
                neg_j = torch.randint(0, num_items, (j.shape[0],1))
                neg_k = torch.randint(0, num_times, (k.shape[0],1))

                neg_set = torch.cat((neg_i,neg_j,neg_k), dim=1).to(DEVICE)
                _, idx, counts = torch.cat((train_data_array, neg_set), dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
                mask = torch.isin(idx, torch.where(counts.gt(1))[0])
                mask1 = mask[:len(train_data_array)] 
                mask2 = mask[len(train_data_array):] 
                out = 1
                while out:
                    true_set2 = torch.where(mask2==True)[0]
                    if true_set2.shape[0] > 0:
        
                        neg_i2, neg_j2,neg_k2 = torch.randint(0, num_users, (true_set2.shape[0],1)), \
                        torch.randint(0, num_items, (true_set2.shape[0],1)), \
                        torch.randint(0, num_times, (true_set2.shape[0],1))     
                        neg_set2 = torch.cat((neg_i2,neg_j2,neg_k2), dim=1).to(DEVICE)
                        _, idx, counts = torch.cat((train_data_array, neg_set2), dim=0).unique(
                dim=0, return_inverse=True, return_counts=True)
                        mask = torch.isin(idx, torch.where(counts.gt(1))[0])
                        mask1 = mask[:len(train_data_array)] 
                        mask2 = mask[len(train_data_array):]
                        neg_set[true_set2,0] = neg_i2.squeeze().to(DEVICE)
                        neg_set[true_set2,1] = neg_j2.squeeze().to(DEVICE)
                        neg_set[true_set2,2] = neg_k2.squeeze().to(DEVICE)                         
                    else:
                        out = 0


                neg_set = neg_set.to(DEVICE)
                


                pred_neg, pred_neg_angle = model([neg_set[:,0], neg_set[:,1], neg_set[:,2]])

                pred, pred_angle = model([i, j, k])

                tau = 1
                bpr_loss = 0
                pos_term = pred * pred_angle
                neg_term = pred_neg * pred_neg_angle
                

                score_diff = pos_term - neg_term

                bpr_loss = torch.mean(-torch.log(torch.sigmoid(score_diff)))

                loss = bpr_loss

                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()
            train_rmse = (train_mse/(batch_idx+1))**.5
            model.eval()
            
            result_queue, num_exam = topk_predict(model, [num_users, num_items, num_times], 180, k_list[-1], train_data_array, alpha, True)
            num_exams[test_idx, epoch] = num_exam
            
            prec, recall = precision_recall2(result_queue, val_data_array, 100)
            apk_res = apk_function2(result_queue, val_data_array, 100)
            prec3, recall3 = precision_recall2(result_queue, val_data_array, 1000)
            apk_res3 = apk_function2(result_queue, val_data_array, 1000)
            prec2, recall2 = precision_recall2(result_queue, val_data_array, 10000)
            apk_res2 = apk_function2(result_queue, val_data_array, 10000)     

            end_time = time.time()            

            
            print(f'[{end_time-start_time:.2f}] Epoch: {epoch+1:3d}, '
                  f'TrnRMSE: {bpr_loss:.4f}, TestRMSE: {prec:.4f} and {prec3:.4f} and {prec2:.4f}')
            print(f'[{end_time-start_time:.2f}] Epoch: {epoch+1:3d}, '
                  f'TrnRMSE: {bpr_loss:.4f}, TestRMSE: {apk_res:.4f} and {apk_res3:.4f} and {apk_res2:.4f}\n')

            run_times[test_idx, epoch] = end_time - start_time
            if best_acc <  apk_res2 or epoch ==0:
                torch.save(model.state_dict(), './model_res/sg_best_model'+str(model_name))
                best_acc =  apk_res2

        # Evaluation
        u_indices = torch.LongTensor([i for i in range(num_users)])
        i_indices = torch.LongTensor([i for i in range(num_items)])

        testa = torch.cartesian_prod(u_indices, i_indices, (torch.zeros(1, dtype=torch.long))).to(DEVICE)

        model.load_state_dict(torch.load('./model_res/sg_best_model'+str(model_name)))
        model.eval()

        result_queue, test_num_exam = topk_predict(model, [num_users, num_items, num_times], 180, k_list[-1], train_data_array, alpha, True)
        num_exams[test_idx, -1] = test_num_exam


        for i in range(len(k_list)):
            prec_res, _ = precision_recall2(result_queue, test_data_array, k_list[i])
            ap_res = apk_function2(result_queue, test_data_array, k_list[i])

            prec_list[test_idx,i] = prec_res
            apk_list[test_idx,i] = ap_res
        print(apk_list)

    prec_list = pd.DataFrame(prec_list)
    apk_list = pd.DataFrame(apk_list)
    run_times = pd.DataFrame(run_times)
    num_exams = pd.DataFrame(num_exams)

    prec_list.to_csv('./result/sg/prec_res_' + str(model_name)+'_full.tsv', sep='\t', header=None, index=False)
    apk_list.to_csv('./result/sg/ap_res_' + str(model_name)+'_full.tsv', sep='\t', header=None, index=False)
    run_times.to_csv('./result/sg/rt_' + str(model_name)+'_full.tsv', sep='\t', header=None, index=False)
    num_exams.to_csv('./result/sg/exams_' + str(model_name)+'_full.tsv', sep='\t', header=None, index=False)

if __name__ == '__main__':
    main()
