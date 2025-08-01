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
from model import CP, Tucker, CostCo, MLP
from topk_predict_base import precision_recall, apk_function
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


def main():
    path = './data/sg'
    df_train = pd.read_csv(os.path.join(path,'train_indices.tsv'), header=None, sep='\t')
    df_valid = pd.read_csv(os.path.join(path,'valid_indices.tsv'), header=None, sep='\t')
    df_test = pd.read_csv(os.path.join(path,'test_indices.tsv'), header=None, sep='\t')
    
    train_data = df_train.to_numpy()
    val_data = df_valid.to_numpy()
    test_data = df_test.to_numpy()

    num_users = int(max(max(train_data[:, 0]), max(test_data[:, 0])) + 1)
    num_items = int(max(max(train_data[:, 1]), max(test_data[:, 1])) + 1)
    num_times = int(max(max(train_data[:, 2]), max(test_data[:, 2])) + 1)

    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

    train_data_array = torch.LongTensor(train_data)
    train_data_array = train_data_array.to(DEVICE)    

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

    testa = torch.cartesian_prod(u_indices, i_indices, (torch.zeros(1, dtype=torch.long))).to(DEVICE)

    test_num = 1
    K = 10
    hidden_dim1 = 10
    hidden_dim2 = 10
    lr = 1e-2
    decay = 0 
    epochs = 20
    k_list = [100, 500, 1000, 5000, 10000]
    prec_list = np.zeros((test_num, len(k_list)))
    apk_list = np.zeros((test_num, len(k_list)))
    run_times = np.zeros((test_num, epochs))
    for test_idx in range(test_num):

        # model_name = 'CP'
        # model = CP(num_users, num_items, num_times, K)

        # model_name = 'Tucker'
        # model = Tucker(num_users, num_items, num_times, K)

        # model_name = 'CostCo'
        # model = CostCo(num_users, num_items, num_times, K)   

        model_name = 'MLP'
        model = MLP(num_users, num_items, num_times, K)           

        model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        batch_it = num_items*num_times
        n_negs = 1
        best_acc = 0

        for epoch in tqdm(range(epochs)):
            train_mse = 0.
            test_mse = 0.

            # train the model
            model.train()
            for batch_idx, x in (enumerate(train_dataloader)):
                # get data
                x = x.to(DEVICE)
                pred_res = torch.zeros(x.shape[0], n_negs + 1).to(DEVICE)

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
                true_set = torch.where(mask2==True)[0]

                neg_i2, neg_j2,neg_k2 = torch.randint(0, num_users, (true_set.shape[0],)), \
                torch.randint(0, num_items, (true_set.shape[0],)), \
                torch.randint(0, num_times, (true_set.shape[0],))                
                neg_set[true_set,0] = neg_i2.to(DEVICE)
                neg_set[true_set,1] = neg_j2.to(DEVICE)
                neg_set[true_set,2] = neg_k2.to(DEVICE)

                neg_set = neg_set.to(DEVICE)                
                pred_neg = model(neg_set[:,0], neg_set[:,1], neg_set[:,2])
                pred_res[:,0] = pred_neg.squeeze()
                pred = model(i, j, k)
                pred_res[:, -1] = pred.squeeze()

                bpr_loss = 0
                tau = 1
                bpr_loss = 0
                pos_term = pred
                neg_term = pred_neg
                
                score_diff = pos_term - neg_term
                bpr_loss += torch.mean(-torch.log(torch.sigmoid(score_diff)))            
                loss = bpr_loss

                train_mse += loss.item()

                # backpropagation
                loss.backward()

                # update the parameters
                optimizer.step()

            train_rmse = (train_mse/(batch_idx+1))**.5
            model.eval()
            start_time = time.time()
            testa = torch.cartesian_prod(u_indices, i_indices, torch.zeros(1, dtype=torch.long)).to(DEVICE)

            prec = 0
            prec2 = 0
            prec3 = 0
            recall = 0
            recall2 = 0
            apk_res = 0
            apk_res2 = 0
            apk_res3 = 0
            pred_res1 = np.zeros((num_users*num_items*num_times,))
            for num_index in tqdm(range(num_times)):
                split_a = torch.split(testa, testa.shape[0]//1)
                partial_res = np.zeros(testa.shape[0],)
                partial_index = 0
                for partial_indices in (split_a):
                    partial_res[partial_index:partial_index+partial_indices.shape[0]] =\
                    model(partial_indices[:,0], partial_indices[:,1], partial_indices[:,2]).squeeze().detach().cpu().numpy()
                    partial_index += partial_indices.shape[0]

                pred_res1[num_index*(num_users*num_items):(num_index+1)*(num_users*num_items)] \
                = partial_res
                testa[:,2] += 1

            pred_res1[train_answer_indices] = -np.inf
            k=10000

            idx_test = np.argpartition(pred_res1, -k)[-k:]  # Indices not sorted
            sorted_list2 = idx_test[np.argsort(pred_res1[idx_test])][::-1]  # Indices sorted by value from largest to smallest

            
            prec, recall = precision_recall(val_answer_indices, sorted_list2, 100)
            apk_res = apk_function(val_answer_indices, sorted_list2, 100)
            prec3, recall3 = precision_recall(val_answer_indices, sorted_list2, 1000)
            apk_res3 = apk_function(val_answer_indices, sorted_list2, 1000)
            prec2, recall2 = precision_recall(val_answer_indices, sorted_list2, 10000)
            apk_res2 = apk_function(val_answer_indices, sorted_list2, 10000)
      
            end_time = time.time()

            print(f'[{end_time-start_time:.2f}] Epoch: {epoch+1:3d}, '
                  f'TrnRMSE: {bpr_loss:.4f}, TestRMSE: {prec:.4f} and {prec3:.4f} and {prec2:.4f}')
            print(f'[{end_time-start_time:.2f}] Epoch: {epoch+1:3d}, '
                  f'TrnRMSE: {bpr_loss:.4f}, TestRMSE: {apk_res:.4f} and {apk_res3:.4f} and {apk_res2:.4f}\n')

            run_times[test_idx, epoch] = end_time - start_time
            print(run_times[test_idx, epoch])
            if best_acc <  apk_res2 or epoch == 0:
                torch.save(model.state_dict(), './model_res/sg_best_model'+str(model_name))
                best_acc =  apk_res2

        # Evaluation
        u_indices = torch.LongTensor([i for i in range(num_users)])
        i_indices = torch.LongTensor([i for i in range(num_items)])

        testa = torch.cartesian_prod(u_indices, i_indices, (torch.zeros(1, dtype=torch.long))).to(DEVICE)

        model.load_state_dict(torch.load('./model_res/sg_best_model'+str(model_name)))
        model.eval()
        pred_res1 = np.zeros((num_users*num_items*num_times,))
        for num_index in tqdm(range(num_times)):
            split_a = torch.split(testa, testa.shape[0]//1)
            partial_res = np.zeros(testa.shape[0],)
            partial_index = 0
            for partial_indices in (split_a):
                partial_res[partial_index:partial_index+partial_indices.shape[0]] =\
                model(partial_indices[:,0], partial_indices[:,1], partial_indices[:,2]).squeeze().detach().cpu().numpy()
                partial_index += partial_indices.shape[0]
            
            pred_res1[num_index*(num_users*num_items):(num_index+1)*(num_users*num_items)] \
            = partial_res
            testa[:,2] += 1

        pred_res1[train_answer_indices] = -np.inf
        k=10000

        idx_test = np.argpartition(pred_res1, -k)[-k:]  # Indices not sorted
        sorted_list = idx_test[np.argsort(pred_res1[idx_test])][::-1]  # Indices sorted by value from largest to smallest


        for i in range(len(k_list)):
            prec_res, _ = precision_recall(test_answer_indices, sorted_list, k_list[i])
            ap_res = apk_function(test_answer_indices, sorted_list, k_list[i])

            prec_list[test_idx,i] = prec_res
            apk_list[test_idx,i] = ap_res

    prec_list = pd.DataFrame(prec_list)
    apk_list = pd.DataFrame(apk_list)
    run_times = pd.DataFrame(run_times)

    prec_list.to_csv('./result/sg/prec_res_' + str(model_name)+'_full.tsv', sep='\t', header=None, index=False)
    apk_list.to_csv('./result/sg/ap_res_' + str(model_name)+'_full.tsv', sep='\t', header=None, index=False)
    run_times.to_csv('./result/sg/rt_' + str(model_name)+'_full.tsv', sep='\t', header=None, index=False)

if __name__ == '__main__':
    main()
