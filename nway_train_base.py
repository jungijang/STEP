import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import time
import argparse 
import itertools
import torch.nn.functional as F

from modelN_base import CP, Tucker, CostCo, MLP, MDMTF, NeAT
from metric import precision_recall, apk_function

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TensorDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx, :])

def get_flat_indices(data, dims):
    return np.ravel_multi_index(data.T, dims, order='C')

def evaluate_model(model, dims, train_masked_indices, target_answer_indices, batch_size, device, k_metric=10000):

    model.eval()
    start_time = time.time()
    
    static_dims = dims[:-1]
    last_dim = dims[-1]
    total_combinations = int(np.prod(dims))
    
    grid_args = [torch.arange(d, device=device) for d in static_dims]
    try:
        base_grid = torch.cartesian_prod(*grid_args)
    except RuntimeError:
        print("Error: Base grid too large for GPU/RAM.")
        return 0.0

    base_grid_len = base_grid.shape[0]

    last_dim_col = torch.zeros((base_grid_len, 1), dtype=torch.long, device=device)
    test_batch = torch.cat([base_grid, last_dim_col], dim=1) # (User*Item, N)

    try:
        full_preds = np.zeros(total_combinations, dtype=np.float32)
    except MemoryError:
        print("Error: full_preds array too large.")
        return 0.0

    reshaped_view = full_preds.reshape(-1, last_dim)
    
    with torch.no_grad():
        for t in tqdm(range(last_dim)):
            
            time_step_preds = np.zeros(base_grid_len, dtype=np.float32)
            
            for i in range(0, base_grid_len, batch_size):
                batch_indices = test_batch[i : i + batch_size]
                batch_out = model(batch_indices).detach().cpu().numpy().squeeze()
                time_step_preds[i : i + batch_size] = batch_out
            
            reshaped_view[:, t] = time_step_preds
            
            test_batch[:, -1] += 1

    full_preds[train_masked_indices] = -np.inf

    k_eval = k_metric 
    if k_eval > len(full_preds): k_eval = len(full_preds)
        
    idx_test = np.argpartition(full_preds, -k_eval)[-k_eval:]
    sorted_list = idx_test[np.argsort(full_preds[idx_test])][::-1]
    valid_time = time.time() - start_time

    prec, _ = precision_recall(target_answer_indices, sorted_list, k_metric)
    ap_score = apk_function(target_answer_indices, sorted_list, k_metric)
    
    return prec, valid_time, sorted_list # sorted_list는 Test 단계에서 상세 Metric 계산용

def main():

    parser = argparse.ArgumentParser(description=' Tensor Factorization')
    parser.add_argument('--model', type=str, default='CP', 
                        choices=['CP', 'Tucker', 'CostCo', 'MLP', 'MDMTF', 'NeAT'])
    parser.add_argument('--dataset', type=str, default='gowalla')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', type=str, default='0')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Model: {args.model}, Dataset: {args.dataset}, Batch: {args.batch_size}")


    path = os.path.join('./data', args.dataset)
    train_path = os.path.join(path, f'{args.dataset}_train.tsv')
    valid_path = os.path.join(path, f'{args.dataset}_valid.tsv')
    test_path = os.path.join(path, f'{args.dataset}_test.tsv')

    df_train = pd.read_csv(train_path, header=None, sep='\t')
    df_valid = pd.read_csv(valid_path, header=None, sep='\t')
    df_test = pd.read_csv(test_path, header=None, sep='\t')

    train_data = df_train.to_numpy()
    val_data = df_valid.to_numpy()
    test_data = df_test.to_numpy()

    num_modes = train_data.shape[1]
    
    all_data = np.vstack([train_data, val_data, test_data])
    dims = []
    for i in range(num_modes):
        dims.append(int(np.max(all_data[:, i]) + 1))
    
    print(f"Dimensions: {dims}")

    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    train_answer_indices = get_flat_indices(train_data, dims)
    val_answer_indices = get_flat_indices(val_data, dims)
    test_answer_indices = get_flat_indices(test_data, dims)


    K = 10
    lr = 1e-2
    
    if args.model == 'CP': model = CP(dims, K)
    elif args.model == 'Tucker': model = Tucker(dims, K)
    elif args.model == 'CostCo': model = CostCo(dims, K)
    elif args.model == 'MLP': model = MLP(dims, K)
    elif args.model == 'MDMTF': model = MDMTF(dims, K)
    elif args.model == 'NeAT': model = NeAT(dims, K, hidden_dims=[10], dropout=0.0)
    else: raise ValueError(f"Unknown model: {args.model}")

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_dir = './model_res'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_name = f"{args.model}_{args.dataset}_ized"
    
    test_num = 1
    k_list = [10, 50, 100, 500, 1000, 5000, 10000]
    
    prec_list = np.zeros((test_num, len(k_list)))
    recall_list = np.zeros((test_num, len(k_list)))
    apk_list = np.zeros((test_num, len(k_list)))
    run_times = np.zeros((test_num, args.epochs + 1))

    eval_batch_size = 1000000

    # ---------------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------------
    for test_idx in range(test_num):
        if test_idx > 0:
            pass

        best_score = 0.0

        for epoch in tqdm(range(args.epochs), desc=f"Run {test_idx+1}/{test_num}"):
            model.train()
            train_loss = 0.
            
            # --- Training Step ---
            for batch_idx, x in enumerate(train_dataloader):
                x = x.to(DEVICE)
                optimizer.zero_grad()
                batch_size_curr = x.shape[0]

                neg_samples_list = [torch.randint(0, dims[d], (batch_size_curr, 1)).to(DEVICE) for d in range(num_modes)]
                neg_set = torch.cat(neg_samples_list, dim=1)

                pred_pos = model(x)
                pred_neg = model(neg_set)

                loss = torch.mean(-F.logsigmoid(pred_pos - pred_neg))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()
           
            val_ap, valid_time, _ = evaluate_model(
                model, dims, 
                train_masked_indices=train_answer_indices, 
                target_answer_indices=val_answer_indices, 
                batch_size=eval_batch_size, 
                device=DEVICE, 
                k_metric=k_list[-1]
            )
            
            run_times[test_idx, epoch] = valid_time
            print(f"Ep {epoch+1}: Loss {train_loss:.4f} | Val Prec@10k: {val_ap:.4f}")

            if val_ap > best_score:
                best_score = val_ap
                torch.save(model.state_dict(), f'{save_dir}/best_model_{test_idx}_{model_name}2.pth')


        print("Starting Final Test Evaluation...")
        
        best_model_path = f'{save_dir}/best_model_{test_idx}_{model_name}.pth'
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
        else:
            print("Warning: Best model not found. Using last epoch model.")

        _, test_time, sorted_list = evaluate_model(
            model, dims, 
            train_masked_indices=train_answer_indices, 
            target_answer_indices=test_answer_indices,
            batch_size=eval_batch_size, 
            device=DEVICE, 
            k_metric=k_list[-1]
        )
        run_times[test_idx, args.epochs] = test_time
        
        
        for i, k in enumerate(k_list):
            p, r = precision_recall(test_answer_indices, sorted_list, k)
            ap = apk_function(test_answer_indices, sorted_list, k)
            
            prec_list[test_idx, i] = p
            recall_list[test_idx, i] = r
            apk_list[test_idx, i] = ap
        
        print(f"Run {test_idx} Final APs: {apk_list[test_idx]}")

    result_path = os.path.join('./result', args.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    pd.DataFrame(prec_list).to_csv(f'{result_path}/prec_{model_name}2.tsv', sep='\t', header=None, index=False)
    pd.DataFrame(recall_list).to_csv(f'{result_path}/recall_{model_name}2.tsv', sep='\t', header=None, index=False)
    pd.DataFrame(apk_list).to_csv(f'{result_path}/ap_{model_name}2.tsv', sep='\t', header=None, index=False)
    pd.DataFrame(run_times).to_csv(f'{result_path}/rt_{model_name}2.tsv', sep='\t', header=None, index=False)

if __name__ == '__main__':
    main()
