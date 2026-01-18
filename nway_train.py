import argparse
import os
import time
import psutil

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from modelN import SCPN, STuckerN, SMLPN, SCostCoN, SMDMTF, SNeAT, Ablation_MLP, Ablation_Norm
from topk_predictN import topk_predict, apk_function2, precision_recall2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TensorDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx, :])


def compute_num_modes(train_data: np.ndarray, valid_data: np.ndarray, test_data: np.ndarray):
    D = train_data.shape[1]
    num_modes = []
    for d in range(D):
        mx = max(train_data[:, d].max(), valid_data[:, d].max(), test_data[:, d].max())
        num_modes.append(int(mx + 1))
    return num_modes


def sample_negative_batch(num_modes, batch_size, device):
    cols = [
        torch.randint(0, num_modes[d], (batch_size, 1), device=device)
        for d in range(len(num_modes))
    ]
    return torch.cat(cols, dim=1)


def resample_duplicates(train_data_array, neg_set, num_modes, device):
    _, idx, counts = torch.cat((train_data_array, neg_set), dim=0).unique(
        dim=0, return_inverse=True, return_counts=True
    )
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    mask2 = mask[len(train_data_array):]

    out = True
    while out:
        dup_rows = torch.where(mask2 == True)[0]
        if dup_rows.numel() == 0:
            out = False
            break

        new_rows = []
        for d in range(len(num_modes)):
            new_rows.append(
                torch.randint(0, num_modes[d], (dup_rows.numel(), 1), device=device)
            )
        new_rows = torch.cat(new_rows, dim=1)
        neg_set[dup_rows] = new_rows

        _, idx, counts = torch.cat((train_data_array, neg_set), dim=0).unique(
            dim=0, return_inverse=True, return_counts=True
        )
        mask = torch.isin(idx, torch.where(counts.gt(1))[0])
        mask2 = mask[len(train_data_array):]

    return neg_set


def build_model(model_name: str, num_modes, K: int, beta: float):
    name = model_name.lower()
    if name == "smlpn":
        return SMLPN(num_modes=num_modes, K=K, beta=beta)
    if name == "scostcon":
        return SCostCoN(num_modes=num_modes, K=K, beta=beta)
    if name == "scpn":
        return SCPN(num_modes=num_modes, K=K)
    if name == "stuckern":
        return STuckerN(num_modes=num_modes, K=K)
    
    if name == "smdmtf":
        return SMDMTF(num_modes=num_modes, K=K)
    if name == "sneat":
        return SNeAT(num_modes=num_modes, K=K, beta=beta)

    if name == "ablation_mlp":
        return Ablation_MLP(num_modes=num_modes, K=K)
    if name == "ablation_norm":
        return Ablation_Norm(num_modes=num_modes, K=K)
        
        
    raise ValueError(f"Unknown model_name={model_name}. Choose from: SCPN, STuckerN, SMLPN, SCostCoN, SMDMTF, SNeAT.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/gowalla")
    parser.add_argument("--train_file", type=str, default="gowalla_train.tsv")
    parser.add_argument("--valid_file", type=str, default="gowalla_valid.tsv")
    parser.add_argument("--test_file", type=str, default="gowalla_test.tsv")
    parser.add_argument("--result_dir", type=str, default="./result/gowalla")
    parser.add_argument("--sample_number", type=str, default="0")

    parser.add_argument("--model", type=str, default="SCostCoN", 
                        choices=["SMLPN", "SCostCoN", "SCPN", "STuckerN", "SMDMTF", "SNeAT", "Ablation_MLP", "Ablation_Norm"])
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--beta", type=float, default=2.0)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--stop", type=int, default=1)

    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--candidate_budget", type=int, default=200)
    parser.add_argument("--k_list", type=int, nargs="+", default=[10, 50, 100, 500, 1000, 5000, 10000])

    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./model_res")

    parser.add_argument("--use_angle_weight_reg", default=True)
    parser.add_argument("--angle_weight_lambda", type=float, default=1e-2)

    args = parser.parse_args()

    if args.model in ["SCPN", "STuckerN", "SMDMTF", "Ablation_MLP", "Ablation_Norm"]:
        args.use_angle_weight_reg = False
        print(f"[*] Model is {args.model}: forcing use_angle_weight_reg = False")
    elif args.model in ["SNeAT", "SMLPN", "SCostCoN"]:
        args.use_angle_weight_reg = True
        print(f"[*] Model is {args.model}: forcing use_angle_weight_reg = True")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    df_train = pd.read_csv(os.path.join(args.data_path, args.train_file), header=None, sep="\t")
    df_valid = pd.read_csv(os.path.join(args.data_path, args.valid_file), header=None, sep="\t")
    df_test = pd.read_csv(os.path.join(args.data_path, args.test_file), header=None, sep="\t")

    train_data = df_train.to_numpy()
    val_data = df_valid.to_numpy()
    test_data = df_test.to_numpy()

    D = train_data.shape[1] 
    num_modes = compute_num_modes(train_data, val_data, test_data)

    train_data_array = torch.LongTensor(train_data).to(DEVICE)
    val_data_array = torch.LongTensor(val_data).to(DEVICE)
    test_data_array = torch.LongTensor(test_data).to(DEVICE)

    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    prec_list = np.zeros((args.test_num, len(args.k_list)))
    recall_list = np.zeros((args.test_num, len(args.k_list)))
    apk_list = np.zeros((args.test_num, len(args.k_list)))
    run_times = np.zeros((args.test_num, args.epochs + 1))
    num_exams = np.zeros((args.test_num, args.epochs + 1))

    gpu_mem_max_list = np.zeros((args.test_num, args.epochs + 1)) 
    ram_mem_list = np.zeros((args.test_num, args.epochs + 1))

    for test_idx in range(args.test_num):
        model_tag = f"{args.model}_K{args.K}_b{args.beta}"
        model = build_model(args.model, num_modes=num_modes, K=args.K, beta=args.beta).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_prec = -1.0
        data_name = args.data_path.split('/')[-1]
        best_path = os.path.join(args.save_dir, f"best_{data_name}_{model_tag}_{test_idx}_{args.sample_number}.pt")

        for epoch in tqdm(range(args.epochs), desc=f"[{model_tag}] run={test_idx}"):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            model.train()

            for _, x in enumerate(train_loader):
                x = x.to(DEVICE)
                B = x.shape[0]
                idxs = [x[:, d] for d in range(D)]

                optimizer.zero_grad()

                neg_set = sample_negative_batch(num_modes, B, device=DEVICE)
                neg_set = resample_duplicates(train_data_array, neg_set, num_modes, device=DEVICE)

                pred, pred_angle = model(*idxs)
                pred_neg, pred_neg_angle = model(*[neg_set[:, d] for d in range(D)])

                pos_term = pred * pred_angle
                neg_term = pred_neg * pred_neg_angle
                score_diff = pos_term - neg_term
                bpr_loss = torch.mean(-F.logsigmoid(score_diff))
                loss = bpr_loss

                if args.use_angle_weight_reg and hasattr(model, "angle_weight"):
                    aw = model.angle_weight(*idxs)
                    loss = loss + args.angle_weight_lambda * torch.mean(F.softplus(aw))

                loss.backward()
                optimizer.step()

            model.eval()
            start_time = time.time()

            with torch.no_grad():
                result_queue, num_exam = topk_predict(
                    model,
                    num_modes,
                    args.candidate_budget,
                    max(args.k_list),
                    train_data_array,
                    args.alpha,
                    args.stop,
                )
            num_exams[test_idx, epoch] = num_exam
            end_time = time.time()
            run_times[test_idx, epoch] = end_time - start_time
            
            if torch.cuda.is_available():
                gpu_mem_max_list[test_idx, epoch] = torch.cuda.max_memory_allocated() / 1048576.0 
            
            process = psutil.Process(os.getpid())
            ram_mem_list[test_idx, epoch] = process.memory_info().rss / 1048576.0 

            prec100, _ = precision_recall2(result_queue, val_data_array, 100)
            apk100 = apk_function2(result_queue, val_data_array, 100)
            prec1k, _ = precision_recall2(result_queue, val_data_array, 1000)
            apk1k = apk_function2(result_queue, val_data_array, 1000)
            prec10k, _ = precision_recall2(result_queue, val_data_array, 10000)
            apk10k = apk_function2(result_queue, val_data_array, 10000)

            print(
                f"[{end_time-start_time:.2f}s] epoch {epoch+1:3d} "
                f"loss={bpr_loss:.4f} prec@100/1k/10k={prec100:.4f}/{prec1k:.4f}/{prec10k:.4f}"
            )
            print(
                f"[{end_time-start_time:.2f}s] epoch {epoch+1:3d} "
                f"apk@100/1k/10k ={apk100:.4f}/{apk1k:.4f}/{apk10k:.4f}\n"
            )

            if prec10k > best_prec or epoch==0:
                best_prec = prec10k
                torch.save(model.state_dict(), best_path)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))    
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            result_queue, test_num_exam = topk_predict(
                model,
                num_modes,
                args.candidate_budget,
                max(args.k_list),
                train_data_array,
                args.alpha,
                args.stop,
            )
        num_exams[test_idx, -1] = test_num_exam
        end_time = time.time()
        run_times[test_idx, args.epochs] = end_time - start_time
        
        if torch.cuda.is_available():
            gpu_mem_max_list[test_idx, -1] = torch.cuda.max_memory_allocated() / 1048576.0
        
        process = psutil.Process(os.getpid())
        ram_mem_list[test_idx, -1] = process.memory_info().rss / 1048576.0
        
        for kk_i, kk in enumerate(args.k_list):
            prec_res, recall_res = precision_recall2(result_queue, test_data_array, kk)
            ap_res = apk_function2(result_queue, test_data_array, kk)
            prec_list[test_idx, kk_i] = prec_res
            recall_list[test_idx, kk_i] = recall_res
            apk_list[test_idx, kk_i] = ap_res

        print("test prec:", prec_list[test_idx])

    prefix = f"{args.model}_K{args.K}_bs{args.batch_size}_{args.sample_number}_{args.alpha}_{args.stop}_{args.beta}_{args.candidate_budget}"
    pd.DataFrame(prec_list).to_csv(os.path.join(args.result_dir, f"prec_{prefix}.tsv"),
                                   sep="\t", header=None, index=False)
    pd.DataFrame(apk_list).to_csv(os.path.join(args.result_dir, f"apk_{prefix}.tsv"),
                                  sep="\t", header=None, index=False)
    pd.DataFrame(recall_list).to_csv(os.path.join(args.result_dir, f"recall_{prefix}.tsv"),
                                     sep="\t", header=None, index=False)
    pd.DataFrame(run_times).to_csv(os.path.join(args.result_dir, f"runtime_{prefix}.tsv"),
                                   sep="\t", header=None, index=False)
    pd.DataFrame(num_exams).to_csv(os.path.join(args.result_dir, f"exams_{prefix}.tsv"),
                                   sep="\t", header=None, index=False)
                                   
    pd.DataFrame(gpu_mem_max_list).to_csv(os.path.join(args.result_dir, f"memory_gpu_{prefix}.tsv"),
                                          sep="\t", header=None, index=False)
    pd.DataFrame(ram_mem_list).to_csv(os.path.join(args.result_dir, f"memory_ram_{prefix}.tsv"),
                                      sep="\t", header=None, index=False)


if __name__ == "__main__":
    main()