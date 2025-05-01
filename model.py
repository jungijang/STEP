import numpy as np
import torch
import time
import scipy
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import string
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


class CP(torch.nn.Module):
    def __init__(self, num_users, num_items, num_times, K):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, K)
        self.item_emb = torch.nn.Embedding(num_items, K)
        self.time_emb = torch.nn.Embedding(num_times, K)

        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
        torch.nn.init.xavier_uniform_(self.time_emb.weight)
        

    def forward(self, user_idx, item_idx, time_idx):
        out = ((self.user_emb(user_idx) * self.item_emb(item_idx) * self.time_emb(time_idx))).sum(dim=1)
        return out

class Tucker(torch.nn.Module):
    def __init__(self, num_users, num_items, num_times, K):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, K)
        self.item_emb = torch.nn.Embedding(num_items, K)
        self.time_emb = torch.nn.Embedding(num_times, K)
        self.core = torch.nn.Parameter(torch.ones(K, K, K))
        

        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
        torch.nn.init.xavier_uniform_(self.time_emb.weight)
        torch.nn.init.xavier_uniform_(self.core)

    def forward(self, user_idx, item_idx, time_idx):
        out = torch.einsum('bi, bj, bk, ijk->b', self.user_emb(user_idx), self.item_emb(item_idx), self.time_emb(time_idx), self.core)
        return out

class CostCo(torch.nn.Module):
    def __init__(self, num_users, num_items, num_times, K):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, K)
        self.item_emb = torch.nn.Embedding(num_items, K)
        self.time_emb = torch.nn.Embedding(num_times, K)
        self.m1 = torch.nn.Conv2d(1, K, (1,3), stride=1)
        self.m2 = torch.nn.Conv2d(K, K, (K,1), stride=1)
        self.agg1 = torch.nn.Linear(K, K)
        self.agg2 = torch.nn.Linear(K, 1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
        torch.nn.init.xavier_uniform_(self.time_emb.weight)

        torch.nn.init.xavier_uniform_(self.m1.weight)
        torch.nn.init.xavier_uniform_(self.m2.weight)
        torch.nn.init.xavier_uniform_(self.agg1.weight)
        torch.nn.init.xavier_uniform_(self.agg2.weight)

    def forward(self, user_idx, item_idx, time_idx):

        concat = torch.cat((self.user_emb.weight, self.item_emb.weight, self.time_emb.weight), dim=0)

        user_shape = self.user_emb.weight.shape[0]
        item_shape = self.item_emb.weight.shape[0]
        time_shape = self.time_emb.weight.shape[0]

        cat_vec1 = torch.stack((self.user_emb(user_idx),
                                self.item_emb(item_idx),
                                self.time_emb(time_idx)
                                ), dim=2).unsqueeze(1)

        out1 = self.relu(self.m2(self.relu(self.m1(cat_vec1)))).squeeze()
        out1 = self.relu(self.agg1(out1))
        pred = (self.agg2(out1))

        return pred

class SCP(torch.nn.Module):
    def __init__(self, num_users, num_items, num_times, K):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, K)
        self.item_emb = torch.nn.Embedding(num_items, K)
        self.time_emb = torch.nn.Embedding(num_times, K)

        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
        torch.nn.init.xavier_uniform_(self.time_emb.weight)

    def forward(self, user_idx, item_idx, time_idx, perturbed=False):

        u_final = self.user_emb(user_idx)
        i_final = self.item_emb(item_idx)
        t_final = self.time_emb(time_idx)

        u_final = torch.norm(u_final, dim=1)
        i_final = torch.norm(i_final, dim=1)
        t_final = torch.norm(t_final, dim=1)

        angle = self.angle(user_idx, item_idx, time_idx)

        return u_final * i_final * t_final,  angle
    
    def angle(self, user_idx, item_idx, time_idx):
        u_final = self.user_emb(user_idx)
        i_final = self.item_emb(item_idx)
        t_final = self.time_emb(time_idx)

        u_final = u_final/torch.norm(u_final, dim=1, keepdim=True)
        i_final = i_final/torch.norm(i_final, dim=1, keepdim=True)
        t_final = t_final/torch.norm(t_final, dim=1, keepdim=True)
        out = (u_final * i_final * t_final).sum(dim=1)
        return out.squeeze()

    def embeddings(self):
        
        u_final = self.user_emb.weight
        i_final = self.item_emb.weight
        t_final = self.time_emb.weight
        
        u_final = torch.norm(u_final, dim=1)
        i_final = torch.norm(i_final, dim=1)
        t_final = torch.norm(t_final, dim=1)
        
        return u_final, i_final, t_final  


class STucker(torch.nn.Module):
    def __init__(self, num_users, num_items, num_times, K):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, K)
        self.item_emb = torch.nn.Embedding(num_items, K)
        self.time_emb = torch.nn.Embedding(num_times, K)
        self.core = torch.nn.Parameter(torch.ones(K, K, K))
        

        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
        torch.nn.init.xavier_uniform_(self.time_emb.weight)
        torch.nn.init.xavier_uniform_(self.core)
     

    def forward(self, user_idx, item_idx, time_idx, perturbed=False):

        u_final = self.user_emb(user_idx)
        i_final = self.item_emb(item_idx)
        t_final = self.time_emb(time_idx)

        u_final = torch.norm(u_final, dim=1)
        i_final = torch.norm(i_final, dim=1)
        t_final = torch.norm(t_final, dim=1)
        core_final = torch.norm(self.core)

        angle = self.angle(user_idx, item_idx, time_idx)

        return u_final * i_final * t_final,  angle
    
    def angle(self, user_idx, item_idx, time_idx):
        core = self.core
        u_final = self.user_emb(user_idx)
        i_final = self.item_emb(item_idx)
        t_final = self.time_emb(time_idx)

        if self.training:
            angle_weight = torch.norm(u_final, dim=1) * torch.norm(i_final, dim=1) * torch.norm(t_final, dim=1)
            out = (torch.einsum('bi, bj, bk, ijk->b', u_final, i_final, t_final, core))/angle_weight            
        else:
            angle_weight = torch.norm(self.core, p=1, keepdim=True) * torch.norm(u_final, dim=1) * torch.norm(i_final, dim=1) * torch.norm(t_final, dim=1)
            out = (torch.einsum('bi, bj, bk, ijk->b', u_final, i_final, t_final, core))/angle_weight
        return out.squeeze()

    def embeddings(self):
        
        u_final = self.user_emb.weight
        i_final = self.item_emb.weight
        t_final = self.time_emb.weight
        
        u_final = torch.norm(u_final, dim=1)
        i_final = torch.norm(i_final, dim=1)
        t_final = torch.norm(t_final, dim=1)
        core_final = torch.norm(self.core)
        
        return u_final, i_final, t_final

class SCostCo(torch.nn.Module):
    def __init__(self, num_users, num_items, num_times, K, beta):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, K)
        self.item_emb = torch.nn.Embedding(num_items, K)
        self.time_emb = torch.nn.Embedding(num_times, K)       
        self.m1 = torch.nn.Conv2d(1, K, (1,3), stride=1)
        self.m2 = torch.nn.Conv2d(K, K, (K,1), stride=1)
        self.softplus = torch.nn.Softplus()
        self.K = K
        self.beta = beta
        self.agg12 = torch.nn.Linear(K, K)
        self.agg13 = torch.nn.Linear(K, 1)
        self.agg5 = torch.nn.Linear(K, 1)
        self.relu = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
        torch.nn.init.xavier_uniform_(self.time_emb.weight)   
        torch.nn.init.xavier_uniform_(self.m1.weight)
        torch.nn.init.xavier_uniform_(self.m2.weight)

        torch.nn.init.xavier_uniform_(self.agg12.weight)
        torch.nn.init.xavier_uniform_(self.agg13.weight)
        torch.nn.init.xavier_uniform_(self.agg5.weight)

    def forward(self, user_idx, item_idx, time_idx, perturbed=False):

        u_final = self.user_emb(user_idx)
        i_final = self.item_emb(item_idx)
        t_final = self.time_emb(time_idx)

        u_final = 1+self.softplus(self.agg5(u_final)).squeeze()
        i_final = 1+self.softplus(self.agg5(i_final)).squeeze()
        t_final = 1+self.softplus(self.agg5(t_final)).squeeze()

        angle = self.angle(user_idx, item_idx, time_idx)

        return u_final * i_final * t_final,  angle

    def angle(self, user_idx, item_idx, time_idx):
        
        u_final = self.user_emb(user_idx)
        i_final = self.item_emb(item_idx)
        t_final = self.time_emb(time_idx)
 
        u_final = 1+self.softplus(self.agg5(u_final))
        i_final = 1+self.softplus(self.agg5(i_final))
        t_final = 1+self.softplus(self.agg5(t_final))
        
        cat_vec1 = torch.stack((self.user_emb(user_idx),
                                self.item_emb(item_idx),
                                self.time_emb(time_idx)
                                ), dim=2).unsqueeze(1)        

        out1 = self.relu(self.m2(self.relu(self.m1(cat_vec1)))).squeeze()
        out1 = self.relu(self.agg12(out1))
        pred = self.agg13(out1).squeeze(dim=-1)
        pred = pred/(u_final * i_final * t_final).squeeze(dim=-1)


        return pred

    def embeddings(self):
        
        u_final = self.user_emb.weight
        i_final = self.item_emb.weight
        t_final = self.time_emb.weight
        u_final = 1+self.softplus(self.agg5(u_final)).squeeze()
        i_final = 1+self.softplus(self.agg5(i_final)).squeeze()
        t_final = 1+self.softplus(self.agg5(t_final)).squeeze()  
        
        return u_final, i_final, t_final  

    def angle_weight(self, user_idx, item_idx, time_idx):
        
        u_final = self.user_emb(user_idx)
        i_final = self.item_emb(item_idx)
        t_final = self.time_emb(time_idx)

        u_final = 1+self.softplus(self.agg5(u_final)).squeeze()
        i_final = 1+self.softplus(self.agg5(i_final)).squeeze()
        t_final = 1+self.softplus(self.agg5(t_final)).squeeze()  
 
        cat_vec1 = torch.stack((self.user_emb(user_idx),
                                self.item_emb(item_idx),
                                self.time_emb(time_idx)
                                ), dim=2).unsqueeze(1)        

        out1 = self.relu(self.m2(self.relu(self.m1(cat_vec1)))).squeeze()
        out1 = self.relu(self.agg12(out1))
        angle_weight = torch.norm(out1, dim=-1) * torch.norm(self.agg13.weight) + torch.norm(self.agg13.bias)
        
        return self.beta*angle_weight - u_final * i_final * t_final


class SMLP(torch.nn.Module):
    def __init__(self, num_users, num_items, num_times, K, beta):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, K)
        self.item_emb = torch.nn.Embedding(num_items, K)
        self.time_emb = torch.nn.Embedding(num_times, K)      
        self.softplus = torch.nn.Softplus()
        self.K = K
        self.beta = beta
        self.agg1 = torch.nn.Linear(3*K, 3*K)
        self.agg2 = torch.nn.Linear(3*K, K)
        self.agg3 = torch.nn.Linear(K, 1)
        self.agg5 = torch.nn.Linear(K, 1)
        self.relu = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
        torch.nn.init.xavier_uniform_(self.time_emb.weight)

        torch.nn.init.xavier_uniform_(self.agg1.weight)
        torch.nn.init.xavier_uniform_(self.agg2.weight)
        torch.nn.init.xavier_uniform_(self.agg3.weight)
        torch.nn.init.xavier_uniform_(self.agg5.weight)

    def forward(self, user_idx, item_idx, time_idx, perturbed=False):

        u_final = self.user_emb(user_idx)
        i_final = self.item_emb(item_idx)
        t_final = self.time_emb(time_idx)
 
        u_final = 1+self.softplus(self.agg5(u_final)).squeeze()
        i_final = 1+self.softplus(self.agg5(i_final)).squeeze()
        t_final = 1+self.softplus(self.agg5(t_final)).squeeze()

        angle = self.angle(user_idx, item_idx, time_idx)

        return u_final * i_final * t_final,  angle

    def angle(self, user_idx, item_idx, time_idx):
        

        u_final = self.user_emb(user_idx)
        i_final = self.item_emb(item_idx)
        t_final = self.time_emb(time_idx)
     
        u_final = 1+self.softplus(self.agg5(u_final))
        i_final = 1+self.softplus(self.agg5(i_final))
        t_final = 1+self.softplus(self.agg5(t_final))
        
        concat = torch.cat((self.user_emb(user_idx), self.item_emb(item_idx), self.time_emb(time_idx)), dim=1)
        out1 = (self.relu(self.agg2(self.relu(self.agg1(concat)))))
        pred = self.agg3(out1).squeeze(dim=-1)
        pred = pred/(u_final * i_final * t_final).squeeze(dim=-1)

        return pred

        return pred.squeeze()    

    def embeddings(self):
        
        u_final = self.user_emb.weight
        i_final = self.item_emb.weight
        t_final = self.time_emb.weight
        
        u_final = 1+self.softplus(self.agg5(u_final)).squeeze()
        i_final = 1+self.softplus(self.agg5(i_final)).squeeze()
        t_final = 1+self.softplus(self.agg5(t_final)).squeeze()  
        
        return u_final, i_final, t_final  

    def angle_weight(self, user_idx, item_idx, time_idx):
        
        u_final = self.user_emb(user_idx)
        i_final = self.item_emb(item_idx)
        t_final = self.time_emb(time_idx)
        
        u_final = 1+self.softplus(self.agg5(u_final)).squeeze()
        i_final = 1+self.softplus(self.agg5(i_final)).squeeze()
        t_final = 1+self.softplus(self.agg5(t_final)).squeeze()  

        concat = torch.cat((self.user_emb(user_idx), self.item_emb(item_idx), self.time_emb(time_idx)), dim=1)
        out1 = (self.relu(self.agg2(self.relu(self.agg1(concat)))))
        angle_weight = torch.norm(out1, dim=-1) * torch.norm(self.agg3.weight) + torch.norm(self.agg3.bias)
        
        return self.beta*angle_weight - u_final * i_final * t_final

class MDMTF(torch.nn.Module):
    def __init__(self, num_modes, sizes, K):

        super().__init__()
        assert len(sizes) == num_modes, "The length of sizes list is the same as num_modes"
        
        self.num_modes = num_modes
        self.K = K
        
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(size, K // 2) for size in sizes
        ])
        self.W1s = torch.nn.ModuleList([
            torch.nn.Linear(K // 2, K) for _ in range(num_modes)
        ])
        self.W2s = torch.nn.ModuleList([
            torch.nn.Linear(K, K) for _ in range(num_modes)
        ])
        
        self.relu = torch.nn.ReLU()
        
        self.core = torch.nn.Parameter(torch.ones(*([K] * num_modes)))
        
        for emb in self.embeddings:
            torch.nn.init.xavier_uniform_(emb.weight)
        for lin in self.W1s:
            torch.nn.init.xavier_uniform_(lin.weight)
        for torch.lin in self.W2s:
            torch.nn.init.xavier_uniform_(lin.weight)
        torch.nn.init.xavier_uniform_(self.core)
        
    def forward(self, indices):
        assert len(indices) == self.num_modes, "The length of sizes list must be the same as num_modes"
        
        mode_outputs = []
        for i in range(self.num_modes):
            emb = self.embeddings[i](indices[i])
            out = self.relu(self.W1s[i](emb))
            out = self.relu(self.W2s[i](out))
            mode_outputs.append(out)  
        
        letters = string.ascii_lowercase 
        mode_letters = [letters[i+2] for i in range(self.num_modes)]
        
        einsum_str = ", ".join(f"b{letter}" for letter in mode_letters)
        einsum_str += ", " + "".join(mode_letters) + "->b"
        
        out = torch.einsum(einsum_str, *mode_outputs, self.core)
        
        return out

class SMDMTF(torch.nn.Module):
    def __init__(self, num_modes, sizes, K):
        super().__init__()
        assert len(sizes) == num_modes, "The length of sizes list must be the same as num_modes"
        
        self.num_modes = num_modes
        self.sizes = sizes
        self.K = K     

        
        self.embeds = torch.nn.ModuleList([torch.nn.Embedding(size, K // 2) for size in sizes])
        self.W1s = torch.nn.ModuleList([torch.nn.Linear(K // 2, K) for _ in range(num_modes)])
        
        self.W2s = torch.nn.ModuleList([torch.nn.Linear(K, K) for _ in range(num_modes)])

        self.softplus = torch.nn.Softplus()
        self.agg5 = torch.nn.Linear(K, 1)           
        self.relu = torch.nn.ReLU()
        
        self.core = torch.nn.Parameter(torch.ones(*([K] * num_modes)))
        
        for emb in self.embeds:
            torch.nn.init.xavier_uniform_(emb.weight)
        for lin in self.W1s:
            torch.nn.init.xavier_uniform_(lin.weight)
        for lin in self.W2s: 
            torch.nn.init.xavier_uniform_(lin.weight)
            
        torch.nn.init.xavier_uniform_(self.core)
        
    def forward(self, indices):

        assert len(indices) == self.num_modes, "."

        mode_outputs = []
        for i in range(self.num_modes):
            emb = self.embeds[i](indices[i])
            out = self.relu(self.W1s[i](emb))
            out = self.relu(self.W2s[i](out))
            mode_outputs.append(out)  
        
        embeds = [torch.norm(mode_outputs[i], dim=-1)
                for i in range(len(self.sizes))]
        
        angle = self.angle(indices)
        
        return torch.prod(torch.stack(embeds, dim=0), dim=0), angle

    def angle(self, indices):  
        
        mode_outputs = []
        for i in range(self.num_modes):
            emb = self.embeds[i](indices[i])
            out = self.relu(self.W1s[i](emb))
            out = self.relu(self.W2s[i](out))
            mode_outputs.append(out)  

        embeds = [torch.norm(mode_outputs[i], dim=-1)
                for i in range(len(self.sizes))]          
        
        letters = string.ascii_lowercase 
        mode_letters = [letters[i+2] for i in range(self.num_modes)]
        
        einsum_str = ", ".join(f"b{letter}" for letter in mode_letters)
        einsum_str += ", " + "".join(mode_letters) + "->b"

        if self.training:
            out = torch.einsum(einsum_str, *mode_outputs, self.core)/(torch.prod(torch.stack(embeds, dim=0), dim=0))
        else:
            out = torch.einsum(einsum_str, *mode_outputs, self.core)/(torch.norm(self.core, p=1) * torch.prod(torch.stack(embeds, dim=0), dim=0))

        
        return out

    def embeddings(self):

        mode_outputs = []
        for i in range(self.num_modes):
            emb = self.embeds[i].weight
            out = self.relu(self.W1s[i](emb))
            out = self.relu(self.W2s[i](out))
            mode_outputs.append(out)  
            
        embeds = [torch.norm(mode_outputs[i], dim=-1)
                for i in range(len(self.sizes))]
        
        return embeds

class MLP(torch.nn.Module):
    def __init__(self, num_users, num_items, num_times, K):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, K)
        self.item_emb = torch.nn.Embedding(num_items, K)
        self.time_emb = torch.nn.Embedding(num_times, K)

        self.agg1 = torch.nn.Linear(3*K, K)
        self.agg2 = torch.nn.Linear(K, K)
        self.agg3 = torch.nn.Linear(K, 1)        
        self.relu = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
        torch.nn.init.xavier_uniform_(self.time_emb.weight)

        torch.nn.init.xavier_uniform_(self.agg1.weight)
        torch.nn.init.xavier_uniform_(self.agg2.weight)
        torch.nn.init.xavier_uniform_(self.agg3.weight)

    def forward(self, user_idx, item_idx, time_idx, perturbed=False):

        concat = torch.cat((self.user_emb(user_idx), self.item_emb(item_idx), self.time_emb(time_idx)), dim=1)
        pred = self.agg3(self.relu(self.agg2(self.relu(self.agg1(concat)))))

        return pred
