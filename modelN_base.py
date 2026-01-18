import torch
import torch.nn as nn
import torch.nn.functional as F
import opt_einsum as oe

class CP(nn.Module):
    def __init__(self, dims, K):
        super().__init__()
        self.num_modes = len(dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, K) for dim in dims
        ])
        
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, indices):
        batch_size = indices.shape[0]
        element_wise_prod = self.embeddings[0](indices[:, 0])
        
        for i in range(1, self.num_modes):
            element_wise_prod = element_wise_prod * self.embeddings[i](indices[:, i])
            
        out = element_wise_prod.sum(dim=1)
        return out

class Tucker(nn.Module):
    def __init__(self, dims, K, batch_size_hint=32):
        super().__init__()
        self.num_modes = len(dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, K) for dim in dims
        ])
        
        self.core_dims = [K] * self.num_modes
        self.core = nn.Parameter(torch.ones(*self.core_dims))
        
        batch_char = 'z'
        mode_chars = [chr(ord('a') + i) for i in range(self.num_modes)]
        inputs_str = ','.join([f"{batch_char}{c}" for c in mode_chars])
        core_str = ''.join(mode_chars)
        equation = f"{inputs_str},{core_str}->{batch_char}"
        
        shapes = [(batch_size_hint, K)] * self.num_modes 
        shapes.append(tuple(self.core_dims))
        
        self.expr = oe.contract_expression(equation, *shapes, optimize='optimal')

        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.core)

    def forward(self, indices):
        embeddings_list = [self.embeddings[i](indices[:, i]) for i in range(self.num_modes)]
        
        out = self.expr(*embeddings_list, self.core, backend='torch')
        
        return out

class CostCo(nn.Module):
    def __init__(self, dims, K):
        super().__init__()
        self.num_modes = len(dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, K) for dim in dims
        ])
        
        self.m1 = nn.Conv2d(1, K, kernel_size=(1, self.num_modes), stride=1)
        self.m2 = nn.Conv2d(K, K, kernel_size=(K, 1), stride=1)
        
        self.agg1 = nn.Linear(K, K)
        self.agg2 = nn.Linear(K, 1)
        self.relu = nn.ReLU()
        
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.m1.weight)
        nn.init.xavier_uniform_(self.m2.weight)
        nn.init.xavier_uniform_(self.agg1.weight)
        nn.init.xavier_uniform_(self.agg2.weight)

    def forward(self, indices):
        emb_list = [self.embeddings[i](indices[:, i]) for i in range(self.num_modes)]
        
        stack = torch.stack(emb_list, dim=2)
        
        stack = stack.unsqueeze(1)
        
        out1 = self.relu(self.m1(stack))
        
        out1 = self.relu(self.m2(out1))
        
        out1 = out1.squeeze() 
        
        if out1.dim() == 1:
            out1 = out1.unsqueeze(0)

        out1 = self.relu(self.agg1(out1))
        pred = self.agg2(out1)

        return pred

class MLP(nn.Module):
    def __init__(self, dims, K):
        super().__init__()
        self.num_modes = len(dims)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, K) for dim in dims
        ])

        input_dim = self.num_modes * K

        self.agg1 = nn.Linear(input_dim, K)
        self.agg2 = nn.Linear(K, K)
        self.agg3 = nn.Linear(K, 1)
        
        self.relu = nn.ReLU()

        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
        
        nn.init.xavier_uniform_(self.agg1.weight)
        nn.init.xavier_uniform_(self.agg2.weight)
        nn.init.xavier_uniform_(self.agg3.weight)

    def forward(self, indices):
        emb_list = [self.embeddings[i](indices[:, i]) for i in range(self.num_modes)]

        concat = torch.cat(emb_list, dim=1)

        out = self.relu(self.agg1(concat))
        out = self.relu(self.agg2(out))
        pred = self.agg3(out)

        return pred

class MDMTF(nn.Module):
    def __init__(self, dims, K):
        super().__init__()
        self.num_modes = len(dims)
        self.K = K

        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, K // 2) for dim in dims
        ])
        
        self.W1s = nn.ModuleList([
            nn.Linear(K // 2, K) for _ in range(self.num_modes)
        ])
        self.W2s = nn.ModuleList([
            nn.Linear(K, K) for _ in range(self.num_modes)
        ])
        
        self.relu = nn.ReLU()
        

        self.core = nn.Parameter(torch.ones(*([K] * self.num_modes)))
        
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
        for lin in self.W1s:
            nn.init.xavier_uniform_(lin.weight)
        for lin in self.W2s: 
            nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.core)

    def forward(self, indices):
        mode_outputs = []
        
        for i in range(self.num_modes):
            idx = indices[:, i]
            emb = self.embeddings[i](idx)
            
            out = self.relu(self.W1s[i](emb))
            out = (self.W2s[i](out))
            
            mode_outputs.append(out)

        batch_char = 'z'
        mode_chars = [chr(ord('a') + i) for i in range(self.num_modes)]
        
        inputs_str = ','.join([f"{batch_char}{c}" for c in mode_chars])
        
        core_str = ''.join(mode_chars)
        
        einsum_eq = f"{inputs_str},{core_str}->{batch_char}"
        
        out = torch.einsum(einsum_eq, *mode_outputs, self.core)
        
        return out        

class NeAT(nn.Module):
    def __init__(self, dims, K, hidden_dims=[64, 32], dropout=0.1):

        super().__init__()
        self.num_modes = len(dims)
        self.K = K
        self.dropout_rate = dropout
        self.depth = len(hidden_dims) + 1

        self.embeds = nn.ModuleList([
            nn.Embedding(dim, K) for dim in dims
        ])

        layer_sizes = [self.num_modes] + hidden_dims + [1]
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            
            w = nn.Parameter(torch.Tensor(K, out_dim, in_dim))
            b = nn.Parameter(torch.Tensor(K, out_dim))
            
            nn.init.xavier_uniform_(w)
            nn.init.zeros_(b)
            
            self.weights.append(w)
            self.biases.append(b)

        self.dropout = nn.Dropout(p=dropout)
        
        self._initialize_embeds()

    def _initialize_embeds(self):
        for emb in self.embeds:
            nn.init.uniform_(emb.weight.data)

    def forward(self, indices):

        embed_list = []
        for i in range(self.num_modes):
            e = self.embeds[i](indices[:, i])
            e = e.permute(1, 0).unsqueeze(-1)
            embed_list.append(e)
        
        x = torch.cat(embed_list, dim=-1)
        
        x = F.normalize(x, dim=-1)


        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            
            x = torch.matmul(x, w.transpose(1, 2))
            
            x = x + b.unsqueeze(1)
            
            if i < len(self.weights) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        x = x.squeeze(-1)
        
        out = x.sum(dim=0)
        
        return out
