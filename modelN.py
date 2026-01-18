import torch
import torch.nn as nn
from functools import reduce
import operator
import string
import opt_einsum as oe
import torch.nn.functional as F

class SCPN(nn.Module):
    def __init__(self, num_modes, K, eps=1e-12):
        super().__init__()
        self.K = K
        self.N = len(num_modes)
        self.eps = eps
        
        self.embs = nn.ModuleList([nn.Embedding(n, K) for n in num_modes])

        for emb in self.embs:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")

        vecs = [emb(idx) for emb, idx in zip(self.embs, idxs)]

        norms = [v.norm(dim=1) for v in vecs]

        mag = norms[0]
        for nm in norms[1:]:
            mag = mag * nm

        angle = self.angle_from_vecs(vecs)
        return mag, angle

    @torch.no_grad()
    def embeddings(self):
        return [emb.weight.norm(dim=1) for emb in self.embs]

    def angle(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")
        vecs = [emb(idx) for emb, idx in zip(self.embs, idxs)]
        return self.angle_from_vecs(vecs)

    def angle_from_vecs(self, vecs):
        nvecs = []
        for v in vecs:
            denom = v.norm(dim=1, keepdim=True).clamp_min(self.eps)
            nvecs.append(v / denom)

        prod_vec = nvecs[0]
        for nv in nvecs[1:]:
            prod_vec = prod_vec * nv

        angle = prod_vec.sum(dim=1)
        return angle

    def score(self, *idxs):
        mag, angle = self.forward(*idxs)
        return mag * angle

class STuckerN(nn.Module):
    def __init__(self, num_modes, K, eps=1e-12, batch_size_hint=256):
        super().__init__()
        self.K = K
        self.N = len(num_modes)
        self.eps = eps

        self.embs = nn.ModuleList([nn.Embedding(n, K) for n in num_modes])

        core_shape = (K,) * self.N
        self.core = nn.Parameter(torch.ones(*core_shape))

        letters = "ijklmnopqrstuvwxyzabcdefgh"
        if self.N > len(letters):
            raise ValueError("N too large for this simple einsum letter set.")

        mode_letters = letters[:self.N]
        vec_terms = [f"b{c}" for c in mode_letters]
        core_term = mode_letters
        
        self.eq = ",".join(vec_terms + [core_term]) + "->b"

        shapes = [(batch_size_hint, K)] * self.N
        shapes.append(core_shape)

        self.expr = oe.contract_expression(self.eq, *shapes, optimize='optimal')

        for emb in self.embs:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.core)

    def forward(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")

        vecs = [emb(idx) for emb, idx in zip(self.embs, idxs)]
        norms = [v.norm(dim=1) for v in vecs]

        mag = norms[0]
        for nm in norms[1:]:
            mag = mag * nm

        angle = self.angle_from_vecs(vecs, norms=norms)
        return mag, angle

    def angle(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")
        vecs = [emb(idx) for emb, idx in zip(self.embs, idxs)]
        norms = [v.norm(dim=1) for v in vecs]
        return self.angle_from_vecs(vecs, norms=norms)

    def angle_from_vecs(self, vecs, norms=None):
        if norms is None:
            norms = [v.norm(dim=1) for v in vecs]

        numerator = self.expr(*vecs, self.core, backend='torch')

        denom = norms[0].clamp_min(self.eps)
        for nm in norms[1:]:
            denom = denom * nm.clamp_min(self.eps)

        if not self.training:
            core_l1 = self.core.abs().sum().clamp_min(self.eps)
            denom = denom * core_l1

        return (numerator / denom).squeeze()

    def embeddings(self):
        return [emb.weight.norm(dim=1) for emb in self.embs]

    def score(self, *idxs):
        mag, angle = self.forward(*idxs)
        return mag * angle

class SCostCoN(nn.Module):
    def __init__(self, num_modes, K, beta, eps=1e-12):
        super().__init__()
        self.K = K
        self.N = len(num_modes)
        self.beta = beta
        self.eps = eps

        self.embs = nn.ModuleList([nn.Embedding(n, K) for n in num_modes])

        self.scale = nn.Linear(K, 1)

        self.m1 = nn.Conv2d(1, K, (1, self.N), stride=1)
        self.m2 = nn.Conv2d(K, K, (K, 1), stride=1)

        self.agg12 = nn.Linear(K, K)
        self.agg13 = nn.Linear(K, 1)

        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        for emb in self.embs:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.m1.weight)
        nn.init.xavier_uniform_(self.m2.weight)
        nn.init.xavier_uniform_(self.agg12.weight)
        nn.init.xavier_uniform_(self.agg13.weight)

        init_scale = 0.1
        torch.nn.init.uniform_(self.scale.weight, -init_scale, init_scale)

        if self.scale.bias is not None:
            torch.nn.init.zeros_(self.scale.bias)

    def _mode_vecs(self, idxs):
        return [emb(idx) for emb, idx in zip(self.embs, idxs)]

    def _mode_scales(self, vecs, squeeze=True):
        outs = []
        for v in vecs:
            s = 1.0 + self.softplus(self.scale(v))
            outs.append(s.squeeze(-1) if squeeze else s)
                        
        return outs

    def forward(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")

        vecs = self._mode_vecs(idxs)
        scales = self._mode_scales(vecs, squeeze=True)

        mag = scales[0]
        for s in scales[1:]:
            mag = mag * s

        angle = self.angle_from_vecs(vecs, mag=mag)
        return mag, angle

    def angle(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")
        vecs = self._mode_vecs(idxs)
        scales = self._mode_scales(vecs, squeeze=True)
        mag = scales[0]
        for s in scales[1:]:
            mag = mag * s
        return self.angle_from_vecs(vecs, mag=mag)

    def angle_from_vecs(self, vecs, mag):
        cat = torch.stack(vecs, dim=2)
        cat = cat.unsqueeze(1)

        out = self.relu(self.m1(cat))
        out = self.relu(self.m2(out))
        out = out.squeeze(-1).squeeze(-1)

        out = self.relu(self.agg12(out))
        pred = self.agg13(out).squeeze(-1)

        angle = pred / mag.clamp_min(self.eps)
        return angle

    def embeddings(self):
        mags = []
        for emb in self.embs:
            m = 1.0 + self.softplus(self.scale(emb.weight)).squeeze(-1)
            mags.append(m)
        return mags

    def angle_weight(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")

        vecs = self._mode_vecs(idxs)
        scales = self._mode_scales(vecs, squeeze=True)
        mag = scales[0]
        for s in scales[1:]:
            mag = mag * s

        cat = torch.stack(vecs, dim=2).unsqueeze(1)
        out = self.relu(self.m1(cat))
        out = self.relu(self.m2(out)).squeeze(-1).squeeze(-1)
        out = self.relu(self.agg12(out))
        
        angle_w = torch.norm(out, dim=-1) * torch.norm(self.agg13.weight) + torch.norm(self.agg13.bias)
        return self.beta * angle_w - mag


class SMLPN(nn.Module):
    def __init__(self, num_modes, K, beta, eps=1e-12):
        super().__init__()
        self.K = K
        self.N = len(num_modes)
        self.beta = beta
        self.eps = eps

        self.embs = nn.ModuleList([nn.Embedding(n, K) for n in num_modes])

        self.scale = nn.Linear(K, 1)

        in_dim = self.N * K
        self.agg1 = nn.Linear(in_dim, in_dim)
        self.agg2 = nn.Linear(in_dim, K)
        self.agg3 = nn.Linear(K, 1)

        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        for emb in self.embs:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.agg1.weight)
        nn.init.xavier_uniform_(self.agg2.weight)
        nn.init.xavier_uniform_(self.agg3.weight)

    def _vecs(self, idxs):
        return [emb(idx) for emb, idx in zip(self.embs, idxs)]

    def _scales(self, vecs, squeeze=True):
        outs = []
        for v in vecs:
            s = 1.0 + self.softplus(self.scale(v))
            outs.append(s.squeeze(-1) if squeeze else s)
        return outs

    def forward(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")

        vecs = self._vecs(idxs)
        scales = self._scales(vecs, squeeze=True)

        mag = scales[0]
        for s in scales[1:]:
            mag = mag * s

        angle = self.angle_from_vecs(vecs, mag=mag)
        return mag, angle

    def angle(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")
        vecs = self._vecs(idxs)
        scales = self._scales(vecs, squeeze=True)
        mag = scales[0]
        for s in scales[1:]:
            mag = mag * s
        return self.angle_from_vecs(vecs, mag=mag)

    def angle_from_vecs(self, vecs, mag):
        concat = torch.cat(vecs, dim=1)

        out = self.relu(self.agg1(concat))
        out = self.relu(self.agg2(out))
        pred = self.agg3(out).squeeze(-1)

        return pred / mag.clamp_min(self.eps)

    def embeddings(self):
        mags = []
        for emb in self.embs:
            m = 1.0 + self.softplus(self.scale(emb.weight)).squeeze(-1)
            mags.append(m)
        return mags

    def angle_weight(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")

        vecs = self._vecs(idxs)
        scales = self._scales(vecs, squeeze=True)
        mag = scales[0]
        for s in scales[1:]:
            mag = mag * s

        concat = torch.cat(vecs, dim=1)
        out = self.relu(self.agg1(concat))
        out = self.relu(self.agg2(out))

        angle_w = torch.norm(out, dim=-1) * torch.norm(self.agg3.weight) + torch.norm(self.agg3.bias)
        return self.beta * angle_w - mag

class SMDMTF(nn.Module):
    def __init__(self, num_modes, K, eps=1e-12):
        super().__init__()
        self.num_modes = num_modes
        self.N = len(num_modes)
        self.K = K
        self.eps = eps
        
        self.embs = nn.ModuleList([nn.Embedding(n, K // 2) for n in num_modes])
        
        self.W1s = nn.ModuleList([nn.Linear(K // 2, K) for _ in range(self.N)])
        self.W2s = nn.ModuleList([nn.Linear(K, K) for _ in range(self.N)])
        
        self.relu = nn.ReLU()

        self.core = nn.Parameter(torch.ones(*([K] * self.N)))

        for emb in self.embs:
            nn.init.xavier_uniform_(emb.weight)
        for lin in self.W1s:
            nn.init.xavier_uniform_(lin.weight)
        for lin in self.W2s:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.core)

    def get_projected_vecs(self, idxs):
        vecs = []
        for i, idx in enumerate(idxs):
            e = self.embs[i](idx)
            h = self.relu(self.W1s[i](e))
            v = (self.W2s[i](h))
            vecs.append(v)
        return vecs

    def embeddings(self):
        vecs = []
        for i, emb in enumerate(self.embs):
            h = self.relu(self.W1s[i](emb.weight))
            v = (self.W2s[i](h))
            vecs.append(v)
            
        mags = [v.norm(dim=1) for v in vecs]
        return mags        

    def forward(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")

        vecs = self.get_projected_vecs(idxs)

        norms = [v.norm(dim=1) for v in vecs]

        mag = norms[0]
        for nm in norms[1:]:
            mag = mag * nm

        angle = self.angle_from_vecs(vecs)

        return mag, angle

    def angle(self, *idxs):
        if len(idxs) != self.N:
            raise ValueError(f"Expected {self.N} index tensors, got {len(idxs)}")
        vecs = self.get_projected_vecs(idxs)
        return self.angle_from_vecs(vecs)

    def angle_from_vecs(self, vecs):
        norms = [v.norm(dim=1) for v in vecs]        
        
        letters = string.ascii_lowercase 
        
        batch_char = 'a'
        mode_chars = [letters[i+1] for i in range(self.N)]
        
        input_strs = [f"{batch_char}{m_char}" for m_char in mode_chars]
        core_str = "".join(mode_chars)
        einsum_cmd = f"{', '.join(input_strs)}, {core_str} -> {batch_char}"
        
        numerator = torch.einsum(einsum_cmd, *vecs, self.core)

        denom = norms[0].clamp_min(self.eps)
        for nm in norms[1:]:
            denom = denom * nm.clamp_min(self.eps)

        if not self.training:
            core_l1 = self.core.abs().sum().clamp_min(self.eps)
            denom = denom * core_l1
        
        return (numerator / denom).squeeze()

    def score(self, *idxs):
        mag, angle = self.forward(*idxs)
        return mag * angle        

class MLP(nn.Module):
    def __init__(self, dims, act='ReLU'):
        super(MLP, self).__init__()
        layers = []
        for i in range(0, len(dims)-2):
            in_dim, out_dim = dims[i], dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            if act != '':
                if act == 'ReLU':
                    layers.append(nn.ReLU())
                elif act == 'Softplus':
                    layers.append(nn.Softplus())
                elif act == 'Tanh':
                    layers.append(nn.Tanh())

        in_dim, out_dim = dims[-2], dims[-1]
        layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SNeAT(nn.Module):
    def __init__(self, num_modes, K, beta=1.0, dropout=0.0, dropout2=0.0, act='ReLU'):
        super(SNeAT, self).__init__()

        self.sizes = num_modes
        self.rank = K
        self.beta = beta
        
        self.layer_dims = [len(num_modes), K, 1]
        self.act = act
        self.depth = len(self.layer_dims)

        self.softplus = torch.nn.Softplus()
        self.agg5 = torch.nn.Linear(self.rank, 1)

        self.embeds = nn.ModuleList([
            nn.Embedding(self.sizes[i], self.rank) for i in range(len(self.sizes))
        ])

        self.dropout_layer = nn.Dropout(p=dropout)
        self.dropout2_layer = nn.Dropout(p=dropout2)

        self.make_mlps()
        self._initialize()

        init_scale = 0.1
        
        torch.nn.init.uniform_(self.agg5.weight, -init_scale, init_scale)

        if self.agg5.bias is not None:
            torch.nn.init.zeros_(self.agg5.bias)        

    def _initialize(self):        
        for i in range(len(self.embeds)):
            nn.init.uniform_(self.embeds[i].weight.data)

    def make_mlps(self):
        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()

        for i in range(self.depth - 1):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i+1]
            
            w = nn.Parameter(torch.empty(self.rank, out_dim, in_dim))
            b = nn.Parameter(torch.empty(self.rank, out_dim))
            
            nn.init.xavier_uniform_(w)
            nn.init.zeros_(b)
            
            self.weight.append(w)
            self.bias.append(b)

    def _normalize(self):
        for i in range(len(self.embeds)):
            self.embeds[i].weight.data = F.normalize(self.embeds[i].weight.data)
            
    def calc(self, x):
        angle_weight = 0
        
        for d in range(self.depth - 1):
            if d == self.depth - 2:
                w_norm = torch.norm(self.weight[d], dim=-1) 
                x_norm = torch.norm(x, dim=-1)
                
                aw = (w_norm.unsqueeze(1) * x_norm.unsqueeze(-1)) + torch.norm(self.bias[d], dim=-1).unsqueeze(0).unsqueeze(1)
                
                angle_weight = aw.sum(dim=-1).sum(dim=0)

            x = x @ self.weight[d].permute(0, 2, 1)
            x = x + self.bias[d].unsqueeze(1)

            if d != self.depth - 2:
                if self.act == 'ReLU':
                    x = torch.relu(x)
                elif self.act == 'Softplus':
                    x = F.softplus(x)
                elif self.act == 'Tanh':
                    x = torch.tanh(x)
                
                x = self.dropout2_layer(x)
                
        return x, angle_weight
    
    def forward(self, *idxs):
        if len(idxs) != len(self.sizes):
             raise ValueError(f"Expected {len(self.sizes)} indices, got {len(idxs)}")

        embeds = [1 + self.softplus(self.agg5(self.embeds[i](idxs[i]))).squeeze(dim=-1)
                for i in range(len(self.sizes))]
        
        angle = self.angle(*idxs)
        
        return torch.prod(torch.stack(embeds, dim=0), dim=0), angle

    def angle(self, *idxs):
        embeds = [1 + self.softplus(self.agg5(self.embeds[i](idxs[i]))).squeeze(dim=-1)
                for i in range(len(self.sizes))]

        embeds2 = [self.embeds[i](idxs[i]).permute(1, 0).unsqueeze(-1)
                for i in range(len(self.sizes))]        
        
        x = torch.cat(embeds2, dim=-1)
        x = F.normalize(x, dim=-1)

        x, _ = self.calc(x)
        
        x = self.dropout_layer(x)
        
        x = x.sum(0).view(-1)
        
        pred = x / torch.prod(torch.stack(embeds, dim=0), dim=0)
        
        return pred    
        
    def embeddings(self):
        embeds = [1 + self.softplus(self.agg5(self.embeds[i].weight)).squeeze(dim=-1)
                for i in range(len(self.sizes))]
        return embeds

    def angle_weight(self, *idxs):
        embeds = [1 + self.softplus(self.agg5(self.embeds[i](idxs[i]))).squeeze(dim=-1)
                for i in range(len(self.sizes))]
        
        embeds2 = [self.embeds[i](idxs[i]).permute(1, 0).unsqueeze(-1)
                for i in range(len(self.sizes))]          
        
        x = torch.cat(embeds2, dim=-1)
        x = F.normalize(x, dim=-1)

        _, angle_weight_val = self.calc(x)
        
        return self.beta * angle_weight_val - torch.prod(torch.stack(embeds, dim=0), dim=0)        


class Ablation_MLP(nn.Module):
    def __init__(self, num_modes, K, beta=1.0):
        super().__init__()
        self.num_modes = len(num_modes)
        self.K = K
        
        self.embeds = nn.ModuleList([
            nn.Embedding(dim, K) for dim in num_modes
        ])
        
        self.agg5 = nn.Linear(K, 1)
        
        self.softplus = nn.Softplus()
        
        for i in range(len(self.embeds)):
            nn.init.uniform_(self.embeds[i].weight.data)
        nn.init.xavier_uniform_(self.agg5.weight)

    def forward(self, *indices):
        processed_embs = []
        
        for i, idx in enumerate(indices):
            emb = self.embeds[i](idx)
            
            val = 1.0 + self.softplus(self.agg5(emb)).squeeze()
            processed_embs.append(val)

        out = processed_embs[0]
        for i in range(1, self.num_modes):
            out = out * processed_embs[i]

        angle_val = self.angle(*indices)

        return out, angle_val

    def embeddings(self):
        embeds = [1 + self.softplus(self.agg5(self.embeds[i].weight)).squeeze(dim=-1)
                for i in range(self.num_modes)]
        return embeds
    
    def angle(self, *indices):
        batch_size = indices[0].shape[0]
        device = indices[0].device
        return torch.ones(batch_size, device=device)

    def angle_weight(self, *indices):
        batch_size = indices[0].shape[0]
        device = indices[0].device
        return torch.zeros(batch_size, device=device)


class Ablation_Norm(nn.Module):
    def __init__(self, num_modes, K, beta=1.0):
        super().__init__()
        self.num_modes = len(num_modes)
        self.K = K
        
        self.embeds = nn.ModuleList([
            nn.Embedding(dim, K) for dim in num_modes
        ])
        
        for i in range(len(self.embeds)):
            nn.init.uniform_(self.embeds[i].weight.data)

    def forward(self, *indices):
        norms = []
        for i, idx in enumerate(indices):
            emb = self.embeds[i](idx)
            norm_val = torch.norm(emb, dim=1)
            norms.append(norm_val)

        out = norms[0]
        for i in range(1, self.num_modes):
            out = out * norms[i]

        angle_val = self.angle(*indices)

        return out, angle_val

    def embeddings(self):
        embeds = [torch.norm(self.embeds[i].weight, dim=1)
                for i in range(self.num_modes)]
        return embeds    

    def angle(self, *indices):
        batch_size = indices[0].shape[0]
        device = indices[0].device
        return torch.ones(batch_size, device=device)

    def angle_weight(self, *indices):
        batch_size = indices[0].shape[0]
        device = indices[0].device
        return torch.zeros(batch_size, device=device)
    
    def get_all_weights_norm(self):
        processed_weights = []
        for emb in self.embeddings:
            w = emb.weight
            w_norm = torch.norm(w, dim=1)
            processed_weights.append(w_norm)
        return processed_weights