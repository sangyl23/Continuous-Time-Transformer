import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils import TriangularCausalMask, ProbMask
from models.CT_linear import ODELinear, InterpLinear

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, his_timeslot, label_pre_timeslot, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
class CTAttention(nn.Module):
    def __init__(self, mask_flag = True, factor = 5, scale = None, attention_dropout = 0.1, output_attention = False, 
                 odefunc_q = None,
                 odefunc_k = None,
                 odefunc_v = None,
                 odeint_rtol = 1e-3, 
                 odeint_atol = 1e-4, 
                 method = 'rk4'):
        super(CTAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        self.CT_Q = InterpLinear() if odefunc_q is None else ODELinear(odefunc_q, odeint_rtol, odeint_atol, method)
        self.CT_K = InterpLinear() if odefunc_k is None else ODELinear(odefunc_k, odeint_rtol, odeint_atol, method)
        self.CT_V = InterpLinear() if odefunc_v is None else ODELinear(odefunc_v, odeint_rtol, odeint_atol, method)
        
    def forward(self, queries, keys, values, his_timeslot, label_pre_timeslot, attn_mask):
        # B: Batch size
        # L: his_len for encoder, label_len + pre_len for decoder
        # H: head
        # E: Feature dim
        B, L, H, E = queries.shape
        scale = self.scale or 1./sqrt(E)
        
        CT_queries = self.CT_Q(x = queries, timeslot = his_timeslot) # (B, L, L, H, E, 2)
        CT_keys = self.CT_K(x = keys, timeslot = his_timeslot) # (B, L, L, H, E, 2)
        CT_values = self.CT_V(x = values, timeslot = his_timeslot) # (B, L, L, H, E, 2)
        
        CT_queries = CT_queries.permute(0, 3, 1, 2, 5, 4) # (B, H, L, L, 2, E)
        CT_keys = CT_keys.permute(0, 3, 1, 2, 5, 4) # (B, H, L, L, 2, E)
        CT_values = CT_values.permute(0, 3, 1, 2, 5, 4) # (B, H, L, L, 2, E)
                     
        # scores = torch.einsum("blhe,bshe->bhls", queries, keys)        
        # scores = (CT_queries * CT_keys.flip(dims = [-2])).sum(dim = -1).sum(dim = -1) # (B, H, L, L) 
        scores = (0.5 * CT_queries * CT_keys.transpose(2, 3)).sum(dim = -1).sum(dim = -1) # (B, H, L, L) 
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim = -1)) # (B, H, L, L)
        
        # V = (A.unsqueeze(-1) * CT_values.transpose(2, 3).sum(dim = -2)).sum(dim = -2) # (B, H, L, E)
        V = (A.unsqueeze(-1) * 0.5 * CT_values.sum(dim = -2)).sum(dim = -2) # (B, H, L, E)
        
        V = V.permute(0, 2, 1, 3) # (B, L, H, E)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).double().to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.view(B, H, L_Q, -1)
        keys = keys.view(B, H, L_K, -1)
        values = values.view(B, H, L_K, -1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn

class Prob_CTAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 odefunc_q = None,
                 odefunc_k = None,
                 odefunc_v = None,
                 odeint_rtol = 1e-3, 
                 odeint_atol = 1e-4, 
                 method = 'rk4'):
        super(Prob_CTAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # self.CT_Q = InterpLinear() if odefunc_q is None else ODELinear(odefunc_q, odeint_rtol, odeint_atol, method)
        # self.CT_K = InterpLinear() if odefunc_k is None else ODELinear(odefunc_k, odeint_rtol, odeint_atol, method)
        ## only support (q, k, v) = (interp, interp, ode) since time slot is not consistent for each B and H in sparse attention, which is computational heavy for ODE
        # self.CT_Q = InterpLinear() 
        # self.CT_K = InterpLinear() 
        self.CT_V = InterpLinear() if odefunc_v is None else ODELinear(odefunc_v, odeint_rtol, odeint_atol, method)

    def _prob_QK(self, Q, K, sample_k, n_top, his_timeslot): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        # Q_reduce = Q[torch.arange(B)[:, None, None],
        #              torch.arange(H)[None, :, None],
        #              M_top, :] # factor*ln(L_q)
        # Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k
        
        ########################################################################
        ## continuous-time prob QK
        # Q: [B, H, L_Q, E], M_top: [B, H, n_top]
        
        # 初始化输出张量
        CT_queries = torch.zeros(B, H, n_top, L_K, E, 2, device = Q.device, dtype = Q.dtype)
        CT_keys = torch.zeros(B, H, L_K, n_top, E, 2, device = Q.device, dtype = Q.dtype)
        
        # 获取被选中的 Query 向量: [B, H, n_top, E]
        Q_selected = Q.gather(2, M_top.unsqueeze(-1).expand(-1, -1, -1, E))  # [B, H, n_top, E]
        K_selected = K.gather(2, M_top.unsqueeze(-1).expand(-1, -1, -1, E))  # [B, H, n_top, E]
        
        # 获取所有 Query 向量并扩展为 [B, H, 1, L_K, E]
        Q_all = Q.unsqueeze(2).expand(-1, -1, n_top, -1, -1)  # [B, H, n_top, L_K, E]
        K_all = K.unsqueeze(3).expand(-1, -1, -1, n_top, -1)  # [B, H, L_K, n_top, E]
        
        # 填充 CT_queries
        CT_queries[..., 0] = Q_selected.unsqueeze(3).expand(-1, -1, -1, L_K, -1)  # [B, H, n_top, L_K, E]
        CT_queries[..., 1] = Q_all  # [B, H, n_top, L_K, E]
        
        # 填充 CT_keys
        CT_keys[..., 0] = K_all  # [B, H, L_K, n_top, E]
        CT_keys[..., 1] = K_selected.unsqueeze(2).expand(-1, -1, L_K, n_top, -1)  # [B, H, L_K, n_top, E]

        # CT_queries = self.CT_Q(x = Q.permute(0, 2, 1, 3), timeslot = his_timeslot)  # (B, L, L, H, E, 2)
        # CT_keys = self.CT_K(x = K.permute(0, 2, 1, 3), timeslot = his_timeslot) # (B, L, L, H, E, 2)
        
        CT_queries = CT_queries.permute(0, 1, 2, 3, 5, 4) # (B, H, n_top, L_K, 2, E)
        CT_keys = CT_keys.permute(0, 1, 2, 3, 5, 4) # (B, H, L_K, n_top, 2, E)
        
        scores = (0.5 * CT_queries * CT_keys.transpose(2, 3)).sum(dim = -1).sum(dim = -1) # (B, H, n_top, L_K)        
        ########################################################################
        
        return scores, M_top

    def _get_initial_context(self, CT_values, L_K):
        # B, H, L_V, D = V.shape
        # if not self.mask_flag:
        #     # V_sum = V.sum(dim=-2)
        #     V_sum = V.mean(dim=-2)
        #     contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        # else: # use mask
        #     assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
        #     contex = V.cumsum(dim=-2)
        CT_values = 0.5 * CT_values.sum(dim = -2) # (B, H, L, L, E)
        
        base = torch.tril(torch.ones(L_K, L_K, device = CT_values.device, dtype = torch.float32))  # [L, L]
        mask = base.unsqueeze(0).unsqueeze(0).expand(CT_values.shape[0], CT_values.shape[1], L_K, L_K)  # [B, H, L, L]
        
        contex = (mask.unsqueeze(dim = -1) * CT_values).sum(dim = -2) # (B, H, L, E)
        
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, _, _, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)
        
        V_selected = V[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index,]
        
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = (attn.unsqueeze(-1) * 0.5 * V_selected.sum(dim = -2)).sum(dim = -2)
        context_in = context_in.permute(0, 2, 1, 3) # (B, L, H, E)        
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).double().to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, his_timeslot, label_pre_timeslot, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.view(B, H, L_Q, -1)
        keys = keys.view(B, H, L_K, -1)
        values = values.view(B, H, L_K, -1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u, his_timeslot = his_timeslot) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        
        ########################################################################
        ## continuous-time prob QK
        CT_values = self.CT_V(x = values.permute(0, 2, 1, 3), timeslot = his_timeslot) # (B, L, L, H, E, 2)
        CT_values = CT_values.permute(0, 3, 1, 2, 5, 4) # (B, H, L, L, 2, E)
        ########################################################################
            
        # get the context
        context = self._get_initial_context(CT_values, L_K)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, CT_values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, his_timeslot, label_pre_timeslot, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # print(L, S)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            his_timeslot,
            label_pre_timeslot,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn