import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class local_policy_att(nn.Module):
    def __init__(self, model_params, idx=0):
        super().__init__()
        self.emb_dim = model_params['local_att_hidden_dim']
        self.head_num = model_params['local_att_head_num']
        self.qkv_dim = model_params['local_att_qkv_dim']
        self.model_params = model_params
        self.local_size = model_params['local_size'][idx]
        if model_params['demand']:
            self.init_emb = nn.Linear(3, self.emb_dim)
        else:
            self.init_emb = nn.Linear(2, self.emb_dim)
        self.cur_token_emb = nn.Parameter(torch.Tensor(self.emb_dim))
        self.cur_token_emb.data.uniform_(-1, 1)
        self.Wq = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.emb_dim)

        # For positional encoding
        num_timescales = self.emb_dim // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))
        self.inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)

    def get_position_encoding(self, x):
        self.inv_timescales = self.inv_timescales.to(x.device)

        max_length = x.size()[2]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.emb_dim % 2))
        signal = signal.view(1, max_length, self.emb_dim)
        return signal

    def forward(self, theta, dist, xy, norm_demand=None, ninf_mask=None):
        # theta shape: (batch, multi, problem)
        # dist shape: (batch, multi, problem)

        valid_nodes = dist.shape[2]
        multi_width = dist.shape[1]
        # Unselected neighbor nodes except depot (depot will be added below)
        mask = ninf_mask.clone()
        # mask depot
        mask[:, :, 0] = float('-inf')
        dist -= mask
        valid_nodes -= mask.isinf().sum(-1).min()

        if self.local_size > valid_nodes:
            local_size = valid_nodes
        else:
            local_size = self.local_size
        
        # Zero idx for depot
        depot_idx = torch.zeros(dist.shape[0], dist.shape[1], 1, device=dist.device).long()

        if local_size > 0:
            # Topk except depot, but the idx should add 1 for alignment
            dist_, idx = dist[:, :, 1:].topk(local_size, dim=-1, largest=False)
            # norm factor is computed before depot is added
            if dist_.isinf().any():
                dist_[dist_.isinf()] = 0.
            norm_idx = dist_.max(-1)[0] != 0
            norm_fac = dist_[norm_idx].max(-1)[0].unsqueeze(-1)

            idx += 1
            # shape: (batch, multi, local)
            # Add depot idx
            idx = torch.cat((depot_idx, idx), dim=-1)
        else:
            # No other nodes can be selected except depot
            idx = depot_idx
            norm_idx = None

        sorted_dist = torch.take_along_dim(dist, idx, dim=-1)
        sorted_theta = torch.take_along_dim(theta, idx, dim=-1)
        sorted_demand = torch.take_along_dim(norm_demand, idx, dim=-1)
        sorted_mask = torch.take_along_dim(ninf_mask, idx, dim=-1)
        # shape: (batch, multi, local)
        if self.model_params['euclidean'] == True:
            sorted_x = torch.take_along_dim(xy[:, :, :, 0], idx, dim=-1)
            sorted_y = torch.take_along_dim(xy[:, :, :, 1], idx, dim=-1)

        # Padding 0
        # Check if there are some dims that require padding
        if sorted_dist.isinf().any():
            sorted_theta[sorted_dist.isinf()] = 0.
            sorted_demand[sorted_dist.isinf()] = 0.
            if self.model_params['euclidean'] == True:
                sorted_x[sorted_dist.isinf()] = 0.
                sorted_y[sorted_dist.isinf()] = 0.
            sorted_dist[sorted_dist.isinf()] = 0.
        
        if norm_idx is None:
            norm_idx = sorted_dist.max(-1)[0] != 0
            sorted_dist[norm_idx] = sorted_dist[norm_idx] / sorted_dist[norm_idx].max(-1)[0].unsqueeze(-1)
            if self.model_params['euclidean'] == True:
                sorted_x[norm_idx] = sorted_x[norm_idx] / sorted_dist[norm_idx].max(-1)[0].unsqueeze(-1)
                sorted_y[norm_idx] = sorted_y[norm_idx] / sorted_dist[norm_idx].max(-1)[0].unsqueeze(-1)
        else:
            sorted_dist[norm_idx] = sorted_dist[norm_idx] / norm_fac
            if self.model_params['euclidean'] == True:
                sorted_x[norm_idx] = sorted_x[norm_idx] / norm_fac
                sorted_y[norm_idx] = sorted_y[norm_idx] / norm_fac

        if self.model_params['euclidean'] == True:
            sorted_dist_theta = torch.cat((sorted_x[:, :, :, None], sorted_y[:, :, :, None]), dim=-1)
        else:
            sorted_dist_theta = torch.cat((sorted_dist[:, :, :, None], sorted_theta[:, :, :, None]), dim=-1)
        # shape: (batch, multi, local, 2)
        if self.model_params['demand']:
            sorted_input = torch.cat((sorted_dist_theta, sorted_demand[:, :, :, None]), dim=-1)
            # shape: (batch, multi, local, 3)
        else:
            sorted_input = sorted_dist_theta
        
        cur_token = self.cur_token_emb[None, None, :].expand(dist.shape[0], dist.shape[1], self.emb_dim)
        # shape: (batch, multi, emb)

        if self.model_params['positional']:
            # Positional embedding
            signal = self.get_position_encoding(sorted_input)[:, None, :, :].expand(-1, multi_width, -1, -1)
            # print(signal.shape)
            init_k = self.init_emb(sorted_input) + signal
            # init_k = self.init_emb(sorted_input)
        else:
            init_k = self.init_emb(sorted_input)
            # shape: (batch, multi, local, emb) 

        q = reshape_by_heads(self.Wq(cur_token), head_num=self.head_num).unsqueeze(3)
        # shape: (batch, head_num, multi, 1, qkv_dim)
        k = reshape_by_heads(self.Wk(init_k), head_num=self.head_num)
        # shape: (batch, head_num, multi, local, qkv_dim)
        v = reshape_by_heads(self.Wv(init_k), head_num=self.head_num)
        # shape: (batch, head_num, multi, local, qkv_dim)

        out_concat = multi_head_attention(q, k, v, rank3_ninf_mask=sorted_mask)
        # out_concat = multi_head_attention(q, k, v)
        # shape: (batch, multi, head_num * qkv_dim)
        
        mh_atten_out = self.multi_head_combine(out_concat).unsqueeze(2)
        # shape: (batch, multi, 1, emb)

        score = torch.matmul(mh_atten_out, init_k.transpose(2, 3)).squeeze(2)
        # shape: (batch, multi, local)
        
        sqrt_emb_dim = self.emb_dim ** 0.5
        score_scaled = score / sqrt_emb_dim

        out = score_scaled
        # print(sorted_mask[out.isnan().any(-1)])
        out_mat = torch.zeros(dist.shape, device=dist.device)
        # shape: (batch, multi, problem+1)
        # out_mat = torch.zeros(dist.shape, device=dist.device)

        out = out_mat.scatter_(-1, idx, out)
        # shape: (batch, multi, problem+1)

        return out


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        # self.global_encoder = CVRP_Global_Encoder(**model_params)

    def forward(self, depot_xy, node_xy_demand, dist):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)
        # dist.shape: (batch, problem+1, problem+1)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        
        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

        self.local = False
    
    def add_local_policy(self, device):
        self.local_policies = nn.ModuleList([local_policy_att(self.model_params, idx=i).to(device) for i in range(self.model_params['ensemble_size'])])
        self.local = True

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load, cur_dist, cur_theta, xy, norm_demand, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['embedding_dim'] ** 0.5
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim

        if self.model_params['distance_penalty']:
            local_size = self.model_params['local_size'][0]
            valid_nodes = cur_dist.shape[2]
            dist = cur_dist.clone()
            # Unselected neighbor nodes except depot (depot will be added below)
            mask = ninf_mask.clone()
            # mask depot
            mask[:, :, 0] = float('-inf')
            dist -= mask
            valid_nodes -= mask.isinf().sum(-1).min()

            if local_size > valid_nodes:
                local_size = valid_nodes
            else:
                local_size = local_size
            
            # Zero idx for depot
            depot_idx = torch.zeros(dist.shape[0], dist.shape[1], 1, device=dist.device).long()

            if local_size > 0:
                # Topk except depot, but the idx should add 1 for alignment
                dist_, idx = dist[:, :, 1:].topk(local_size, dim=-1, largest=False)
                # norm factor is computed before depot is added
                if dist_.isinf().any():
                    dist_[dist_.isinf()] = 0.
                norm_idx = dist_.max(-1)[0] != 0
                norm_fac = dist_[norm_idx].max(-1)[0].unsqueeze(-1)

                idx += 1
                # shape: (batch, multi, local)
                # Add depot idx
                idx = torch.cat((depot_idx, idx), dim=-1)
            else:
                # No other nodes can be selected except depot
                idx = depot_idx
                norm_idx = None

            sorted_dist = torch.take_along_dim(dist, idx, dim=-1)
            # shape: (batch, multi, local)

            # Padding 0
            # Check if there are some dims that require padding
            if sorted_dist.isinf().any():
                sorted_dist[sorted_dist.isinf()] = 0.
            
            if norm_idx is None:
                norm_idx = sorted_dist.max(-1)[0] != 0
                sorted_dist[norm_idx] = sorted_dist[norm_idx] / sorted_dist[norm_idx].max(-1)[0].unsqueeze(-1)
            else:
                sorted_dist[norm_idx] = sorted_dist[norm_idx] / norm_fac

            dist_penalty = - sorted_dist
            out_mat = self.model_params['xi'] * torch.ones(cur_dist.shape, device=cur_dist.device)
            score_scaled += out_mat.scatter_(-1, idx, dist_penalty)

        if self.model_params['ensemble'] and self.local:
            score_local = 0.
            for i in range(self.model_params['ensemble_size']):
                score_local += self.local_policies[i](theta=cur_theta, dist=cur_dist, xy=xy, norm_demand=norm_demand, ninf_mask=ninf_mask)
            score_scaled += score_local / self.model_params['ensemble_size']
            # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs
    

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    if len(qkv.shape) == 4:
        # q.shape: (batch, n, local, head_num * key_dim)
        batch_s = qkv.size(0)
        n1 = qkv.size(1)
        n2 = qkv.size(2)
        q_reshaped = qkv.reshape(batch_s, n1, n2, head_num, -1)
        # shape: (batch, n, local, head_num, key_dim)

        q_transposed = q_reshaped.transpose(2, 3).transpose(1, 2)
        # shape: (batch, head, n, local, key_dim)
    else:
        batch_s = qkv.size(0)
        n = qkv.size(1)

        q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
        # shape: (batch, n, head_num, key_dim)

        q_transposed = q_reshaped.transpose(1, 2)
        # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(-1)

    input_s = k.size(2)

    if len(k.shape) == 5:
        # q.shape: (batch, head_num, n, 1, key_dim)
        # k.shape: (batch, head_num, n, local, key_dim)
        score = torch.matmul(q, k.transpose(3, 4)).squeeze(-2)
        # shape: (batch, head_num, n, local)
        input_s = k.size(3)
    else:
        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, n, problem)
        
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    if len(k.shape) == 5:
        weights = nn.Softmax(dim=3)(score_scaled).unsqueeze(3)
        # shape: (batch, head_num, n, 1, local)
        out = torch.matmul(weights, v).squeeze(3)
        # shape: (batch, head_num, n, key_dim)
        
    else:
        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch, head_num, n, problem)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, key_dim)
        
    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding) or (batch, multi, problem, embeddin)

        embedding_dim = input1.shape[-1]
        norm_dim = 1.
        for shape in input1.shape[:-1]:
            norm_dim = norm_dim * shape
        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(norm_dim, embedding_dim))
        back_trans = normalized.reshape(input1.shape)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))