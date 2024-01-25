import torch
import torch.nn as nn
import torch.nn.functional as F


class local_policy(nn.Module):
    def __init__(self, model_params, idx=0):
        super().__init__()
        emb_dim = model_params['ff_hidden_dim']
        self.model_params = model_params
        self.penalty = model_params['penalty']
        self.local_size = model_params['local_size'][idx]
        self.w_1 = nn.Linear(2 * self.local_size + 1, emb_dim)
        self.w_2 = nn.Linear(emb_dim, emb_dim * 2)
        self.w_3 = nn.Linear(emb_dim * 2, emb_dim)
        self.w_4 = nn.Linear(emb_dim, self.local_size)

        self.norm = nn.InstanceNorm1d(emb_dim * 2, affine=True)
    
    def zero_init(self):
        for name, param in self.named_parameters():
            if 'w_4' in name:
                nn.init.constant_(param, 0)

    def forward(self, dist, theta, scale, ninf_mask=None):
        # theta shape: (batch, multi, problem)
        # dist shape: (batch, multi, problem)
        valid_nodes = dist.shape[2]
        padding = False
        if self.local_size > valid_nodes:
            padding = True
            padding_len = (0, self.local_size - valid_nodes)
            local_size = valid_nodes
        else:
            local_size = self.local_size

        sorted_dist, idx = dist.topk(local_size, dim=-1, largest=False)
        # shape: (batch, multi, local)
        sorted_dist = sorted_dist / sorted_dist.max(-1)[0].unsqueeze(-1)
        sorted_theta = torch.take_along_dim(theta, idx, dim=-1)
        if padding == True:
            sorted_dist = F.pad(sorted_dist, padding_len, 'constant', 0.)
            sorted_theta = F.pad(sorted_theta, padding_len, 'constant', 0.)
        sorted_dist_theta = torch.cat((sorted_dist, sorted_theta), dim=-1)
        # shape: (batch, multi, 2 * local)
        x_in = torch.cat((sorted_dist_theta, scale), dim=-1)
        # shape: (batch, multi, 2 * local + 1)
        emb = F.relu(self.w_1(x_in))
        # shape: (batch, multi, emb)
        emb = self.norm(F.relu(self.w_2(emb)).transpose(1, 2)).transpose(1, 2)
        emb = F.relu(self.w_3(emb))
        out = self.w_4(emb) - sorted_dist
        # shape: (batch, multi, local)

        out_mat = self.penalty * torch.ones(dist.shape, device=dist.device)
        # shape: (batch, multi, problem+1)

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

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


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

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        if self.model_params['ensemble'] == 'learn':
            self.local_policy_0 = local_policy(self.model_params)
            self.local_policy_0.zero_init()
            if self.model_params['ensemble_size'] == 2:
                self.local_policy_1 = local_policy(self.model_params, idx=1)
                self.local_policy_1.zero_init()

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, cur_dist, cur_theta, scale, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q_first + q_last
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
        # shape: (batch, pomo, problem)
        
        if self.model_params['ensemble'] == 'learn':
            score_scaled += self.local_policy_0(dist=cur_dist, theta=cur_theta, ninf_mask=ninf_mask, scale=scale)
            if self.model_params['ensemble_size'] == 2:
                score_scaled += self.local_policy_1(dist=cur_dist, theta=cur_theta, ninf_mask=ninf_mask, scale=scale, id=1)
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


class Add_And_Normalization_Module(nn.Module):
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


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
