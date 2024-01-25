import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TSP_Encoder, TSP_Decoder, local_policy_att, _get_encoding
from models import reshape_by_heads, multi_head_attention


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)
    
    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def one_step_rollout(self, state, cur_dist, cur_theta, xy, eval_type):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.tensor(random.sample(range(0, pomo_size), pomo_size), device=state.BATCH_IDX.device)[
                           None, :] \
                    .expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size), device=state.BATCH_IDX.device)

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, cur_dist=cur_dist, cur_theta=cur_theta, xy=xy, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            if eval_type == 'sample':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob
        

class Att_Local_policy(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.att = local_policy_att(self.model_params)

    def get_type(self):
        return 'local_mlp'
    
    def one_step_rollout(self, state, cur_dist, cur_theta, eval_type):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.tensor(random.sample(range(0, pomo_size), pomo_size), device=state.BATCH_IDX.device)[
                           None, :] \
                    .expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size), device=state.BATCH_IDX.device)

        else:
            action_scores = self.att(dist=cur_dist, theta=cur_theta, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)
            logit_clipping = self.model_params['logit_clipping']
            score_clipped = logit_clipping * torch.tanh(action_scores)

            score_masked = score_clipped + state.ninf_mask

            probs = F.softmax(score_masked, dim=2)
            # shape: (batch, pomo, problem)

            if eval_type == 'sample':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob