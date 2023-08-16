import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np


class GCN_prop(MessagePassing):
    def __init__(self, L, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.L = L

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        reps = []
        for k in range(self.L):
            x = self.propagate(edge_index, x=x, norm=norm)
            reps.append(x)

        return reps

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.L,)



class MIP_prop(MessagePassing):
    def __init__(self, args, **kwargs):
        super(MIP_prop, self).__init__(aggr='add', **kwargs)
        self.layers = args.layers

    def forward(self, x, edge_index, edge_weight):
        embed_layer = []
        if(self.layers != [0]):
            for layer in self.layers:
                x = self.propagate(edge_index[layer - 1], x=x, norm=edge_weight[layer - 1])
                embed_layer.append(x)
        return embed_layer


class HeterSNE(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()

        self.L = args.L
        self.dropout = args.dropout
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.input_size = dataset.graph['node_feat'].shape[1]

        if args.Init=='random':
            # random
            bound = np.sqrt(3/(self.L))
            logits = np.random.uniform(-bound, bound, self.L)
            logits = logits/np.sum(np.abs(logits))
            self.logits = Parameter(torch.tensor(logits))
            print(f"init logits: {logits}")
        else:
            # fixed
            logits = np.array([1, float('-inf'), float('-inf')])
            self.logits = torch.tensor(logits)

        self.FFN = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size)
        )

        self.prop = GCN_prop(self.L)
        self.MH_prop = MIP_prop(args)

    def forward(self, x):
        return self.FFN(x)

    @torch.no_grad()
    def get_embedding(self, x):
        self.FFN.eval()
        return self.FFN(x)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.logits)
        bound = np.sqrt(3/(self.L))
        logits = np.random.uniform(-bound, bound, self.L)
        logits = logits/np.sum(np.abs(logits))
        for k in range(self.L):
            self.logits.data[k] = logits[k]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def ANC_loss(self, h1, h2, gamma, temperature=1, bias=1e-8):
        z1 = F.normalize(h1, dim=-1, p=2)
        z2 = gamma*F.normalize(h2, dim=-1, p=2)

        numerator = torch.exp(
            torch.sum(z1 * z2, dim=1, keepdims=True) / temperature)

        E_1 = torch.matmul(z1, torch.transpose(z1, 1, 0))

        denominator = torch.sum(
            torch.exp(E_1 / temperature), dim=1, keepdims=True)

        return -torch.mean(torch.log(numerator / (denominator + bias) + bias))

    def ANC_total(self, h0, hs):
        loss = torch.tensor(0, dtype=torch.float32).cuda()
        gamma = F.softmax(self.logits, dim=0)
        for i in range(len(hs)):
            loss += self.ANC_loss(h0, hs[i], gamma[i])

        return loss

