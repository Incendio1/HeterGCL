import os
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import numpy as np
from torch_sparse import SparseTensor
from google_drive_downloader import GoogleDriveDownloader as gdd
import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def to_planetoid(dataset):
    """
        Takes in a NCDataset and returns the dataset in H2GCN Planetoid form, as follows:

        x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ty => the one-hot labels of the test instances as numpy.ndarray object;
        ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        split_idx => The ogb dictionary that contains the train, valid, test splits
    """
    split_idx = dataset.get_idx_split('random', 0.25)
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    graph, label = dataset[0]

    label = torch.squeeze(label)

    print("generate x")
    x = graph['node_feat'][train_idx].numpy()
    x = sp.csr_matrix(x)

    tx = graph['node_feat'][test_idx].numpy()
    tx = sp.csr_matrix(tx)

    allx = graph['node_feat'].numpy()
    allx = sp.csr_matrix(allx)

    y = F.one_hot(label[train_idx]).numpy()
    ty = F.one_hot(label[test_idx]).numpy()
    ally = F.one_hot(label).numpy()

    edge_index = graph['edge_index'].T

    graph = defaultdict(list)

    for i in range(0, label.shape[0]):
        graph[i].append(i)

    for start_edge, end_edge in edge_index:
        graph[start_edge.item()].append(end_edge.item())

    return x, tx, allx, y, ty, ally, graph, split_idx


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def normalize(edge_index):
    """ normalizes the edge_index
    """
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def gen_normalized_adjs(dataset):
    """ returns the normalized adjacency matrix
    """
    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0

    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1) * adj
    AD = adj * D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    return train_acc, valid_acc, test_acc, out

@torch.no_grad()
def evaluate_f1(model, dataset, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset)

    label = dataset.label.detach().cpu().numpy()
    y_pred = out.detach().argmax(dim=-1).cpu().numpy()

    # print(split_idx['train'].device)

    train_f1_micro = eval_func(
        label[split_idx['train'].cpu()], y_pred[split_idx['train'].cpu()], average='micro')
    valid_f1_micro = eval_func(
        label[split_idx['valid'].cpu()], y_pred[split_idx['valid'].cpu()], average='micro')
    test_f1_micro = eval_func(
        label[split_idx['test'].cpu()], y_pred[split_idx['test'].cpu()], average='micro')

    train_f1_macro = eval_func(
        label[split_idx['train'].cpu()], y_pred[split_idx['train'].cpu()], average='macro')
    valid_f1_macro = eval_func(
        label[split_idx['valid'].cpu()], y_pred[split_idx['valid'].cpu()], average='macro')
    test_f1_macro = eval_func(
        label[split_idx['test'].cpu()], y_pred[split_idx['test'].cpu()], average='macro')

    return train_f1_micro, valid_f1_micro, test_f1_micro, train_f1_macro, valid_f1_macro, test_f1_macro, out


def load_fixed_splits(dataset, sub_dataset):
    name = dataset
    if sub_dataset:
        name += f'-{sub_dataset}'

    if not os.path.exists(f'./data/splits/{name}-splits.npy'):
        assert dataset in splits_drive_url.keys()
        gdd.download_file_from_google_drive(
            file_id=splits_drive_url[dataset], \
            dest_path=f'./data/splits/{name}-splits.npy', showsize=True) 
    
    splits_lst = np.load(f'./data/splits/{name}-splits.npy', allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


dataset_drive_url = {
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
}

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N', 
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_', 
}





def edgelist2graph(edge_index, nodenum):
    """ Preprocess to get the graph data wanted
    """
    edge_index = edge_index.cpu().detach().numpy()
    adjlist = {i: [] for i in range(nodenum)}

    for i in range(len(edge_index[0])):
        adjlist[edge_index[0][i]].append(edge_index[1][i])


    return adjlist, nx.adjacency_matrix(nx.from_dict_of_lists(adjlist)).toarray()


def edge_info(dataset, args):
    adjlist, adjmatrix = edgelist2graph(dataset.graph['edge_index'], dataset.graph['node_feat'].size(0))
    hop_edge_index, hop_edge_att = ILD_decomposition(dataset.graph['node_feat'].size(0), args.MH_layer, adjlist, dataset.graph['edge_index'])
    torch.save(hop_edge_index, './MH_info/hop_edge_index_' + args.dataset + '_' + str(args.MH_layer))
    torch.save(hop_edge_att, './MH_info/hop_edge_att_' + args.dataset + '_' + str(args.MH_layer))


def propagate(x, edge_index):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes = x.size(0))

    #calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype = x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    edge_weight = deg_inv_sqrt[row]*deg_inv_sqrt[col]    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]

    out = edge_weight.view(-1, 1)*x[row]    # normalize the features on the starting point of the edge

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def weight_deg(nodenum, edge_index, K):
    att = []
    x = torch.eye(nodenum)    # E
    for i in range(K):
        x = propagate(x, edge_index)    # E*A=A
        att.append(x)

    return att


def ILD_decomposition(nodenum, K, adjlist, edge_index):
    # calculate the attention, the weight of each edge
    att = weight_deg(nodenum, edge_index, K)

    # to save the space, we use torch.tensor instead of list to save the edge_index
    hop_edge_index = [np.zeros((2, nodenum**2), dtype=int) for i in range(K)]    # at most nodenum**2 edges
    hop_edge_att = [np.zeros(nodenum**2) for i in range(K)]
    hop_edge_pointer = np.zeros(K, dtype=int)

    for i in range(nodenum):
        hop_edge_index, hop_edge_att, hop_edge_pointer = BFS_adjlist(adjlist, nodenum, hop_edge_index, hop_edge_att, hop_edge_pointer, att, source=i, depth_limit=K)

    for i in range(K):
        hop_edge_index[i] = hop_edge_index[i][:, :hop_edge_pointer[i]]
        hop_edge_att[i] = hop_edge_att[i][:hop_edge_pointer[i]]

        hop_edge_index[i] = torch.tensor(hop_edge_index[i], dtype=torch.long)
        hop_edge_att[i] = torch.tensor(hop_edge_att[i], dtype=torch.float)

    return hop_edge_index, hop_edge_att


def BFS_adjlist(adjlist, nodenum, hop_edge_index, hop_edge_att, hop_edge_pointer, deg_att, source, depth_limit):
    visited = {}
    for node in adjlist.keys():
        visited[node] = 0

    queue, output = [], []
    queue.append(source)
    visited[source] = 1
    level = 1

    # initialize the edge pointed to the source node itself
    for i in range(len(hop_edge_index)):
        hop_edge_index[i][0, hop_edge_pointer[i]] = source
        hop_edge_index[i][1, hop_edge_pointer[i]] = source

    tmp = 0
    for k in range(0, depth_limit):
        tmp += deg_att[k][source, source]
    hop_edge_att[0][hop_edge_pointer[0]] = tmp
    hop_edge_pointer[0] += 1

    while queue:
        level_size = len(queue)
        while(level_size != 0):
            vertex = queue.pop(0)
            level_size -= 1
            for vrtx in adjlist[vertex]:
                if(visited[vrtx] == 0):
                    queue.append(vrtx)
                    visited[vrtx] = 1

                    hop_edge_index[level - 1][0, hop_edge_pointer[level - 1]] = source    # distance = 1 is the first group in the hop_edge_list
                    hop_edge_index[level - 1][1, hop_edge_pointer[level - 1]] = vrtx
                    tmp = 0
                    for k in range((level - 1), depth_limit):
                        tmp += deg_att[k][source, vrtx]
                    hop_edge_att[level - 1][hop_edge_pointer[level - 1]] = tmp
                    hop_edge_pointer[level - 1] += 1

        level += 1
        if (level > depth_limit):
            break

    return hop_edge_index, hop_edge_att, hop_edge_pointer