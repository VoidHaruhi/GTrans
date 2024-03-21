import argparse
import numpy as np
from gtransform_both import GraphAgent
from utils import *
import torch
import random
import time
import sys
st = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--nlayers', type=int, default=5)
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--loss', type=str, default='LC')
parser.add_argument('--debug', type=int, default=1)
parser.add_argument('--ood', type=int, default=1)
parser.add_argument('--with_bn', type=int, default=1)
parser.add_argument('--margin', type=float, default=-1)
parser.add_argument('--existing_space', type=int, default=1, help='enable removing edges from the graph')
parser.add_argument('--test_val', type=int, default=0, help='set to 1 to evaluate performance on validation data')
parser.add_argument('--tune', type=int, default=1)
parser.add_argument('--finetune', type=int, default=0, help='whether to finetune the model')
parser.add_argument('--tent', type=int, default=0, help='use the Tent for finetuning (need to set finetune=1)')
parser.add_argument('--strategy', type=str, default='dropedge')

parser.add_argument('--mode', type=int, default=0, help='0: test time adaptation, 1: adaptation with prompt tuning')
# args for test time adaptation
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr_feat', type=float, default=0.001)
parser.add_argument('--lr_adj', type=float, default=0.1)
parser.add_argument('--ratio', type=float, default=0.1, help='budget B for changing graph structure')
parser.add_argument('--loop_adj', type=int, default=1, help='#loops for optimizing structure')
parser.add_argument('--loop_feat', type=int, default=4, help='#loops for optimizing features')
# args for prompt tuning
parser.add_argument('--p_num', type=int, default=5, help='number of prompts')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

lr_feat = args.lr_feat; epochs = args.epochs; ratio = args.ratio; lr_adj = args.lr_adj
print('===========')
reset_args(args)
if args.model == 'GAT':
    args.loop_adj = 0; args.loop_feat = args.epochs
if args.tune: # set args.tune to 1 to change the model hyperparameters
    args.lr_feat = lr_feat; args.epochs = epochs; args.ratio = ratio; args.lr_adj = lr_adj
if args.epochs == 2:
    args.loop_adj = 1; args.loop_feat = 1

print(args)

from utils import get_gpu_memory_map
mem_st = get_gpu_memory_map()

if args.ood:
    path = 'GraphOOD-EERM/'
    if args.dataset == 'elliptic':
        path = path + 'temp_elliptic'
        sys.path.append(path)
        from main_as_utils import datasets_tr, datasets_val, datasets_te
        data = [datasets_tr, datasets_val, datasets_te]
    elif args.dataset == 'fb100':
        path = path + 'multigraph'
        sys.path.append(path)
        from main_as_utils_fb import datasets_tr, datasets_val, datasets_te
        data = [datasets_tr, datasets_val, datasets_te]
    elif args.dataset == 'amazon-photo':
        path = path + 'synthetic'
        sys.path.append(path)
        from main_as_utils_photo import dataset_tr, dataset_val, datasets_te
        data = [dataset_tr, dataset_val, datasets_te]
    else:
        if args.dataset == 'cora':
            path = path + 'synthetic'
        elif args.dataset == 'ogb-arxiv':
            path = path + 'temp_arxiv'
        elif args.dataset == 'twitch-e':
            path = path + 'multigraph'
        else:
            raise NotImplementedError
        sys.path.append(path)
        from main_as_utils import dataset_tr, dataset_val, datasets_te
        data = [dataset_tr, dataset_val, datasets_te]
else:
    data = get_dataset(args.dataset, args.normalize_features)
if data[0].__class__ == list:
    args.n_feat = data[0][0].d
else:
    args.n_feat= data[0].d

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

res = []
agent = GraphAgent(data, args)
# # preliminary study
# pred = F.softmax(agent.model.predict(data[0].graph['node_feat'].to(agent.device), data[0].graph['edge_index'].to(agent.device))).detach()
# def entropy(pred):
#     return -(pred * torch.log(pred)).sum(dim=1)
# entropy_by_node = entropy(pred)

# entropys = []
# for data_test in data[2]:
#     pred_temp = F.softmax(agent.model.predict(data_test.graph['node_feat'].to(agent.device), data_test.graph['edge_index'].to(agent.device))).detach()
#     entropys.append(entropy(pred_temp))
# entropy_test = torch.stack(entropys)

if args.test_val:
    print('using validation as test...')
    data[-1] = data[-2]
    if type(data[-1]) is not list:
        data[-1] = [data[-1]]
    y_te, out_te = [], []
    for ix, test_data in enumerate(data[-1]):
        if args.finetune:
            acc, output, labels = agent.finetune(test_data)
        else:
            if args.mode == 0: # full-graph adaptation (GTrans)
                acc, output, labels = agent.learn_graph(test_data)
            else:
                acc, output, labels = agent.prompt_tuning(test_data)
        res.append(acc)
        y_te.append(labels)
        out_te.append(output)

        if args.debug == 2:
            break
    # acc_te = agent.model.eval_func(torch.cat(y_te, dim=0), torch.cat(out_te, dim=0))
    # print(f'Results on test sets: {acc_te}')
    print(np.mean(res))
    with open(f'results/{args.dataset}_{args.model}_{args.lr_feat}_{args.lr_adj}_{args.epochs}_{args.ratio}_{args.seed}.out', 'w') as f:
        f.write(f'{np.mean(res)}')
    # print(f'Flatten Test: {acc_te:.2f}')
else:
    if args.dataset != 'elliptic':
        y_te, out_te = [], []
        for ix, test_data in enumerate(data[-1]):
            if args.finetune:
                acc, output, labels = agent.finetune(test_data)
            else:
                acc, output, labels = agent.learn_graph(test_data)
            res.append(acc)
            y_te.append(labels)
            out_te.append(output)

            if args.debug == 2:
                break
        acc_te = agent.model.eval_func(torch.cat(y_te, dim=0), torch.cat(out_te, dim=0))

    else:
        y_te_all, out_te_all = [], []
        y_te, out_te = [], []
        for ii, test_data in enumerate(data[-1]):
            if args.finetune:
                acc, output, labels = agent.finetune(test_data)
            else:
                acc, output, labels = agent.learn_graph(test_data)
            y_te.append(labels)
            out_te.append(output)

            y_te_all.append(labels)
            out_te_all.append(output)

            if ii % 4 == 0 or ii == len(data[-1]) - 1:
                acc_te = agent.model.eval_func(torch.cat(y_te, dim=0), torch.cat(out_te, dim=0))
                res += [float(f'{acc_te:.2f}')]
                y_te, out_te = [], []
                if args.debug==2:
                    break

        acc_te = agent.model.eval_func(torch.cat(y_te_all, dim=0), torch.cat(out_te_all, dim=0))

    print('Results on test sets:', res)
    print('Mean result on test sets:', np.mean(res))

