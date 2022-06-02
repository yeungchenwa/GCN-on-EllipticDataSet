import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataloader import EllipticDataSet_Loader
from utils import accuracy

from models import GCN_2layer, HigherOrderGCN_2layer
from sklearn.metrics import f1_score, precision_score, recall_score

# Test settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--test', action='store_true', default=True, help='Test during training pass.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--seed', type=int, default=111, help='Random seed.')
parser.add_argument('--skip', default=False, help='Whether to use skip-gcn or not')
parser.add_argument('--higherorder', default=True, help='Whether to use higherorder or not')
parser.add_argument('--order', type=int, default=3, help='the order in higher-order gcn if choose higherorder')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.8, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=50, help='Patience')
parser.add_argument('--data_root', default='F:/test/elliptic_bitcoin_dataset', help='Data root path')
parser.add_argument('--max_train_ts', type=int, default=34, help='Max training time step')
parser.add_argument('--end_ts', type=int, default=42, help='End train and test time step')
parser.add_argument('--nums_feat', type=int, default=166, help='Number of Graph nodes features')
parser.add_argument('--nums_class', type=int, default=2, help='Number of dataset classes')
parser.add_argument('--node_embedding', type=int, default=120, help='Number of Graph nodes features')
parser.add_argument('--out_features', type=int, default=90, help='Number of Graph nodes features')   # 高一点比较好？
parser.add_argument('--checkpoint_path', default='D:/checkpoint_BTC_abnormal_trans/hogcn_49/best_29_total_0.831039_0.613112_0.705632.pth', help='Checkpoint saved path')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.set_default_tensor_type(torch.DoubleTensor)

# Model and optimizer
if args.higherorder:
    gcn = HigherOrderGCN_2layer(args.nums_feat, args.node_embedding, args.out_features, args.nums_class, args.order, args.skip)   # hogcn seed123 test42 node emmbedings120  out features90
    print('Loaded HigherOrderGCN_2layer Successfully!')                                                                           # skip-hogcn seed1234 test42 node emmbedings120  out features90
else:
    gcn = GCN_2layer(args.nums_feat, args.node_embedding, args.nums_class, args.skip)    # gcn test42 node emmbedings100 0.8700 0.6971 0.7740
    print('Loaded GCN_2layer Successfully!')                                             # skip-gcn seed1234 node emmbedings100 
                                                                                         # gcn test49 node embedings100 0.8339 0.6142 0.7074
                                                                                         # skip-gcn test49 node embedings100 0.8669 0.6315 0.7307
gcn.load_state_dict(torch.load(args.checkpoint_path))
print("load model state dicet sucessfully!")
if args.cuda:
    gcn = gcn.cuda()

# Load data
print('Loading Dataset...')
begin = time.time()
adj, features, labels_t = EllipticDataSet_Loader(args.data_root, args.max_train_ts, args.end_ts)
end = time.time()
print('Loaded Successfully and loading time is {:2f}'.format(end-begin))
# 0 - illicit, 1 - licit
labels = []
for c in labels_t:
    labels.append(np.array(c['class'] == '2', dtype = np.long))


test_label_all = []
test_pred_all = []
test_precision_ts = []
test_recall_ts = []
test_f1_ts = []
test_ts = range(args.end_ts-args.max_train_ts+1)

for ts in test_ts:
    
    Feat_test = torch.tensor(features[ts].values)
    Adj_test = torch.tensor(adj[ts].values)
    Label_test = torch.tensor(labels[ts], dtype = torch.long)
    test_label_all = test_label_all + labels[ts].tolist()

    gcn.eval()
    if args.cuda:
        Feat_test = Feat_test.cuda()
        Adj_test = Adj_test.cuda()
    
    test_out = gcn(Adj_test, Feat_test)
    # print("test output: {}".format(test_out))
    test_out = test_out.cpu()
    
    test_pred = test_out.max(1)[1].type_as(Label_test).tolist()
    test_pred_all = test_pred_all + test_pred

    # Precison, Recall and F1 per time step
    precision_ts = precision_score(labels[ts].tolist(), test_pred, average=None)[0]
    recall_ts = recall_score(labels[ts].tolist(), test_pred, average=None)[0]   
    f1_ts = f1_score(labels[ts].tolist(), test_pred, average=None)[0]
    print('Time Step: {:04d}'.format(args.max_train_ts+ts+1),  
        'illicit precision: {:.4f}'.format(precision_ts),
        'illicit recall: {:.4f}'.format(recall_ts),
        'illicit f1: {:.4f}'.format(f1_ts))  
    
    test_precision_ts.append(precision_ts)
    test_recall_ts.append(recall_ts)
    test_f1_ts.append(f1_ts)

# Precision, Recall, F1 and MicroAVG F1 on the whole
precision = precision_score(test_label_all, test_pred_all, average=None)[0]
recall = recall_score(test_label_all, test_pred_all, average=None)[0]   
f1 = f1_score(test_label_all, test_pred_all, average=None)[0]
micro_avg_f1 = f1_score(test_label_all, test_pred_all, average='micro')
total = precision + recall + f1
print('illicit precision: {:.4f}'.format(precision),
      'illicit recall: {:.4f}'.format(recall),
    'illicit f1: {:.4f}'.format(f1),
    'MicroAVG-F1:{:.4f}'.format(micro_avg_f1))  

print('Precison per time step:')
print(test_precision_ts)
print('Recall per time step:')
print(test_recall_ts)
print('F1 per time step:')
print(test_f1_ts)