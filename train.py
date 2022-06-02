from __future__ import division
from __future__ import print_function

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

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--test', action='store_true', default=True, help='Test during training pass.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--seed', type=int, default=8, help='Random seed.')
parser.add_argument('--skip', default=False, help='Whether to use skip-gcn or not')
parser.add_argument('--higherorder', default=True, help='Whether to use higherorder or not')
parser.add_argument('--order', type=int, default=2, help='the order in higher-order gcn if choose higherorder')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.8, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=50, help='Patience')
parser.add_argument('--data_root', default='F:/test/elliptic_bitcoin_dataset', help='Data root path')
parser.add_argument('--max_train_ts', type=int, default=34, help='Max training time step')
parser.add_argument('--end_ts', type=int, default=49, help='End train and test time step')
parser.add_argument('--nums_feat', type=int, default=166, help='Number of Graph nodes features')
parser.add_argument('--nums_class', type=int, default=2, help='Number of dataset classes')
parser.add_argument('--node_embedding', type=int, default=120, help='Number of Graph nodes features')
parser.add_argument('--out_features', type=int, default=90, help='Number of Graph nodes features')   

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.benchmark=True

# Model and optimizer
if args.higherorder:
    gcn = HigherOrderGCN_2layer(args.nums_feat, args.node_embedding, args.out_features, args.nums_class, args.order, args.skip)   
    print('Loaded HigherOrderGCN_2layer Successfully!')
    # md-gcn order2 seed8 test42 node emmbedings120  out features90    lr0.001
    # md-gcn order3 seed125 test42 node emmbedings120  out features90    lr0.001
    # md-cn order4 seed123 test42 node emmbedings120  out features80
    # skip-md-gcn order2 seed1234 test42 node emmbedings120  out features90  lr0.001
    # skip-md-gcn order3 seed1024 test42 node emmbedings120  out features90  lr0.001
    # skip-md-gcn order4 seed1234 test42 node emmbedings120  out features80  lr0.001
else:
    gcn = GCN_2layer(args.nums_feat, args.node_embedding, args.nums_class, args.skip)    # gcn test42 node emmbedings100 0.8700 0.6971 0.7740
    print('Loaded GCN_2layer Successfully!')                                             # skip-gcn seed1234 node emmbedings100 
                                                                                         # gcn test49 node embedings100 0.8339 0.6142 0.7074
                                                                                         # skip-gcn test49 node embedings100 0.8669 0.6315 0.7307

# optimizer and loss
optimizer = optim.Adam(gcn.parameters(), lr=args.lr)
loss_weight = torch.DoubleTensor([0.7, 0.3])
if args.cuda:
    gcn = gcn.cuda()
    loss_weight = loss_weight.cuda()
train_loss = nn.CrossEntropyLoss(weight=loss_weight)


# Load data
print('Loading Dataset...')
begin = time.time()
adj, features, labels_t = EllipticDataSet_Loader(args.data_root, 0, args.end_ts)
end = time.time()
print('Loaded Successfully and loading time is {:2f}'.format(end-begin))
# 0 - illicit, 1 - licit
labels = []
for c in labels_t:
    labels.append(np.array(c['class'] == '2', dtype = np.long))


# Train and test model
t_total = time.time()
bad_count = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_total = 0
for epoch in range(args.epochs):
    train_ts = np.arange(args.max_train_ts)
    print('Epoch {} Training...'.format(epoch+1))
    for ts in train_ts:
        Feat = torch.tensor(features[ts].values)
        Adj = torch.tensor(adj[ts].values)
        Label = torch.tensor(labels[ts], dtype = torch.long)

        if args.cuda:
            Feat = Feat.cuda()
            Adj = Adj.cuda()
            Label = Label.cuda()

        t = time.time()
        gcn.train()
        
        optimizer.zero_grad()
        output = gcn(Adj, Feat)
        train_pred = output.max(1)[1].type_as(Label)

        loss = train_loss(output, Label)
        loss.backward()
        optimizer.step()
        # loss = F.nll_loss(torch.log(output), Label)

        train_acc = accuracy(output, Label)
        # train_precision = precision_score(Label, train_pred, average=None).tolist()
        # train_pred = output.max(1)[1].type_as(Label)
        # acc = (train_pred.eq(Label).double().sum())/Label.shape[0]
    
        print('Epoch: {:04d}'.format(epoch+1), 
              'Time Step: {}'.format(ts+1), 
              'Train Loss: {:.4f}'.format((loss.item())), 
              'Accuracy: {:.4f}'.format(train_acc),
              'Graph Nodes Nums: {}'.format(Label.size()))
    
    if args.test:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        # Testing
        test_label_all = []
        test_pred_all = []
        test_ts = np.arange(args.max_train_ts, args.end_ts)

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
            # print("test pred: {}".format(test_pred))
            # test_acc = accuracy(test_out, Label_test)
            # test_pred = output.max(1)[1].type_as(Label)
            # test_acc = (test_pred.eq(Label).double().sum())/Label.shape[0]

            # test_accs.append(test_acc.item())
    
        precision = precision_score(test_label_all, test_pred_all, average=None)[0]
        recall = recall_score(test_label_all, test_pred_all, average=None)[0]   
        f1 = f1_score(test_label_all, test_pred_all, average=None)[0]
        total = precision + recall + f1

        print('Epoch: {:04d}'.format(epoch+1),  
              'illicit precision: {:.4f}'.format(precision),
              'illicit recall: {:.4f}'.format(recall),
              'illicit f1: {:.4f}'.format(f1))  # 'averaged accuracy: {:.4f}'.format(acc), 
        
        if precision > best_precision:
            bad_count = 0
            best_precision = precision
            print('Find the best parameter, saving the checkpoint...')
            torch.save(gcn.state_dict(), 'D:/checkpoint_BTC_abnormal_trans/best_%d_precision_%f_%f_%f.pth'%(epoch+1, precision, recall, f1))
            print('Saved Successfully!')
        else:
            bad_count+=1
        
        if recall > best_recall:
            best_recall = recall
            print('Find the best parameter, saving the checkpoint...')
            torch.save(gcn.state_dict(), 'D:/checkpoint_BTC_abnormal_trans/best_%d_recall_%f_%f_%f.pth'%(epoch+1, precision, recall, f1))
            print('Saved Successfully!')
        
        if f1 > best_f1:
            best_f1 = f1
            print('Find the best parameter, saving the checkpoint...')
            torch.save(gcn.state_dict(), 'D:/checkpoint_BTC_abnormal_trans/best_%d_f1_%f_%f_%f.pth'%(epoch+1, precision, recall, f1))
            print('Saved Successfully!')
        
        if total > best_total:
            best_total = total
            print('Find the best parameter, saving the checkpoint...')
            torch.save(gcn.state_dict(), 'D:/checkpoint_BTC_abnormal_trans/best_%d_total_%f_%f_%f.pth'%(epoch+1, precision, recall, f1))
            print('Saved Successfully!')

        if bad_count>=args.patience:
            print('Early Stop!')
            break
    
    if (epoch+1)%50==0:
        torch.save(gcn.state_dict(), 'D:/checkpoint_BTC_abnormal_trans/{}.pth'.format(epoch+1))

print('The best Precision: {:.4f}'.format(best_precision))
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))