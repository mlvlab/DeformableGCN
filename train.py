import time
import argparse
import torch
import torch.nn.functional as F
from utils import accuracy, preprocess_data
from model import DeformableGCN
import dgl

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='squirrel')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_blocks', type=int, default=1, help='Number of layers')
parser.add_argument('--n_neighbor', type=int, default=5)
parser.add_argument('--n_hops', type=int, default=5)
parser.add_argument('--n_kernels', type=int, default=1)
parser.add_argument('--dataset_split', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument("--phi_dim", type=int, default=4)
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
g, nclass, features, labels, train, val, test = preprocess_data(args.dataset, args.dataset_split)
g = dgl.add_self_loop(g)

features = features.to(device)

labels = labels.to(device)
train = train.to(device)
test = test.to(device)
val = val.to(device)
g = g.to(device)
deg = g.in_degrees().cuda().float().clamp(min=1)
norm = torch.pow(deg, -0.5)
g.ndata['d'] = norm

net = DeformableGCN(g, features.size()[1], args.hidden, nclass, args.num_blocks, args.n_neighbor, args.n_hops, args.n_kernels, args.phi_dim, features, F.relu, args.dropout).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
dur = []
los = []
loc = []
counter = 0
min_loss = 100.0
max_acc = 0.0
best_test_acc = 0.0


for epoch in range(args.epochs):
    if epoch >= 3:
        t0 = time.time()

    net.train()
    logp, l_sep, l_focus = net(features)

    cla_loss = F.nll_loss(logp[train], labels[train])
    loss = cla_loss + args.alpha*l_sep*(-1) + args.beta*l_focus

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc = accuracy(logp[train], labels[train])
    net.eval()
    logp, _, _ = net(features)

    test_acc = accuracy(logp[test], labels[test])
    val_acc = accuracy(logp[val], labels[val])
    loss_val = F.nll_loss(logp[val], labels[val]).item()
    los.append([epoch, loss_val, val_acc, test_acc])
    
    if val_acc >= max_acc and min_loss >= loss_val:
        min_loss = loss_val
        max_acc = val_acc
        best_test_acc = test_acc
        state_dict_early_model = net.state_dict() 

        counter = 0
    else:
        counter += 1

        
print("Best Test Acc {:.5f}".format(best_test_acc))

