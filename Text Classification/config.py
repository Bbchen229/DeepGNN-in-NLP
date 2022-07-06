import argparse


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='R8')
    parser.add_argument("--mode", type=str, default='DeepGCN',help="Choose the GNN model: DeepGCN, GCN")
    parser.add_argument("--lr", type=int, default=0.02,help="learning rate")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=200,help = 'Dimension of the hidden layer')
    parser.add_argument("--dropout", type=int, default=0.5)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--alpha", type= int, default=0.1)
    parser.add_argument("--beta", type = int,default=0.1)
    parser.add_argument("--weight_decay", type=int, default=0.,help = 'L2 loss')
    parser.add_argument("--tolerance", type=int, default=500, help = 'Tolerance for earlying stop')
    parser.add_argument("--var", type=bool, default=False)
    parser.add_argument("--log", type=bool, default=True, help = 'Whether to print training log')
    parser.add_argument("--gpu", type=bool, default=True, help = 'Whether to use gpu')

    args = parser.parse_args()
    return args



