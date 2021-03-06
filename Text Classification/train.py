from __future__ import division
from __future__ import print_function
from sklearn import metrics
import random
import time
import sys
import os

import torch
import torch.nn as nn

import numpy as np
from utils.utils import *
from models.gcn import GCN, DeepGCN

from config import set_config

def main():
    args = set_config()
    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    dataset = args.dataset

    if dataset not in datasets:
        sys.exit("wrong dataset name")

    # Set random seed
    seed = random.randint(1, 200)
    #seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.gpu:
        torch.cuda.manual_seed(seed)


    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    dataset)

    features = sp.identity(features.shape[0])  # featureless


    # Some preprocessing
    features = preprocess_features(features)
    support = preprocess_adj(adj)

    t_features = torch.from_numpy(features.astype(np.float32))
    t_y_train = torch.from_numpy(y_train)
    t_y_val = torch.from_numpy(y_val)
    t_y_test = torch.from_numpy(y_test)
    t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
    tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
    t_support = torch.Tensor(support)


    if args.mode =='DeepGCN':
        model = DeepGCN(input_dim = features.shape[0],\
                        nlayers = args.layers,\
                        nhidden = args.hidden,\
                        nclass = y_train.shape[1],\
                        dropout_rate = args.dropout,\
                        beta = args.beta,\
                        alpha = args.alpha,\
                        var = args.var,\
                        wd = args.weight_decay)
        
        optimizer = torch.optim.Adam([{'params':model.params1,'weight_decay':args.weight_decay},\
                                  {'params':model.params2,'weight_decay':0.},\
                                  {'params':model.params3,'weight_decay':0.}],lr= args.lr)
    else:
        
    #val_losses.cpu()
        model = GCN(input_dim = features.shape[0],\
                        nlayers = args.layers,\
                        nhidden = args.hidden,\
                        nclass = y_train.shape[1],\
                        dropout_rate = args.dropout,\
                        wd = args.weight_decay)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.gpu:
        t_features = t_features.cuda()
        t_y_train = t_y_train.cuda()
        t_y_val = t_y_val.cuda()
        t_y_test = t_y_test.cuda()
        t_train_mask = t_train_mask.cuda()
        tm_train_mask = tm_train_mask.cuda()
        t_support = t_support.cuda()
        model = model.cuda()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    


    # Define model evaluation function
    def evaluate(features, labels, mask):
        t_test = time.time()
        # feed_dict_val = construct_feed_dict(
        #     features, support, labels, mask, placeholders)
        # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
        model.eval()
        with torch.no_grad():
            logits = model(features,t_support)
            t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
            if args.gpu:
                t_mask = t_mask.cuda()
            tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
            loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
            pred = torch.max(logits, 1)[1]
            acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()

        return loss, acc, pred, labels, (time.time() - t_test)



    val_losses = []

    # Train model
    for epoch in range(args.epochs):
        t = time.time()

        # Forward pass
        logits = model(t_features,t_support)
        loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])
        acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
        val_losses.append(val_loss.cpu())
        if args.log:
            print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
                    .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

        if epoch > args.tolerance and val_losses[-1] > np.mean(val_losses[-(args.tolerance+1):-1]):
            print_log("Early stopping...")
            break


    print_log("Optimization Finished!")


    # Testing
    test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
    print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

    test_pred = []
    test_labels = []
    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(pred[i].cpu())
            test_labels.append(np.argmax(labels[i].cpu()))


    print_log("Test Precision, Recall and F1-Score...")
    print_log(metrics.classification_report(test_labels, test_pred, digits=4))
    print_log("Macro average Test Precision, Recall and F1-Score...")
    print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
    print_log("Micro average Test Precision, Recall and F1-Score...")
    print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))



if __name__=="__main__":
    main()
# doc and word embeddings

# tmp = model.layer1.embedding.cpu().numpy()
# word_embeddings = tmp[train_size: adj.shape[0] - test_size]
# train_doc_embeddings = tmp[:train_size]  # include val docs
# test_doc_embeddings = tmp[adj.shape[0] - test_size:]
#
# print_log('Embeddings:')
# print_log('\rWord_embeddings:'+str(len(word_embeddings)))
# print_log('\rTrain_doc_embeddings:'+str(len(train_doc_embeddings)))
# print_log('\rTest_doc_embeddings:'+str(len(test_doc_embeddings)))
# print_log('\rWord_embeddings:')
# print(word_embeddings)
#
# with open('./data/corpus/' + dataset + '_vocab.txt', 'r') as f:
#     words = f.readlines()
#
# vocab_size = len(words)
# word_vectors = []
# for i in range(vocab_size):
#     word = words[i].strip()
#     word_vector = word_embeddings[i]
#     word_vector_str = ' '.join([str(x) for x in word_vector])
#     word_vectors.append(word + ' ' + word_vector_str)
#
# word_embeddings_str = '\n'.join(word_vectors)
# with open('./data/' + dataset + '_word_vectors.txt', 'w') as f:
#     f.write(word_embeddings_str)
#
#
#
# doc_vectors = []
# doc_id = 0
# for i in range(train_size):
#     doc_vector = train_doc_embeddings[i]
#     doc_vector_str = ' '.join([str(x) for x in doc_vector])
#     doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#     doc_id += 1
#
# for i in range(test_size):
#     doc_vector = test_doc_embeddings[i]
#     doc_vector_str = ' '.join([str(x) for x in doc_vector])
#     doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#     doc_id += 1
#
# doc_embeddings_str = '\n'.join(doc_vectors)
# with open('./data/' + dataset + '_doc_vectors.txt', 'w') as f:
#     f.write(doc_embeddings_str)
#
#
#
