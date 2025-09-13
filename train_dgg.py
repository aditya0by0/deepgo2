import copy
import csv
import math
from functools import partial
from itertools import cycle
from multiprocessing import Pool

import click as ck
import dgl
import numpy as np
import pandas as pd
import torch as th
from dgl.nn import GraphConv
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

import wandb
from deepgo.data import load_ppi_data
from deepgo.metrics import compute_roc
from deepgo.torch_utils import FastTensorDataLoader
from deepgo.utils import Ontology, propagate_annots
from epoch_metrics import MacroF1


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot']),
    help='Test data set name')
@ck.option(
    '--batch-size', '-bs', default=37,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=256,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
@ck.option('--seed', '-s', default=0)
def main(data_root, ont, test_data_name, batch_size, epochs, load, device, seed, use_wandb=True):
    if use_wandb:
        wandb.init(
            project='deepgo2',
            name=f'dgg_{ont}_{test_data_name}',
            config={
                'epochs': epochs,
                'batch_size': batch_size,
                'model_name': 'dgg',
                'ontology': ont,
                'test_data': test_data_name,
                'device': device,
            }
        )

    import random
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/dgg.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/{test_data_name}_predictions_dgg.pkl'

    go = Ontology(go_file, with_rels=True)

    loss_func = nn.BCELoss()
    features_length = None
    features_column = 'interpros'
    ppi_graph_file = f'ppi_{test_data_name}.bin'    
    test_data_file = f'{test_data_name}_data.pkl'
    iprs_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df = load_ppi_data(
        data_root, ont, features_length, features_column, test_data_file, ppi_graph_file)
    n_terms = len(terms_dict)
    features_length = len(iprs_dict)

    
    valid_labels = labels[valid_nids].numpy()
    test_labels = labels[test_nids].numpy()

    labels = labels.to(device)

    
    graph = graph.to(device)

    train_nids = train_nids.to(device)
    valid_nids = valid_nids.to(device)
    test_nids = test_nids.to(device)

    net = DeepGraphGOModel(features_length, n_terms, device).to(device)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_dataloader = dgl.dataloading.DataLoader(
        graph, train_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    test_dataloader = dgl.dataloading.DataLoader(
        graph, test_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[1, 3,], gamma=0.1)

    cpu_device = th.device('cpu')
    f1_micro = MultilabelF1Score(num_labels=n_terms, average="micro").to(cpu_device)
    f1_macro = MacroF1(num_labels=n_terms).to(cpu_device)
    tm_auc_roc_macro = MultilabelAUROC(num_labels=n_terms).to(cpu_device)
    tm_auc_roc_micro = MultilabelAUROC(num_labels=n_terms, average="micro").to(cpu_device)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_nids) / batch_size))
            f1_micro.reset()
            f1_macro.reset()
            tm_auc_roc_macro.reset()
            tm_auc_roc_micro.reset()
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for input_nodes, output_nodes, blocks in train_dataloader:
                    bar.update(1)
                    logits = net(input_nodes, output_nodes, blocks)
                    batch_labels = labels[output_nodes]
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
                    f1_macro.update(preds=logits.detach().cpu(), labels=batch_labels.detach().cpu().long())
                    f1_micro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                    tm_auc_roc_macro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                    tm_auc_roc_micro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())

            train_loss /= train_steps
            train_f1_micro_score = f1_micro.compute().item()
            train_f1_macro_score = f1_macro.compute().item()
            train_tm_auc_roc_macro = tm_auc_roc_macro.compute().item()
            train_tm_auc_roc_micro = tm_auc_roc_micro.compute().item()

            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_nids) / batch_size))
                valid_loss = 0
                preds = []
                f1_micro.reset()
                f1_macro.reset()
                tm_auc_roc_macro.reset()
                tm_auc_roc_micro.reset()
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for input_nodes, output_nodes, blocks in valid_dataloader:
                        bar.update(1)
                        logits = net(input_nodes, output_nodes, blocks)
                        batch_labels = labels[output_nodes]
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                        f1_macro.update(preds=logits.detach().cpu(), labels=batch_labels.detach().cpu().long())
                        f1_micro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                        tm_auc_roc_macro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                        tm_auc_roc_micro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())

                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)

                valid_f1_micro_score = f1_micro.compute().item()
                valid_f1_macro_score = f1_macro.compute().item()
                valid_tm_auc_roc_macro = tm_auc_roc_macro.compute().item()
                valid_tm_auc_roc_micro = tm_auc_roc_micro.compute().item()

                print(
                    f"Epoch {epoch}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Train AUC Macro = {train_tm_auc_roc_macro:.4f}, "
                    f"Train AUC Micro = {train_tm_auc_roc_micro:.4f}, "
                    f"Train F1_micro = {train_f1_micro_score:.4f}, "
                    f"Train F1_macro = {train_f1_macro_score:.4f} | "
                    f"Valid Loss = {valid_loss:.4f}, "
                    f"Valid AUC (DeepGO) = {roc_auc:.4f}, "
                    f"Valid AUC Macro = {valid_tm_auc_roc_macro:.4f}, "
                    f"Valid AUC Micro = {valid_tm_auc_roc_micro:.4f}, "
                    f"Valid F1_micro = {valid_f1_micro_score:.4f}, "
                    f"Valid F1_macro = {valid_f1_macro_score:.4f}"
                )
                
                if use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train macro auc (torchmetric)': train_tm_auc_roc_macro,
                        'train micro auc (torchmetric)': train_tm_auc_roc_micro,
                        'train_micro_f1': train_f1_micro_score,
                        'train_macro_f1': train_f1_macro_score,
                        'valid_loss': valid_loss,
                        'valid_auc (deepgo)': roc_auc,
                        'valid macro auc (torchmetric)': valid_tm_auc_roc_macro,
                        'valid micro auc (torchmetric)': valid_tm_auc_roc_micro,
                        'valid_macro_f1': valid_f1_macro_score,
                        'valid_micro_f1': valid_f1_micro_score,
                    })

            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

            scheduler.step()

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_nids) / batch_size))
        test_loss = 0
        preds = []
        f1_micro.reset()
        f1_macro.reset()
        tm_auc_roc_macro.reset()
        tm_auc_roc_micro.reset()
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for input_nodes, output_nodes, blocks in test_dataloader:
                bar.update(1)
                logits = net(input_nodes, output_nodes, blocks)
                batch_labels = labels[output_nodes]
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds.append(logits.detach().cpu().numpy())
                f1_macro.update(preds=logits.detach().cpu(), labels=batch_labels.detach().cpu().long())
                f1_micro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                tm_auc_roc_macro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                tm_auc_roc_micro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())

            test_loss /= test_steps
        preds = np.concatenate(preds)
        roc_auc = compute_roc(test_labels, preds)

        test_f1_micro_score = f1_micro.compute().item()
        test_f1_macro_score = f1_macro.compute().item()
        test_tm_auc_roc_macro = tm_auc_roc_macro.compute().item()
        test_tm_auc_roc_micro = tm_auc_roc_micro.compute().item()

        print(
            f"Test Results: "
            f"Loss = {test_loss:.4f}, "
            f"AUC (DeepGO) = {roc_auc:.4f}, "
            f"Macro AUC = {test_tm_auc_roc_macro:.4f}, "
            f"Micro AUC = {test_tm_auc_roc_micro:.4f}, "
            f"F1_micro = {test_f1_micro_score:.4f}, "
            f"F1_macro = {test_f1_macro_score:.4f}"
        )

        if use_wandb:
            wandb.log({
                'test_loss': test_loss,
                'test_auc (deepgo)': roc_auc,
                'test macro auc (torchmetric)': test_tm_auc_roc_macro,
                'test micro auc (torchmetric)': test_tm_auc_roc_micro,
                'test_micro_f1': test_f1_micro_score,
                'test_macro_f1': test_f1_macro_score,
            })
            wandb.finish()

    preds = list(preds)
    # Propagate scores using ontology structure
    with Pool(32) as p:
        preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)

    test_df['preds'] = preds

    test_df.to_pickle(out_file)


class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=False, dropout=0.5, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DeepGraphGOModel(nn.Module):

    def __init__(self, nb_iprs, nb_gos, device, hidden_dim=1024):
        super().__init__()
        self.nb_gos = nb_gos
        self.net1 = MLPBlock(nb_iprs, hidden_dim)
        self.conv1 = GraphConv(hidden_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        input_length = hidden_dim
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, nb_gos),
            nn.Sigmoid())

        
    def forward(self, input_nodes, output_nodes, blocks, residual=True):
        g1 = blocks[0]
        g2 = blocks[1]
        features = g1.ndata['feat']['_N']
        x = self.net1(features)
        x = self.conv1(g1, x)
        x = self.conv2(g2, x)
        logits = self.net2(x)
        return logits
        
    

if __name__ == '__main__':
    main()
