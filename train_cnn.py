import copy
import csv
import math
import random
from functools import partial
from itertools import cycle
from multiprocessing import Pool

import click as ck
import dgl
import numpy as np
import pandas as pd
import torch as th
from dgl.nn import GraphConv
from sklearn.metrics import auc, matthews_corrcoef, roc_curve
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

import wandb
from deepgo.aminoacids import MAXLEN, to_onehot
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
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot', 'valid']),
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
            name=f'deepgocnn_{ont}_{test_data_name}',
            config={
                'epochs': epochs,
                'batch_size': batch_size,
                'model_name': 'deepgocnn',
                'ontology': ont,
                'test_data': test_data_name,
                'device': device,
            }
        )

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/deepgocnn.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/{test_data_name}_predictions_deepgocnn.pkl'
    
    go = Ontology(go_file, with_rels=True)
    loss_func = nn.BCELoss()
    test_data_file = f'{test_data_name}_data.pkl'
    terms_dict, train_data, valid_data, test_data, test_df = load_data(
        data_root, ont, terms_file, test_data_file=test_data_file)
    n_terms = len(terms_dict)
    
    net = DGCNNModel(n_terms, device).to(device)
    
    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data
    
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)

    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    
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
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            f1_micro.reset()
            f1_macro.reset()
            tm_auc_roc_macro.reset()
            tm_auc_roc_micro.reset()
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(batch_features)
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
                    f1_macro.update(preds=logits.detach().cpu(), labels=batch_labels.detach().cpu().long())
                    f1_micro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                    tm_auc_roc_macro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                    tm_auc_roc_micro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())

            train_f1_micro_score = f1_micro.compute().item()
            train_f1_macro_score = f1_macro.compute().item()
            train_tm_auc_roc_macro = tm_auc_roc_macro.compute().item()
            train_tm_auc_roc_micro = tm_auc_roc_micro.compute().item()
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                f1_micro.reset()
                f1_macro.reset()
                tm_auc_roc_macro.reset()
                tm_auc_roc_micro.reset()
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(batch_features)
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds.append(logits.detach().cpu().numpy())
                        f1_macro.update(preds=logits.detach().cpu(), labels=batch_labels.detach().cpu().long())
                        f1_micro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                        tm_auc_roc_macro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                        tm_auc_roc_micro.update(preds=logits.detach().cpu(), target=batch_labels.detach().cpu().long())
                valid_loss /= valid_steps
                preds = np.concatenate(preds)
                valid_roc_auc = compute_roc(valid_labels, preds)

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
                    f"Valid AUC (DeepGo) = {valid_roc_auc:.4f}, "
                    f"Valid AUC Macro = {valid_tm_auc_roc_macro:.4f} "
                    f"Valid AUC Micro = {valid_tm_auc_roc_micro:.4f} "
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
                        'valid_auc (deepgo)': valid_roc_auc,
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
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        f1_micro.reset()
        f1_macro.reset()
        tm_auc_roc_macro.reset()
        tm_auc_roc_micro.reset()
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
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
            f"AUC (Deepgo) = {roc_auc:.4f}, "
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


class DGCNNModel(nn.Module):

    def __init__(self, nb_gos, device, nb_filters=512, max_kernel=129, hidden_dim=1024):
        super().__init__()
        self.nb_gos = nb_gos
        # DeepGOCNN
        kernels = range(8, max_kernel, 8)
        convs = []
        for kernel in kernels:
            convs.append(
                nn.Sequential(
                    nn.Conv1d(22, nb_filters, kernel, device=device),
                    nn.MaxPool1d(MAXLEN - kernel + 1)
                ))
        self.convs = nn.ModuleList(convs)
        self.fc1 = nn.Linear(len(kernels) * nb_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, nb_gos)
        
    def deepgocnn(self, proteins):
        n = proteins.shape[0]
        output = []
        for conv in self.convs:
            output.append(conv(proteins))
        output = th.cat(output, dim=1)
        output = th.relu(self.fc1(output.view(n, -1)))
        output = th.sigmoid(self.fc2(output))
        return output
    
    def forward(self, proteins):
        return self.deepgocnn(proteins)
    
def load_data(data_root, ont, terms_file, test_data_file='test_data.pkl'):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/{test_data_file}')

    train_data = get_data(train_df, terms_dict)
    valid_data = get_data(valid_df, terms_dict)
    test_data = get_data(test_df, terms_dict)

    return terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, terms_dict):
    data = th.zeros((len(df), 22, MAXLEN), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        seq = row.sequences
        seq = th.FloatTensor(to_onehot(seq))
        data[i, :, :] = seq
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

if __name__ == '__main__':
    main()
