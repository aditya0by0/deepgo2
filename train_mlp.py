import click as ck
import pandas as pd
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from deepgo.aminoacids import to_onehot, MAXLEN
from dgl.nn import GraphConv
import dgl
from deepgo.torch_utils import FastTensorDataLoader
import csv
from torch.optim.lr_scheduler import MultiStepLR
from deepgo.models import MLPModel
from deepgo.data import load_data
from deepgo.utils import Ontology, propagate_annots
from multiprocessing import Pool
from functools import partial
from deepgo.metrics import compute_roc
import wandb
from epoch_metrics import MacroF1
from torchmetrics.classification import MultilabelF1Score
import random


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--model-name', '-m', type=ck.Choice([
        'mlp', 'mlp_esm']),
    default='mlp',
    help='Prediction model name')
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
def main(data_root, ont, model_name, test_data_name, batch_size, epochs, load, device, seed):
    wandb.init(
        project='deepgo2',
        name=f'{model_name}_{ont}_{test_data_name}',
        config={
            'epochs': epochs,
            'batch_size': batch_size,
            'model_name': model_name,
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

    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/{model_name}.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/{test_data_name}_predictions_{model_name}.pkl'

    go = Ontology(go_file, with_rels=True)
    loss_func = nn.BCELoss()

    # Load the datasets
    if model_name.find('esm') != -1:
        features_length = 2560
        features_column = 'esm2'
    else:
        features_length = None # Optional in this case
        features_column = 'interpros'

    test_data_file = f'{test_data_name}_data.pkl'
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(
        data_root, ont, terms_file, features_length, features_column, test_data_file=test_data_file)
    n_terms = len(terms_dict)
    if features_column == 'interpros':
        features_length = len(iprs_dict)
    net = MLPModel(features_length, n_terms, device).to(device)
    print(net)
    
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

    f1_micro = MultilabelF1Score(num_labels=n_terms, average="micro").to(device=device)
    f1_macro = MacroF1(num_labels=n_terms).to(device=device)
    
    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            train_preds = []
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
                    train_preds = np.append(train_preds, logits.detach().cpu().numpy())

            train_roc_auc = compute_roc(train_labels, train_preds)
            train_preds_tensor = th.tensor(train_preds.flatten())
            train_true_tensor = th.tensor(train_labels.flatten())
            f1_micro.reset()
            f1_macro.reset()
            train_f1_micro_score = f1_micro(train_preds_tensor, train_true_tensor).item()
            train_f1_macro_score = f1_macro(train_preds_tensor, train_true_tensor).item()
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(batch_features)
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                valid_roc_auc = compute_roc(valid_labels, preds)

                valid_preds_tensor = th.tensor(preds.flatten())
                valid_true_tensor = th.tensor(valid_labels.flatten())
                f1_micro.reset()
                f1_macro.reset()
                valid_f1_micro_score = f1_micro(valid_preds_tensor, valid_true_tensor).item()
                valid_f1_macro_score = f1_macro(valid_preds_tensor, valid_true_tensor).item()

                print(
                    f"Epoch {epoch}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Train AUC = {train_roc_auc:.4f}, "
                    f"Train F1_micro = {train_f1_micro_score:.4f}, "
                    f"Train F1_macro = {train_f1_macro_score:.4f} | "
                    f"Valid Loss = {valid_loss:.4f}, "
                    f"Valid AUC = {valid_roc_auc:.4f}, "
                    f"Valid F1_micro = {valid_f1_micro_score:.4f}, "
                    f"Valid F1_macro = {valid_f1_macro_score:.4f}"
                )

                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_auc': train_roc_auc,
                    'train_micro': train_f1_micro_score,
                    'train_macro': train_f1_macro_score,
                    'valid_loss': valid_loss,
                    'valid_auc': valid_roc_auc,
                    'valid_macro': valid_f1_macro_score,
                    'valid_micro': valid_f1_micro_score,
                })

            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

        
    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds.append(logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = np.concatenate(preds)
        roc_auc = compute_roc(test_labels, preds)

        test_preds_tensor = th.tensor(preds.flatten())
        test_true_tensor = th.tensor(test_labels.flatten())
        f1_micro.reset()
        f1_macro.reset()
        test_f1_micro_score = f1_micro(test_preds_tensor, test_true_tensor).item()
        test_f1_macro_score = f1_macro(test_preds_tensor, test_true_tensor).item()

        print(
            f"Test Results: "
            f"Loss = {test_loss:.4f}, "
            f"AUC = {roc_auc:.4f}, "
            f"F1_micro = {test_f1_micro_score:.4f}, "
            f"F1_macro = {test_f1_macro_score:.4f}"
        )

        wandb.log({
            'test_loss': test_loss,
            'test_auc': roc_auc,
            'test_micro': test_f1_micro_score,
            'test_macro': test_f1_macro_score,
        })

    preds = list(preds)
    # Propagate scores using ontology structure
    with Pool(32) as p:
        preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)

    test_df['preds'] = preds

    test_df.to_pickle(out_file)
    wandb.finish()


if __name__ == '__main__':
    main()
