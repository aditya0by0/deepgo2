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
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

import wandb
from deepgo.data import load_normal_forms, load_ppi_data
from deepgo.metrics import compute_roc
from deepgo.models import DeepGOGATModel
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
    '--model-name', '-m', type=ck.Choice([
        'deepgogat', 'deepgogat_plus', 'deepgogat_mf', 'deepgozero_mf_plus',
        'deepgogat_mfpreds', 'deepgogat_mfpreds_plus']),
    default='deepgogat',
    help='Prediction model name')
@ck.option(
    '--model-id', '-mi', type=int, required=False)
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
def main(data_root, ont, model_name, model_id, test_data_name, batch_size, epochs, load, device, seed):
    """
    This script is used to train DeepGOGAT models
    """
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
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

    if model_id is not None:
        model_name = f'{model_name}_{model_id}'

    if ont == 'mf' and model_name.find('mf') != -1:
        raise ValueError('Molecular function based model cannot be trained for MF ontology')
    if model_name.find('plus') != -1:
        go_norm_file = f'{data_root}/go-plus.norm'
    else:
        go_norm_file = f'{data_root}/go.norm'
    
    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/{model_name}.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/{test_data_name}_predictions_{model_name}.pkl'

    go = Ontology(go_file, with_rels=True)

    
    # Load the datasets
    features_length = 2560
    features_column = 'esm2'
    if model_name.find('mfpreds') != -1:
        features_length = None # Optional in this case
        features_column = 'mf_preds'
    elif model_name.find('mf') != -1:
        features_length = None
        features_column = 'prop_annotations'
    ppi_graph_file = f'ppi_{test_data_name}.bin'    
    test_data_file = f'{test_data_name}_data.pkl'
    
    mfs_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df = load_ppi_data(
        data_root, ont, features_length, features_column, test_data_file, ppi_graph_file)
    n_terms = len(terms_dict)

    if features_column != 'esm2':
        features_length = len(mfs_dict)

    valid_labels = labels[valid_nids].numpy()
    test_labels = labels[test_nids].numpy()

    labels = labels.to(device)
    graph = graph.to(device)

    train_nids = train_nids.to(device)
    valid_nids = valid_nids.to(device)
    test_nids = test_nids.to(device)

    # Load normal forms
    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        go_norm_file, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)
    
    normal_forms = nf1, nf2, nf3, nf4
    nf1 = th.LongTensor(nf1).to(device)
    nf2 = th.LongTensor(nf2).to(device)
    nf3 = th.LongTensor(nf3).to(device)
    nf4 = th.LongTensor(nf4).to(device)
    normal_forms = nf1, nf2, nf3, nf4


    loss_func = nn.BCELoss()
    net = DeepGOGATModel(features_length, n_terms, n_zeros, n_rels, device).to(device)
    print(net)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
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
    scheduler = MultiStepLR(optimizer, milestones=[5, 10,], gamma=0.1)

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
                    el_loss = net.el_loss(normal_forms)
                    total_loss = loss + el_loss
                    train_loss += loss.detach().item()
                    train_elloss = el_loss.detach().item()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
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
                valid_roc_auc = compute_roc(valid_labels, preds)

                valid_f1_micro_score = f1_micro.compute().item()
                valid_f1_macro_score = f1_macro.compute().item()
                valid_tm_auc_roc_macro = tm_auc_roc_macro.compute().item()
                valid_tm_auc_roc_micro = tm_auc_roc_micro.compute().item()
                
                print(
                    f"Epoch {epoch}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Train AUC Macro (torchmetric) = {train_tm_auc_roc_macro:.4f}, "
                    f"Train AUC Micro (torchmetric) = {train_tm_auc_roc_micro:.4f}, "
                    f"Train F1_micro = {train_f1_micro_score:.4f}, "
                    f"Train F1_macro = {train_f1_macro_score:.4f} | "
                    f"Valid Loss = {valid_loss:.4f}, "
                    f"Valid AUC (DeepGo) = {valid_roc_auc:.4f}, "
                    f"Valid AUC Macro (torchmetrics) = {valid_tm_auc_roc_macro:.4f} "
                    f"Valid AUC Micro (torchmetrics) = {valid_tm_auc_roc_micro:.4f} "
                    f"Valid F1_micro = {valid_f1_micro_score:.4f}, "
                    f"Valid F1_macro = {valid_f1_macro_score:.4f}"
                )

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
        valid_steps = int(math.ceil(len(valid_nids) / batch_size))
        valid_loss = 0
        preds = []
        with ck.progressbar(length=valid_steps, show_pos=True) as bar:
            for input_nodes, output_nodes, blocks in valid_dataloader:
                bar.update(1)
                logits = net(input_nodes, output_nodes, blocks)
                batch_labels = labels[output_nodes]
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                valid_loss += batch_loss.detach().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            valid_loss /= valid_steps
                
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
            f"AUC (Deepgo) = {roc_auc:.4f}, "
            f"Macro AUC (torchmetric) = {test_tm_auc_roc_macro:.4f}, "
            f"Mirco AUC (torchmetric) = {test_tm_auc_roc_micro:.4f}, "
            f"F1_micro = {test_f1_micro_score:.4f}, "
            f"F1_macro = {test_f1_macro_score:.4f}"
        )

        wandb.log({
            'test_loss': test_loss,
            'test_auc (deepgo)': roc_auc,
            'test macro auc (torchmetric)': test_tm_auc_roc_macro,
            'test micro auc (torchmetric)': test_tm_auc_roc_micro,
            'test_micro_f1': test_f1_micro_score,
            'test_macro_f1': test_f1_macro_score,
        })

    # Save the performance into a file
    with open(f'{data_root}/{ont}/valid_{model_name}.pf', 'w') as f:
        f.write(f'Valid Loss - {valid_loss}, Test Loss - {test_loss}, Test AUC - {roc_auc}\n')
    
    preds = list(preds)
    # Propagate scores using ontology structure
    with Pool(32) as p:
        preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)

    test_df['preds'] = preds

    test_df.to_pickle(out_file)
    

if __name__ == '__main__':
    main()
