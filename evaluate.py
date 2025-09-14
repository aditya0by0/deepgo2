#!/usr/bin/env python

import logging
import math
import sys
import time
from collections import deque

import click as ck
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import distance
from sklearn.metrics import auc, classification_report, matthews_corrcoef, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

from deepgo.metrics import compute_metrics
from deepgo.utils import EXP_CODES, FUNC_DICT, NAMESPACES, Ontology

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--model-name', '-m', required=True, help='Prediction model name')
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot']),
    help='Test data set name')
def main(data_root, ont, model_name, test_data_name):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/{test_data_name}_predictions_{model_name}.pkl'
    test_df = pd.read_pickle(test_data_file)
    
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go.calculate_ic(annotations + test_annotations)

    # Print IC values of terms
    ics = {}
    for i, term in enumerate(terms):
        ics[term] = go.get_ic(term)
    
    # Combine scores for diamond and deepgo
    eval_preds = []
    
    for i, row in enumerate(test_df.itertuples()):
        preds = row.preds
        eval_preds.append(preds)

    eval_preds = np.concatenate(eval_preds).reshape(-1, len(terms))
    # np.save(f'{data_root}/{ont}/{model_name}_preds.npy', eval_preds)
    # return
    fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic, fmax_spec_match = compute_metrics(
        test_df, go, terms_dict, terms, ont, eval_preds)
    print(model_name, ont)
    print("------ DeepGO Metrics ----------")
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}, spec: {fmax_spec_match}')
    print(f'WFmax: {wfmax:0.3f}, threshold: {wtmax}')
    print(f'AUC: {avg_auc:0.3f}')
    print(f'AUPR: {aupr:0.3f}')
    print(f'AVGIC: {avgic:0.3f}')


if __name__ == '__main__':
    main()
