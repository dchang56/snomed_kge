import re
import os
import json
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, RandomSampler
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.optim as optim

import pickle
import random

## Define simple model to do n-class classification
class NodeClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(NodeClassifier, self).__init__()
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.input_dim, self.num_labels)
    
    def forward(self, inputs, labels=None):
        inputs = self.dropout(inputs)
        logits = self.classifier(inputs)
        
        return logits
    
class NodeClassifier2(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(NodeClassifier2, self).__init__()
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.dropout = nn.Dropout(0.1)
        self.layer1 = nn.Linear(self.input_dim, 64)
        self.classifier = nn.Linear(64, self.num_labels)
        self.activation = nn.ReLU()
    
    def forward(self, inputs, labels=None):
        logits = self.classifier(self.activation(self.layer1(self.dropout(inputs))))
                
        
        return logits

def evaluate(model, dataloader):
    results = {}
    device = torch.device('cuda')
    preds = None
    out_label_ids = None
    for batch in dataloader:
        model.eval()
        inputs, labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(inputs)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)
    result = (preds==out_label_ids).mean()
    results['acc'] = result
    return results

def split_dataset(df, p=0.9):
    train_size = int(len(df)*p)
    train = df[:train_size]
    test = df[train_size:]
    train_labels = train.pop('labels')
    test_labels = test.pop('labels')
    return train, test, train_labels, test_labels

def run_experiment(dataset, num_epoch=20, label_map=None, num_labels=None):    
    train, test, train_labels, test_labels = split_dataset(dataset)
    train_embeddings = torch.tensor(train.to_numpy()).float()
    test_embeddings = torch.tensor(test.to_numpy()).float()
    train_labels = torch.tensor([label_map[label] for label in train_labels], dtype=torch.long)
    test_labels = torch.tensor([label_map[label] for label in test_labels], dtype=torch.long)

    train_dataset = TensorDataset(train_embeddings, train_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=64)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=64)

    dim = train.shape[1]
    
    model = NodeClassifier(dim, num_labels)
    
    device = torch.device('cuda')
    model.to(device)
    model.zero_grad()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss_fct = nn.CrossEntropyLoss()

    for _ in range(num_epoch):
        for step, batch in enumerate(train_dataloader):
            model.train()
            
            
            inputs, labels = tuple(t.to(device) for t in batch)
            logits = model(inputs)
            loss = loss_fct(logits, labels)
            loss.backward()
            optimizer.step()
            model.zero_grad()

        results = evaluate(model, test_dataloader)
        print('epoch acc: {}'.format(results['acc']))
    return results

def main():
    ## Set up paths
    embeddings_dir = '/home/dc925/project/data/embeddings'
    snomed2vec_emb_file = os.path.join(embeddings_dir, 'Snomed2Vec/Node2Vec/snomed2vec.txt')
    cui2vec_emb_file = os.path.join(embeddings_dir, 'Cui2Vec/cui2vec_pretrained.csv')
    kge_models = ['TransE', 'DistMult', 'SimplE', 'ComplEx', 'RotatE']
    kge_models_paths = {}
    for m in kge_models:
        kge_models_paths[m] = os.path.join(embeddings_dir, 'kge/{}.pkl'.format(m))
    
    ## Load in mappings
    # scui to cui map
    scui2cui = pd.read_csv(os.path.join(embeddings_dir, 'Snomed2Vec/concept_maps/cui_scui.tsv'), sep='\t', header=None)
    scui2cui = scui2cui[:-1]
    scui2cui.columns = ['CUI', 'SCUI']
    scui2cui = scui2cui.set_index('SCUI')['CUI'].to_dict()
    semantic_info = pd.read_csv('/home/dc925/project/clinical_kge/semantic_info.csv', sep='\t', index_col=0)
    semantic_info = semantic_info.drop_duplicates(subset='CUI')
    cui2sty = semantic_info.set_index('CUI')['STY'].to_dict()
    cui2sg = semantic_info.set_index('CUI')['SemGroup'].to_dict()

    ## Load in embeddings
    # load in snomed2vec
    snomed2vec = {}
    with open(snomed2vec_emb_file, 'r') as fin:
        for i, line in enumerate(fin):
            if i == 0:
                continue
            line = line.strip().split()
            scui = int(line[0])
            embedding = np.array(line[1:], dtype=float)
            if scui in scui2cui:
                snomed2vec[scui2cui[scui]] = embedding
    snomed2vec = pd.DataFrame.from_dict(snomed2vec, orient='index')

    # load in cui2vec
    cui2vec = pd.read_csv(cui2vec_emb_file, index_col=0)

    # load in kge
    kge_embeddings = {}
    for m, p in kge_models_paths.items():
        print('loading {}'.format(m))
        with open(p, 'rb') as fin:
            model = pickle.load(fin)
            embeddings = model.solver.entity_embeddings
            embeddings = pd.DataFrame(embeddings)
            embeddings['CUI'] = [model.graph.id2entity[i] for i in range(len(embeddings))]
            embeddings = embeddings.set_index('CUI')
        kge_embeddings[m] = embeddings

    ## Get intersecting CUIs and subset
    cuis_intersection = list(set(kge_embeddings['TransE'].index) & set(cui2vec.index) & set(snomed2vec.index) - set(['C0015919']))
    random.seed(42)
    random.shuffle(cuis_intersection)
    snomed2vec = snomed2vec.loc[cuis_intersection]
    cui2vec = cui2vec.loc[cuis_intersection]
    for m in kge_models:
        kge_embeddings[m] = kge_embeddings[m].loc[cuis_intersection]
        
    # Semantic groups
    labels = [cui2sg[cui] for cui in cuis_intersection]
    label_map = {label: i for i, label in enumerate(np.unique(labels))}

    models = {m:kge_embeddings[m] for m in kge_models}
    models['snomed2vec'] = snomed2vec
    models['cui2vec'] = cui2vec
    for name, model in models.items():
        model['labels'] = labels
        num_labels = len(set(model['labels']))
    
    
    
    for name, embeddings in models.items():
        print('running {}'.format(name))
        run_experiment(embeddings, num_epoch=25, label_map=label_map, num_labels=num_labels)
        
    
    # Semantic types
    labels = [cui2sty[cui] for cui in cuis_intersection]
    label_map = {label: i for i, label in enumerate(np.unique(labels))}

    models = {m:kge_embeddings[m] for m in kge_models}
    models['snomed2vec'] = snomed2vec
    models['cui2vec'] = cui2vec
    for name, model in models.items():
        model['labels'] = labels
        # filter out infrequent labels
        keep_labels = model['labels'].value_counts()[model['labels'].value_counts()>25]
        model = model[model['labels'].isin(keep_labels.index)]
        models[name] = model
        num_labels = len(set(model['labels']))
        label_map = {label: i for i, label in enumerate(set(model['labels']))}
        
    
    for name, embeddings in models.items():
        print('running sty {}'.format(name))
        run_experiment(embeddings, num_epoch=30, label_map=label_map, num_labels=num_labels)


if __name__ == '__main__':
    main()
    