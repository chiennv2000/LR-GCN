import json
import math
from torch.utils.data import TensorDataset, DataLoader, IterableDataset

from sklearn.preprocessing import MultiLabelBinarizer
from utils.utils import *

class Dataset(object):
    
    def __init__(self,
                 train_data_path,
                 val_data_path,
                 test_data_path,
                 tokenizer,
                 batch_size,
                 max_length,
                 sbert):
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sbert = sbert
        
        self.train_loader, self.val_loader, self.test_loader, self.edges, self.label_features = self.load_dataset(train_data_path, val_data_path, test_data_path)
    
    def load_dataset(self, train_data_path, val_data_path, test_data_path):
        train = json.load(open(train_data_path))
        val = json.load(open(val_data_path))
        test = json.load(open(test_data_path))
        
        train_sents = [clean_string(text) for text in train['content']]
        val_sents = [clean_string(text) for text in val['content']]
        test_sents = [clean_string(text) for text in test['content']]

        
        mlb = MultiLabelBinarizer()
        train_labels = mlb.fit_transform(train['labels'])
        print("Numbers of labels: ", len(mlb.classes_))
        val_labels = mlb.transform(val['labels'])
        test_labels = mlb.transform(test['labels'])

        edges, label_features = self.create_edges_and_features(train, mlb)
        
        train_loader = self.encode_data(train_sents, train_labels, shuffle=True)
        val_loader = self.encode_data(val_sents, val_labels, shuffle=False)
        test_loader = self.encode_data(test_sents, test_labels, shuffle=False)
        
        return train_loader, val_loader, test_loader, edges, label_features
    
    def create_edges_and_features(self, train_data, mlb):
        label2id = {v: k for k, v in enumerate(mlb.classes_)}

        edges = torch.zeros((len(label2id), len(label2id)))
        for label in train_data["labels"]:
            if len(label) >= 2:
                for i in range(len(label) - 1):
                    for j in range(i + 1, len(label)):
                        src, tgt = label2id[label[i]], label2id[label[j]]
                        edges[src][tgt] += 1
                        edges[tgt][src] += 1
        
        marginal_edges = torch.zeros((len(label2id)))
        
        for label in train_data["labels"]:
            for i in range(len(label)):
                marginal_edges[label2id[label[i]]] += 1
        
        for i in range(edges.size(0)):
            for j in range(edges.size(1)):
                if edges[i][j] != 0:
                    edges[i][j] = math.log((edges[i][j] * len(train_data["labels"]))/(marginal_edges[i] * marginal_edges[j]))
                    #filtering and reweighting
                    if edges[i][j] <= 0.05:
                        edges[i][j] = 0
                    else:                 
                        edges[i][j] = 1/(1 + math.exp((-13)*edges[i][j] + 7.0))
                    

        edges = normalizeAdjacency(edges + torch.diag(torch.ones(len(label2id))))
    
        # Get embeddings from wikipedia
        features = torch.zeros(len((label2id)), 768)
        for label, id in tqdm(label2id.items()):
            features[id] = get_embedding_from_wiki(self.sbert, label, n_sent=2)
            
        return edges, features
    
    def encode_data(self, train_sents, train_labels, shuffle=False):
        X_train = self.tokenizer.batch_encode_plus(train_sents, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        y_train = torch.tensor(train_labels, dtype=torch.long)
        
        train_tensor = TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train)
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)
        
        return train_loader
    