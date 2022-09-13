import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from model.gcn import GCNLayer

class BertGCN(nn.Module):
    def __init__(self, edges, features, config, args):
        super(BertGCN, self).__init__()
        self.label_features = features
        self.edges = edges
        self.device = args.device
        self.dropout = nn.Dropout(config['dropout_prob'])
        
        self.bert = AutoModel.from_pretrained(args.pretrained)
        self.gc1 = GCNLayer(features.size(1), self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.label_features.size(0))
        
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)['last_hidden_state'][:, 0]
        bert_output = self.dropout(bert_output)

        label_embed = self.gc1(self.label_features, self.edges)
        label_embed = F.relu(label_embed)

        output = torch.zeros((bert_output.size(0), label_embed.size(0)), device=self.device)
        
        # for i in range(bert_output.size(0)):
        #     for j in range(label_embed.size(0)):
        #         output[i, j] = self.classifier(bert_output[i] + label_embed[j])[j]
        for j in range(label_embed.size(0)):
            output[:, j] = self.classifier((bert_output + label_embed[j, :].unsqueeze(0)))[:, j]
        return output