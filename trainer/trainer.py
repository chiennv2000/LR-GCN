import numpy as np
import torch
import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from sentence_transformers import SentenceTransformer
from model.bert_gcn import BertGCN
from dataset import Dataset

from sklearn import metrics
from torchmetrics import Precision
from tqdm import tqdm

class TrainerModel(object):
    def __init__(self, config, args):
        self.config = config
        self.args = args
        
        self.sbert = SentenceTransformer(args.sbert, device='cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        self.dataset = Dataset(args.train_data_path,
                               args.val_data_path,
                               args.test_data_path,
                               self.tokenizer,
                               config['batch_size'],
                               config['max_length'],
                               self.sbert)
        
        self.train_loader, self.val_loader, self.test_loader = self.dataset.train_loader, self.dataset.val_loader, self.dataset.test_loader
        
        if self.args.mode == "train":
            self.model = BertGCN(self.dataset.edges.to(args.device),
                                self.dataset.label_features.to(args.device),
                                config,
                                args)
        elif self.args.mode == "test":
            self.model = torch.load(args.checkpoint)
            
        self.model.to(args.device)
        
        self.optimizer, self.scheduler = self._get_optimizer()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    
    def _get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        
        no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
        optimizer_grouped_parameters = [
                {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
                    'weight_decay': self.config['weight_decay']},
                    {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
                'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config['learning_rate'])
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_training_steps=self.config['n_epochs']*len(self.train_loader),
                                                    num_warmup_steps=100)
        
        return optimizer, scheduler
    
    def validate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predicted_labels, target_labels = list(), list()

            for i, batch in enumerate(tqdm(dataloader)):
                input_ids, attention_mask, y_true = tuple(t.to(self.config['device']) for t in batch)
                output = self.model.forward(input_ids, attention_mask)
                loss = self.loss_fn(output, y_true.float())

                total_loss += loss.item()

                target_labels.extend(y_true.cpu().detach().numpy())
                predicted_labels.extend(torch.sigmoid(output).cpu().detach().numpy())

            val_loss = total_loss/len(dataloader)

        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels.round())
        micro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='micro')
        macro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='macro')
        
        ndcg1 = metrics.ndcg_score(target_labels, predicted_labels, k=1)
        ndcg3 = metrics.ndcg_score(target_labels, predicted_labels, k=3)
        ndcg5 = metrics.ndcg_score(target_labels, predicted_labels, k=5)
        
        n_classes = self.dataset.label_features.size(0)
        p1 = Precision(num_classes=n_classes, top_k=1)(torch.tensor(predicted_labels), torch.tensor(target_labels))
        p3 = Precision(num_classes=n_classes, top_k=3)(torch.tensor(predicted_labels), torch.tensor(target_labels))
        p5 = Precision(num_classes=n_classes, top_k=5)(torch.tensor(predicted_labels), torch.tensor(target_labels))

        return val_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5
    
    def step(self, batch):
        self.model.train()
        input_ids, attention_mask, label = tuple(t.to(self.args.device) for t in batch)
        self.optimizer.zero_grad()

        y_pred = self.model.forward(input_ids, attention_mask)
        loss = self.loss_fn(y_pred, label.float())
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
    
    def train(self):
        print("Training...")
        best_score = float("-inf")
        for epoch in range(self.config['n_epochs']):
            total_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                loss = self.step(batch)
                total_loss += loss 
                if (i + 1) % 50 == 0 or i == 0 or i == len(self.train_loader) - 1:
                    print("Epoch: {} - iter: {}/{} - train_loss: {}".format(epoch, i + 1, len(self.train_loader), total_loss/(i + 1)))
                if i == len(self.train_loader) - 1:
                    print("Evaluating...")
                    val_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = self.validate(self.val_loader)
                    print("Val_loss: {} - Accuracy: {} - Micro-F1: {} - Macro-F1: {}".format(val_loss, accuracy, micro_f1, macro_f1))
                    print("nDCG1: {} - nDCG@3: {} - nDCG@5: {} - P@1: {} - P@3: {} - P@5: {}".format(ndcg1, ndcg3, ndcg5, p1, p3, p5))

                    if best_score < micro_f1:
                        best_score = micro_f1
                        self.save(epoch)
    def test(self):
        print("Testing...")
        test_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = self.validate(self.test_loader)
        print("Test_loss: {} - Accuracy: {} - Micro-F1: {} - Macro-F1: {}".format(test_loss, accuracy, micro_f1, macro_f1))
        print("nDCG1: {} - nDCG@3: {} - nDCG@5: {} - P@1: {} - P@3: {} - P@5: {}".format(ndcg1, ndcg3, ndcg5, p1, p3, p5))
        
        
    def save(self, epoch):
        torch.save(self.model, f'../ckpt/checkpoint_{epoch}.pt')