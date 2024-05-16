import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from bert_pytorch.model import BERT
from bert_pytorch.trainer import newBERTTrainer
from transformers import glue_convert_examples_to_features, GlueDataset
from transformers import GlueDataTrainingArguments
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        # 将 InputFeatures 转换为张量类型
        input_ids = torch.tensor(feature.input_ids)
        attention_mask = torch.tensor(feature.attention_mask)
        labels = torch.tensor(feature.label)

        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return sample


seq_len = 384
hidden = 768
layers = 12
attn_heads = 12

batch_size = 16
epochs = 5
learning_rate = 2e-5
output_path = "./output/Bert-cola.model"

task_name = 'cola'
data_dir = '/home/hsy/dataset/CoLA'
args = GlueDataTrainingArguments(task_name=task_name, data_dir=data_dir)

tokenizer = BertTokenizer.from_pretrained('tokenizer.json', do_lower_case=True)
train_dataset = GlueDataset(args=args, tokenizer=tokenizer)
dev_dataset = GlueDataset(args=args, tokenizer=tokenizer, mode='dev')
test_dataset = GlueDataset(args=args, tokenizer=tokenizer, mode='test')


train_features = CustomDataset(train_dataset)
dev_features = CustomDataset(dev_dataset)
test_features = CustomDataset(test_dataset)

train_loader = DataLoader(train_features, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_features, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_features, batch_size=batch_size, shuffle=False)

print("hsy: ", len(train_loader))

print("Creating BERT Trainer")
is_train = True
if is_train:
    bert = BERT(seq_len, hidden=hidden, n_layers=layers, attn_heads=attn_heads)
    trainer = newBERTTrainer(bert, seq_len, train_dataloader=train_loader, test_dataloader=test_loader, lr=learning_rate)
    
    for epoch in range(epochs):
        trainer.train(epoch)
        trainer.save(epoch, output_path)
    
    if test_loader is not None:
        trainer.test(epoch)
else:
    bert = torch.load(output_path)
    trainer = newBERTTrainer(bert, seq_len, train_dataloader=train_loader, test_dataloader=test_loader, lr=learning_rate)
    
    for epoch in range(epochs):
        trainer.train(epoch)
        trainer.save(epoch, output_path)
    
    if test_loader is not None:
        trainer.test(epoch)
