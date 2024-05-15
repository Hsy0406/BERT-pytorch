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

#processor = CoLAProcessor()
#train_examples = processor.get_train_examples('/data/huangsy/dataset/CoLA/train.tsv')
#dev_examples = processor.get_dev_examples('/data/huangsy/dataset/CoLA/dev.tsv')
#test_examples = processor.get_dev_examples('/data/huangsy/dataset/CoLA/test.tsv')

# 加载CoLA数据集
task_name = 'cola'  # 设置为你想要的任务名称
data_dir = '/data/huangsy/dataset/CoLA'  # 设置为你想要的任务名称
args = GlueDataTrainingArguments(task_name=task_name, data_dir=data_dir)
tokenizer = BertTokenizer.from_pretrained('tokenizer.json', do_lower_case=True)
train_dataset = GlueDataset(args=args, tokenizer=tokenizer)
dev_dataset = GlueDataset(args=args, tokenizer=tokenizer, mode='dev')


train_features = glue_convert_examples_to_features(train_examples, tokenizer)
dev_features = glue_convert_examples_to_features(dev_examples, tokenizer)

train_dataset = GlueDataset(train_features)
dev_dataset = GlueDataset(dev_features)

# 创建DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

# 加载数据
data = pd.read_csv('/data/huangsy/dataset/CoLA/train.tsv')
question1 = data['question1'].values
question2 = data['question2'].values
labels = data['is_duplicate'].values

test_data = pd.read_csv('/data/huangsy/dataset/quora-question-pairs/test.csv')
test_question1 = test_data['question1'].values
test_question2 = test_data['question2'].values
sample_submission = pd.read_csv('/data/huangsy/dataset/quora-question-pairs/sample_submission.csv')
test_labels = sample_submission['is_duplicate'].values
 
# 定义超参数
batch_size = 16
epochs = 5
learning_rate = 2e-5
output_path = "./output/Bert-CoLA.model"

# 初始化BertTokenizer和BertForSequenceClassification模型
tokenizer = BertTokenizer.from_pretrained('tokenizer.json', do_lower_case=True)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
seq_len = 384
hidden = 768
layers = 12
attn_heads = 12
#model = BERT(seq_len, hidden=hidden, n_layers=layers, attn_heads=attn_heads)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("hsy: ", len(train_loader))
#trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
#       lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
#       with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

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
