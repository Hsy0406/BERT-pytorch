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

# 加载数据
torch.cuda.set_device(1)
data = pd.read_csv('/data/huangsy/dataset/quora-question-pairs/mini_train.csv')
question1 = data['question1'].values
question2 = data['question2'].values
labels = data['is_duplicate'].values

test_data = pd.read_csv('/data/huangsy/dataset/quora-question-pairs/mini_test.csv')
test_question1 = test_data['question1'].values
test_question2 = test_data['question2'].values
sample_submission = pd.read_csv('/data/huangsy/dataset/quora-question-pairs/mini_sample_submission.csv')
test_labels = sample_submission['is_duplicate'].values
 
# 初始化BertTokenizer和BertForSequenceClassification模型
tokenizer = BertTokenizer.from_pretrained('tokenizer.json', do_lower_case=True)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
seq_len = 384
hidden = 768
layers = 12
attn_heads = 12
#model = BERT(seq_len, hidden=hidden, n_layers=layers, attn_heads=attn_heads)

# 划分训练集和验证集
train_q1, val_q1, train_q2, val_q2, train_labels, val_labels = train_test_split(question1, question2, labels, test_size=0.2, random_state=42)

for i in range(len(train_q1)):
    if type(train_q1[i]) != type(train_q1[0]):
        train_q1[i] = str(train_q1[i])
for i in range(len(train_q2)):
    if type(train_q2[i]) != type(train_q2[0]):
        train_q2[i] = str(train_q2[i])
for i in range(len(val_q1)):
    if type(val_q1[i]) != type(val_q1[0]):
        val_q1[i] = str(val_q1[i])
for i in range(len(val_q2)):
    if type(val_q2[i]) != type(val_q2[0]):
        val_q2[i] = str(val_q2[i])
for i in range(len(test_question1)):
    if type(test_question1[i]) != type(test_question1[0]):
        test_question1[i] = str(test_question1[i])
for i in range(len(test_question2)):
    if type(test_question2[i]) != type(test_question2[0]):
        test_question2[i] = str(test_question2[i])
# 将文本转换为BERT输入格式
train_encodings = tokenizer(train_q1.tolist(), train_q2.tolist(), truncation=True, padding=True, max_length=seq_len)
print(list(train_encodings.keys()))
val_encodings = tokenizer(val_q1.tolist(), val_q2.tolist(), truncation=True, padding=True, max_length=seq_len)
test_encodings = tokenizer(test_question1.tolist(), test_question2.tolist(), truncation=True, padding=True, max_length=seq_len)

# 创建PyTorch的TensorDataset
train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_encodings['token_type_ids']),
    torch.tensor(train_labels.tolist())
)
val_dataset = TensorDataset(
    torch.tensor(val_encodings['input_ids']),
    torch.tensor(val_encodings['attention_mask']),
    torch.tensor(val_encodings['token_type_ids']),
    torch.tensor(val_labels.tolist())
)
test_dataset = TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(test_encodings['token_type_ids']),
    torch.tensor(test_labels.tolist())
)

# 定义超参数
batch_size = 16
epochs = 5
learning_rate = 2e-5
output_path = "./output/Bert.model"

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("hsy: ", len(train_loader))
#trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
#       lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
#       with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

print("Creating BERT Trainer")
is_train = False
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
