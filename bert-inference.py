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
import os

# 加载数据
data = pd.read_csv('/data/huangsy/dataset/quora-question-pairs/mini_train.csv')
question1 = data['question1'].values
question2 = data['question2'].values
labels = data['is_duplicate'].values

test_data = pd.read_csv('/data/huangsy/dataset/quora-question-pairs/mini_train.csv')
test_question1 = test_data['question1'].values
test_question2 = test_data['question2'].values
#sample_submission = pd.read_csv('/data/huangsy/dataset/quora-question-pairs/mini_sample_submission.csv')
#test_labels = sample_submission['is_duplicate'].values
test_labels = test_data['is_duplicate'].values
 
# 初始化BertTokenizer和BertForSequenceClassification模型
tokenizer = BertTokenizer.from_pretrained('tokenizer.json', do_lower_case=True)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
seq_len = 384
hidden = 768
layers = 12
attn_heads = 12
batch_size = 16
cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_condition else "cpu")
#model = BERT(seq_len, hidden=hidden, n_layers=layers, attn_heads=attn_heads)

for i in range(len(test_question1)):
    if type(test_question1[i]) != type(test_question1[0]):
        test_question1[i] = str(test_question1[i])
for i in range(len(test_question2)):
    if type(test_question2[i]) != type(test_question2[0]):
        test_question2[i] = str(test_question2[i])
# 将文本转换为BERT输入格式
test_encodings = tokenizer(test_question1.tolist(), test_question2.tolist(), truncation=True, padding=True, max_length=seq_len)

# 创建PyTorch的TensorDataset
test_dataset = TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(test_encodings['token_type_ids']),
    torch.tensor(test_labels.tolist())
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model_path = './bert.model'
if os.path.exists(model_path):
    print("the model_path exits!")
model = torch.load(model_path).to(device)

avg_acc = 0
for data in test_loader:
    input_ids = data[0].to(device)
    attention_mask = data[1].to(device)
    token_type_ids = data[2].to(device)
    labels = data[3].to(device)

    output  = model.forward(input_ids, token_type_ids)
    prediction = torch.argmax(output, dim=2)
    pred_top1 = prediction[:,0]

    for i in range(len(pred_top1)):
        if pred_top1[i] == labels[i]:
            avg_acc += 1 


print("average accuracy: ",avg_acc,  avg_acc/ len(test_loader))

