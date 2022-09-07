import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

## for printing image
import matplotlib.pyplot as plt
import numpy as np

import json

BATCH_SIZE = 8

class HandshapeDataset(Dataset):
    def __init__(self, json_file):
        data = json.load(open(json_file))
        
        self.hs_coords = []
        self.hs = []
        for sign, video in data.items():
            hs = video["lexical"]["handshape"]

            for frame in video["frames"]:
                if not frame["right_handshape"]:
                    continue
                hand_coords = frame["right_handshape"]["distances"]
                self.hs_coords.append(np.array(hand_coords))
                self.hs.append(hs)
                
    def __len__(self):
        return len(self.hs_coords)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        coords = self.hs_coords[idx]
        label = self.hs[idx]
        
        return { "coords" : coords, "label" : label }

hs_data = HandshapeDataset("asllex_single_handshape.json")

hs_dataloader = DataLoader(hs_data, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=0)

hidden_layer_size = 128
class Neural_HS_Classifier(nn.Module):
    def __init__(self):
        super(Neural_HS_Classifier, self).__init__()
        self.d1 = nn.Linear(19, 128)
        self.dropout = nn.Dropout(p=0.2)
        self.d2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.d3 = nn.Linear(256, 50)
        
    def forward(self, x):
        x = x.flatten(start_dim = 1)
        x = self.d1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.droupout2(x)
        logits = self.d3(x)
        out = F.softmax(logits, dim=0)
        return out

learning_rate = 0.001
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Neural_HS_Classifier()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

def get_accuracy(output, target, batch_size):
    corrects = (torch.max(output, 1)[1].view \
                    (target.size()).data == target.data).sum()
    acc = 100.0 * corrects/batch_size
    return acc.item()
model = model.float()

for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0
    
    model = model.train()
    
    for i, batch in enumerate(hs_dataloader):
        hs_coords = batch["coords"]
        hs_labels = batch["label"]
        
        predictions = model(hs_coords.float())
                
        loss = criterion(predictions, hs_labels)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        train_running_loss += loss.detach().item()
        
        train_acc += get_accuracy(predictions,hs_labels,BATCH_SIZE)
        
    model.eval()
    print('Epoch %d | Loss: %.4f | Train Accuracy: %.2f' \
         %(epoch, train_running_loss / i, train_acc / i))