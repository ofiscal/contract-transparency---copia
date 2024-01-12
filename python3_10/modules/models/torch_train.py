# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:13:07 2023

@author: usuario
"""

import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

casa=pd.read_csv(r"C:\Users\usuario\Documents\contract-transparency-copia\data\limpio\value.csv",skiprows=1,nrows=18000,header=None)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BertDataset(Dataset):
    def __init__(self, tokenizer,max_length):
        super(BertDataset, self).__init__()
        self.train_csv=pd.read_csv(r"C:\Users\usuario\Documents\contract-transparency-copia\data\limpio\value.csv",skiprows=1,nrows=18000,header=None)
        self.tokenizer=tokenizer
        self.target=self.train_csv.iloc[:,1]
        self.max_length=max_length

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):

        text1 = self.train_csv.iloc[index,0]

        inputs = self.tokenizer.encode_plus(
            text1 ,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long).to(device),
            'mask': torch.tensor(mask, dtype=torch.long).to(device),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device),
            'target': torch.tensor(self.train_csv.iloc[index, 1], dtype=torch.long).to(device)
            }
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-multilingual-cased")

dataset= BertDataset(tokenizer, max_length=100)
batch_size=64
dataloader=DataLoader(dataset=dataset,batch_size=batch_size)

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-multilingual-cased")
        self.medium2 = nn.Linear(768, 15000)
        self.medium3 = nn.ReLU()
        self.medium4=nn.Linear(15000,15000,bias=True)
        self.medium5 = nn.Sigmoid()
        self.medium6=nn.Linear(15000,1500,bias=True)
        self.medium7 = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(1500, 1,bias=True)
        self.l1_strength=0.00001

    def forward(self,ids,mask,token_type_ids):
        _,o2= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        medium2=self.medium2(o2)
        medium3=self.medium3(medium2)
        medium4=self.medium4(medium3)
        medium5=self.medium5(medium4)
        medium6=self.medium5(medium4)
        medium7=self.medium6(medium5)
        medium8=self.drop(medium7)
        out= self.out(medium8)
        return out
    
    def l1_regularization(self):
        l1_loss_example = 0
        for param in self.parameters():
            l1_loss_example += torch.sum(torch.abs(param))
        return self.l1_strength * (l1_loss_example-13000000)

model=BERT()

loss_fn = nn.L1Loss()

#Initialize Optimizer
optimizer= optim.Adam(model.parameters(),lr= 1e-5)


for param in model.bert_model.parameters():
    param.requires_grad = False
    
    
    
    
def finetune(epochs,dataloader,model,loss_fn,optimizer):
  model.train()
  for  epoch in range(epochs):
    print(epoch)
    num=1
    acc=0
    loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
    for batch, dl in loop:
      ids=dl['ids']
      token_type_ids=dl['token_type_ids']
      mask= dl['mask']
      label=dl['target']
      label = label.unsqueeze(1)

      optimizer.zero_grad()

      output=model(
          ids=ids,
          mask=mask,
          token_type_ids=token_type_ids)
      label = label.type_as(output)

      loss=loss_fn(output,label)
      loss+=model.l1_regularization()
      loss.backward()

      optimizer.step()

      pred = output

      num_correct = sum((a-b)**2 for a, b in zip(pred, label) )/batch_size
      num_samples = pred.shape[0]
      accuracy = num_correct/(1e-6*batch_size)

      #print(output[:4])
      #print(label[:4])
      #print(label[:2]-output[:2])
      # Show progress while training
      loop.set_description(f'Epoch={epoch}/{epochs}')
      loop.set_postfix(loss=loss.item(),acc=accuracy)
      acc+=loss.item()
      num+=1
    if True:
      torch.save(model.state_dict(), r"C:\Users\usuario\Documents\contract-transparency-copia\model_saved_torch\primero.pt")
    if epoch%100==0:
      torch.save(model.state_dict(), r"C:\Users\usuario\Documents\contract-transparency-copia\model_saved_torch\primero_backup.pt")
    print(acc/num)


  return model  
    
def predict(epochs,dataloader,model,loss_fn,optimizer):
    num=1
    acc=0
    loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
    results=[]
    for batch, dl in loop:
      ids=dl['ids']
      token_type_ids=dl['token_type_ids']
      mask= dl['mask']
      label=dl['target']
      label = label.unsqueeze(1)
    
      optimizer.zero_grad()
    
      output=model(
          ids=ids,
          mask=mask,
          token_type_ids=token_type_ids)
      label = label.type_as(output)
      for i in output.cpu().detach().numpy():
          results.append(i[0])
    
    if False:
      torch.save(model.state_dict(), r"C:\Users\usuario\Documents\contract-transparency-copia\model_saved_torch\primero.pt")
      
    print(acc/num)
    
    
    return results 
    
model.load_state_dict(torch.load(r"C:\Users\usuario\Documents\contract-transparency-copia\model_saved_torch\primero.pt"))


  
model.to(device)  

if False:
    predicted=predict(100000, dataloader, model, loss_fn, optimizer)
    to_series=pd.Series(predicted,name="resultados")
    casa["predict"]=to_series
else:
  
    model=finetune(10000, dataloader, model, loss_fn, optimizer)    
    

casa.to_excel(r"C:\Users\usuario\Documents\contract-transparency-copia\data\resultados\torch2.xlsx")    
    

#