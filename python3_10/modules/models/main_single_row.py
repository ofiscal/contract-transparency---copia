"""main.ipynb

Daniel Duque Lozano

Original file is located at
    https://colab.research.google.com/drive/1t6CbdAjL3uM2YDWrsLiQbI96rYxKMDIo
"""
###
###
for numerator in range(0,100):
    import pandas as pd
    import datetime as dt
    import numpy as np
    import os
    def change_dolar(local_value:float,
        change_rate:float):
      return local_value*change_rate
    
    
    
    
    path_change=r"data/sucio/Datos históricos USD_COP.xlsx"
    path_data=r"/content/drive/MyDrive/Hack Corruption - Contractor/datos/records.csv"
    path_CPI=r"data/sucio/USA_CPI.xlsx"
    path_data_gener=r"data/sucio/SECOP_II_-_Contratos_Electr_nicos_20241205.csv"
    
    n=0
    
    
    
    
    #Loading the exchange rate for USD standaricing
    exchange_rate=pd.read_excel(path_change)
    exchange_rate["exchange_rate"]=exchange_rate.apply(lambda row:(row["Máximo"]+row["Mínimo"])/2,
                                              axis=1)
    exchange_rate["Fecha"]=exchange_rate["Fecha"].apply(lambda x:x.year)
    exchange_rate=exchange_rate[["Fecha","exchange_rate"]]
    exchange_ratey=exchange_rate.groupby("Fecha").mean()
    #Loading the dataset  and cleaning dates
    records=pd.read_csv(path_data_gener,nrows=100000,skiprows=lambda x: x in range(1,100000*numerator))
    #
    
    records["Fecha"]=records["Fecha de Firma"].apply(
        lambda x:dt.datetime.strptime(x[:10],"%m/%d/%Y").year if type(x)==str else np.nan
    )
    
    records=records[records["Fecha"]>=2024]
    #Loading CPI for real value updating
    CPI=pd.read_excel(path_CPI)
    CPI=CPI[["Year","Avg"]].dropna()
    last_year=CPI["Avg"][len(CPI)-1]
    
    #merging tables
    records=records.merge(exchange_ratey,how="left",on=["Fecha"])
    
    #We use the last inflation data and normalize everything to the last year
    records=records.merge(CPI,how="left",left_on=["Fecha"],right_on=["Year"])
    records["Avg"]=records["Avg"].fillna(last_year).apply(lambda x:x/last_year)
    try:
        records["Valor del Contrato"]=records["Valor del Contrato"].str.replace(",","")
    except:

        print("no hay problema de string")
    #We scale every value in million dolars
    records["value_thousand_dolar"]=records.apply(lambda row:float(row["Valor del Contrato"])/(row["exchange_rate"]*row["Avg"]*1e3),axis=1)

    records=records[records["value_thousand_dolar"]<=10000]
    
    records=records[records["value_thousand_dolar"]>=0.00001]

    records=records.dropna(subset=["Descripcion del Proceso","value_thousand_dolar"])
    #we want to change any unknown variable to other
    
    #codigo de la entidad
    #Declaramos las variables que vamos a usar en predicción
    variables_y=["value_thousand_dolar"]
    variables_cat=["Ciudad",
                   "Orden","Modalidad de Contratacion",
                   
                   'Tipo de Contrato',
                   "Codigo de Categoria Principal",
                   "Duración del contrato"

                   ]

    joined=records
     
    #variables_reg=["compiledRelease/tender/enquiryPeriod/durationInDays"]
    
    variables_text=["Descripcion del Proceso"]
    
    #joined['compiledRelease/planning/budget/amount/currency']
    
    # -*- coding: utf-8 -*-
    """
    Created on Sun Dec 10 15:13:07 2023
    
    @author: usuario
    """
    
    import pandas as pd
    
    import transformers
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    from tqdm import tqdm
    casa=pd.DataFrame(joined[variables_cat+variables_text].apply(lambda row: ','.join(row.astype(str)), axis=1))
    casa.rename(columns={0: variables_text[0]}, inplace=True)
    casa[variables_y]=  joined[variables_y]
    casa[variables_y]=casa[variables_y].astype(int)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    class BertDataset(Dataset):
        def __init__(self, tokenizer,max_length):
            super(BertDataset, self).__init__()
            self.train_csv=casa
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
                'target': torch.tensor(self.train_csv.iloc[index, 1], dtype=torch.int).to(device)
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
            self.l1_strength=0.00003
    
        def forward(self,ids,mask,token_type_ids):
            _,o2= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
            return o2
            medium2=self.medium2(o2)
            medium3=self.medium3(medium2)
            medium4=self.medium4(medium3)
            medium5=self.medium5(medium4)
            medium6=self.medium5(medium4)
            medium7=self.medium6(medium5)
            medium8=self.drop(medium7)
            out= self.out(medium8)
            return o2
            return medium7
    
        def l1_regularization(self):
            n=0
            l1_loss_example = 0
            #the next structure is to only recognize the layers that are finetuned
            #this makes the exercise faster and allows to update the strength easier
            for param in self.parameters():
                n+=1
                if n>197:
                    l1_loss_example += torch.sum(torch.abs(param))
            return self.l1_strength * (l1_loss_example)
    
    model=BERT()
    
    
    
    
    
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
              results.append(i)
    
    
        print(acc/num)
    
    
        return results
    
    
    
    model.load_state_dict(torch.load(r"model_saved_torch\primero_col.pt"))
    
    
    
    model.to(device)
    
    loss_fn = nn.L1Loss()
    
    #Initialize Optimizer
    optimizer= optim.Adam(model.parameters(),lr= 1e-5)
    
    loss_fn = nn.L1Loss()
    
    #Initialize Optimizer
    optimizer= optim.Adam(model.parameters(),lr= 1e-5)
    
    predicted=pd.DataFrame(predict(100000, dataloader, model, loss_fn, optimizer))
    

    
    
    import pandas as pd
    
    
    from sklearn.metrics import r2_score
    import json
    
    import pickle
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error
    
    #this will be later transplanted to the main file, but for now I need to test
    #the code somewhere and tdidn´t find a better place
    #some paths to where things are
    
    
    #this cant be change, they are how rn where train (will be changed manually)
    
    
    
    #takes only the regresive variables
    
    
    data_value=joined[variables_y]
    
    try:
        with open(r'model_saved_torch\modelxgboostnrow.pkl', "rb") as input_file:
          reg = pickle.load(input_file)
    except:
        ...
    

    
    data_categ=predicted
    

    data_categ_train, data_categ_test, data_value_train, data_value_test = train_test_split(
         data_categ, data_value, test_size=0.15, random_state=1,shuffle=True)
    
    
    dtrain_reg = xgb.DMatrix(data_categ_train, data_value_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(data_categ_test, data_value_test, enable_categorical=True)
    pure_data = xgb.DMatrix(data_categ, data_value, enable_categorical=True)
    
    
    
    if True:
        for i in range(0,1):
            n = 50
            params = {"objective": "reg:pseudohubererror","reg_alpha":60,"reg_lambda":60
                      ,"rate_drop":0.1,"gpu_id":0,'tree_method':'gpu_hist'}
            evals = [(dtest_reg, "validation"),(dtrain_reg, "train") ]
            reg = xgb.train(
               params=params,
               dtrain=dtrain_reg,
               num_boost_round=n,
               evals=evals,
               verbose_eval=25,
               xgb_model=reg,
               early_stopping_rounds=2
               )
            pickle.dump(reg, open(r'model_saved_torch\modelxgboostnrow.pkl','wb'))
    else:
        ...

    predict=reg.predict(pure_data)
    

    joined["predict"]=predict
    joined["predict"]=joined["predict"].apply(lambda x:x if x>0 else 0.2)
    
    
    
    import pandas as pd
    from scipy.stats import norm
    from scipy import stats
    
    data1=joined
    
    size=0.5
    
    data1["range-"]=data1["predict"].apply(lambda x:x*(1-size))
    data1["range+"]=data1["predict"].apply(lambda x:x*(1+size))
    
    data1["in_range"]=data1.apply(
        lambda row:1 if row["value_thousand_dolar"]<row["range+"] and row["value_thousand_dolar"]>row["range-"] else 0,
        axis=1)
    
    data1["in_range"].mean()
    
    data1["perc_error"]=data1.apply(
        lambda row:(row["value_thousand_dolar"]-row["predict"])/row["predict"],
        axis=1)
    ["perc_error"]
    
    
    
    try:
        with open(r'model_saved_torch\modelxgboosterrorrow.pkl', "rb") as input_file:
          reg = pickle.load(input_file)
    except:
        ...
    
    
    
    data_categ=predicted
    
    data_value=data1["perc_error"]
    data_categ_train, data_categ_test, data_value_train, data_value_test = train_test_split(
         data_categ, data_value, test_size=0.15, random_state=1,shuffle=True)
    
    
    dtrain_reg = xgb.DMatrix(data_categ_train, data_value_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(data_categ_test, data_value_test, enable_categorical=True)
    pure_data = xgb.DMatrix(data_categ, data_value, enable_categorical=True)
    
    
    
    if True:
        for i in range(0,1):
            n = 50
            params = {"objective": "reg:pseudohubererror","reg_alpha":60,"reg_lambda":60
                      ,"rate_drop":0.1,"gpu_id":0,'tree_method':'gpu_hist'}
            evals = [(dtest_reg, "validation"),(dtrain_reg, "train") ]
            reg = xgb.train(
               params=params,
               dtrain=dtrain_reg,
               num_boost_round=n,
               evals=evals,
               verbose_eval=25,
               xgb_model=reg,
               early_stopping_rounds=2
               )
            pickle.dump(reg, open(r'model_saved_torch\modelxgboosterrorrow.pkl','wb'))
    else:
        ...

    predicterr=reg.predict(pure_data)
    

    data1["predicterr"]=predicterr
    
    num_contracts=data1["perc_error"].count()
    
    num_contracts
    
    contr_inrange=data1["perc_error"][data1["in_range"]==1].count()
    print(contr_inrange/num_contracts)



    
    mu,sigma,kurt=stats.t.fit(data1["perc_error"][data1["in_range"]==1])
    
    mu
    
    data1["perc_error"][data1["in_range"]==1].hist(bins=100,density=True)
    
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    
    n, bins, patches = plt.hist(
        data1["perc_error"][data1["in_range"]==1],bins=100,density=True
        )
    
    # add a 'best fit' line
    y = stats.t.pdf( bins, mu, sigma,kurt)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    
    distribution=stats.t( mu, sigma,kurt)
    
    distribution
    
    help(distribution.pdf)
    
    mu
    
    distribution.pdf(-0.151242)
    
    for i in range(0,100):
      print(distribution.pdf(i/100-0.5))
    
    data1["likelihood"]=data1["perc_error"].apply(lambda x: distribution.pdf(x))
    
    data1["likelihood"].hist(bins=1000)
    
    #data1.to_excel(r"data/resultados/col_tria.xlsx")
    
    data1[["value_thousand_dolar","predict"]]
    
    import sklearn as sk
    print(sk.metrics.r2_score(data1["value_thousand_dolar"],data1["predict"]))
    data1["miles de dolares sobre estimación"]=data1["value_thousand_dolar"]-data1["predict"]
    """
    try:
        data1.to_excel(r"data/resultados/col_tria"+str(numerator)+".xlsx")
    except Exception as e:
        print(e)
    """
"""
print(data1.count()[0])
print(data1[data1["likelihood"]<0.05].count()[0])

data1[data1["likelihood"]<0.01]

import seaborn
seaborn.histplot(subset[["predict","value_thousand_dolar"]],x="predict",y="value_thousand_dolar",cbar=True,bins=1000)

plt.scatter(subset["value_thousand_dolar"],subset["predict"])
z = np.polyfit(subset["value_thousand_dolar"], subset["predict"], 1)
p = np.poly1d(z)
plt.plot(subset["value_thousand_dolar"],p(subset["value_thousand_dolar"]),"r--")

"""
