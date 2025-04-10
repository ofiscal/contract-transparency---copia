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
    from sentence_transformers import util
    from sentence_transformers import SentenceTransformer
    import sklearn
    prompt = "Pauta | Publicidad | Prensa | Periodismo | Periodista | Divulgación | Multimedia publicitaria | Redes Sociales | propaganda en Televisión | publicidad en Radio | cuña Radial | Periódico |propaganda Audiovisual | Video | Revista | Comunicaciones divulgativas"

    def change_dolar(local_value:float,
        change_rate:float):
      return local_value*change_rate
    
    def semantic_search(prompt, df, model, top_k=1000):
        # Generate embeddings for the descriptions
        df['embeddings'] = df['Descripcion del Proceso'].apply(lambda x: model.encode(x, convert_to_tensor=True))

        prompt_embedding = model.encode(prompt, convert_to_tensor=True)
        similarities = df['embeddings'].apply(lambda x: util.pytorch_cos_sim(prompt_embedding, x).item())
        df['similarity'] = similarities
        return df
    
    
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
    records=pd.read_csv(path_data_gener,nrows=1000000,skiprows=lambda x: x in range(1,1000000*numerator)
                        )
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

    records=records[records["value_thousand_dolar"]<=1000]
    
    records=records[records["value_thousand_dolar"]>=0.00001]
    model = SentenceTransformer("tomaarsen/static-similarity-mrl-multilingual-v1")
    records['duracion numero'] = records["Duración del contrato"].str.extract(r'(\d+[.\d]*)').astype(float).replace(np.nan,0)
    records['duracion valor'] = records["Duración del contrato"].str.replace(r'(\d+[.\d]*)','').replace(np.nan,"0")
    records=records.dropna(subset=["Descripcion del Proceso","value_thousand_dolar"])
    
    #records=semantic_search(prompt,records,model)
    #records=records[records['similarity']>0.20]
    #we want to change any unknown variable to other
    #codigo de la entidad
    #Declaramos las variables que vamos a usar en predicción
    variables_y=["value_thousand_dolar"]
    variables_cat=["Ciudad",
                   "Orden","Modalidad de Contratacion",
                   
                   'Tipo de Contrato',
                   "Codigo de Categoria Principal",
                   'duracion valor'

                   ]
    #genera una reducción de duplicados para evitar problema de nuevas variables en estimación
    if False:
        re=pd.DataFrame()
        for i in variables_cat:
            re[i]=records[i].drop_duplicates().reset_index()[i]
        re.to_csv(r"data/limpio/diccionario_duplicados4.csv")
    re=pd.read_csv(r"data/limpio/diccionario_duplicados4.csv",index_col=False)   
           
    for i in variables_cat:
        #print(i)
        records["keep"]=records[i].isin(re[i])
        records[i]=records.apply(lambda row: row[i] if row["keep"] else "otro",axis=1)
        
    joinedpar=records
    categ=pd.get_dummies(records[variables_cat].reset_index(drop=True).astype("str"), dtype=int)
    
    numpdata=joinedpar['Descripcion del Proceso'].apply(lambda x: model.encode(x, convert_to_numpy=True))
    embeddings=pd.DataFrame(np.array([np.array(xi) for xi in numpdata]))

    #variables_reg=["compiledRelease/tender/enquiryPeriod/durationInDays"]

    neuralmodel= sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50,10,),
        activation='relu', solver='adam', alpha=0.01, batch_size='auto',
        learning_rate='adaptive', validation_fraction=0.1,random_state=0,
        verbose=True, early_stopping=True,warm_start=True)
    
    joined = categ.merge(embeddings,left_index=True, right_index=True)
    X=joined
    X.columns = X.columns.astype(str) 
    neuralmodel.fit(X,records["value_thousand_dolar"])
    hat=neuralmodel.predict(X)
    
    neuralmodel.score(X,records["value_thousand_dolar"])
    
    
    
    
    