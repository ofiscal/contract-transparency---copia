import python3_10.modules.cleaning as cleaning
import python3_10.modules.models.recurrents as recurrent
import python3_10.modules.models.Transformers as Transformers
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
import python3_10.modules.post_processing.distance_functions as di_fu
import json
import io
from sklearn.metrics import r2_score




def main():
    #some paths to where things are
    path_to_models=r"trainedmodels"
    pathToData = r"data/sucio/SECOP_II_-_Contratos_Electr_nicos.csv"
    path_to_result=r"data\resultados"
    #this cant be change, they are how rn where train
    subsetsize=500000
    max_length=200
    
    mean=163740447
    ssd=1716771980
    #loading the cleaned dataset
    data_pred=cleaning.secop_for_prediction(pathToData=pathToData)
    #entrenar los modelos o continuar su entrenamiento
    if True:
        data_train=cleaning.secop2_general(pathToData =pathToData,subsetsize=subsetsize)
        model_rn,tokenizer,mean,ssd=Transformers.transformer_train(
            X=data_train["Descripcion del Proceso"],
            Y=data_train["Valor del Contrato"])
        model_rn.save(path_to_models+"\model1_tr.h5") 
        tokenizer_json=tokenizer.to_json()
        with io.open(path_to_models+"tokenizer.json","w",encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_json,ensure_ascii=False))
        print(mean, " ",ssd)
    else:
        #call the pre trained models
        model_rn=keras.models.load_model(path_to_models+"\model_rn.h5")
        file=open(path_to_models+r"\tokenizer.json")
        pre_token = json.load(file)   
        tokenizer=tokenizer_from_json(pre_token)
    #predict using the models
    
    #we generate series from the description text using the tokens instead of word
    sequences=tokenizer.texts_to_sequences(data_pred["Descripcion del Proceso"])
    #we padd them to make the sequences of equal length
    padded=pad_sequences(sequences,maxlen=max_length)
    data_pred["rn_predict"]=model_rn.predict(x=padded)
    data_pred["rn_denormalized"]=data_pred["rn_predict"].apply(lambda x: x*ssd-mean)
    data_pred["percentage_error"]=di_fu.distance_percentage(
        X1=data_pred["Valor del Contrato"],
        X2=data_pred["rn_denormalized"])
    data_pred.to_excel(path_to_result+r"\results2.xlsx")
    r2_score(data_pred["Valor del Contrato"],data_pred["rn_denormalized"])
    
    #ejecutar los modelos sobre el nuebo conjunto
    
    #

if __name__=="__main__":
    main()
    