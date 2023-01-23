import python3_10.modules.cleaning as cleaning
import python3_10.modules.models.recurrents as recurrent
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
import json
import io

def main():
    path_to_models=r"trainedmodels"
    pathToData = r"data/sucio/SECOP_II_-_Contratos_Electr_nicos.csv"
    subsetsize=100000
    max_length=200
    #loading the cleaned dataset
    data_pred=cleaning.secop_for_prediction(pathToData=pathToData)
    #entrenar los modelos o continuar su entrenamiento
    if True:
        data_train=cleaning.secop2_general(pathToData =pathToData)
        model_rn,tokenizer=recurrent.recurrent_train(
            X=data_train["Descripcion del Proceso"],
            Y=data_train["Valor del Contrato"])
        model_rn.save(path_to_models+"\model1_rn.h5") 
        tokenizer_json=tokenizer.to_json()
        with io.open("tokenizer.json","w",encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_json,ensure_ascii=False))
    else:
        #call the pre trained models
        model_rn=keras.models.load_model(path_to_models+"\model_rn.h5")
        tokenizer=tokenizer_from_json.load(path_to_models+"\tokenizer.json")
    #predict using the models
    
    #we generate series from the description text using the tokens instead of word
    sequences=tokenizer.texts_to_sequences(data_pred["Descripcion del Proceso"])
    #we padd them to make the sequences of equal length
    padded=pad_sequences(sequences,maxlen=max_length)
    data_pred["rn_predict"]=model_rn.predict(x=padded)
    
    
    
    #ejecutar los modelos sobre el nuebo conjunto
    
    #

if __name__=="__main__":
    main()
    