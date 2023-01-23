# Frequently used models.

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from sklearn.neural_network import MLPRegressor


def ols(data,datadum):
    lineal = (
      sm.OLS (
        data["Valor_Contrato"],
        datadum)). fit ()
    lineal.summary()
    pred = lineal.get_prediction(datadum)

    data["predict_sup"] = pred.predicted_mean + pred.se_mean
    data["over_pred"] = data["Valor_Contrato"] - data["predict_sup"]
    data["alert_prob"] = data["over_pred"]*100 / data["over_pred"].max()
    return data.sort_values("alert_prob")

def neuralsimple(data,datadum):
    neural = (
      MLPRegressor(
        max_iter=100000,
        alpha=0.0001,
        hidden_layer_sizes=(200,200,200),
        learning_rate="invscaling",
        verbose=True)). fit( datadum,
             data["valoruni"])
    print(neural.loss_curve_)

    pred = pd.Series(
      neural.predict ( datadum ) )

    # TODO : Why are these defined as two separate but equal columns?
    # Either fix that or explain to Jeff why it's appropriate.
    data["predict"] = -(pred-data["valoruni"]) / data["valoruni"]
    data["pred"] = pred

    data["alert_prob"] = data["predict"]*100 / data["predict"].max()
    return data.sort_values("alert_prob")