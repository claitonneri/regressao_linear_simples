from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar uma instância do FastAPI
app = FastAPI()


# Criar uma classe que terá os dados de entrada para o modelo de regressão
class request_body(BaseModel):
    horas_estudo: float


# Carregar o modelo de regressão treinado para fazer previsões
modelo = joblib.load("./reg_model_pontuacao.pkl")


@app.post("/predict")
def predict(data: request_body):
    # Preparar os dados para a predição
    input_feature = [[data.horas_estudo]]

    # Realizar a predição usando o modelo carregado
    y_pred = modelo.predict(input_feature)[0].astype(int)

    return {"pontuacao_prevista": y_pred.tolist()}
