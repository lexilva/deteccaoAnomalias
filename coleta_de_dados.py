import requests
import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def acessando_api():
    response = requests.get(url)

    if response.status_code == 200:
        df = pd.read_csv(url)
        return df

    print("Erro ao acessar API:", response.status_code)
    return None