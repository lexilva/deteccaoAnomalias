from utils import  executar_pipeline
from coleta_de_dados import acessando_api
import pandas as pd     
df = acessando_api()

if df is not None:
    print(df.head())
    executar_pipeline(df)

