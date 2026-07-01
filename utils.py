import pandas as pd
from coleta_de_dados import acessando_api
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def executar_pipeline(df=None):
    print()
    print("Verificação do Problema das Classificações desbalanceadas")

    if df is None:
        df = acessando_api()

    if df is None:
        print("Não foi possível carregar os dados.")
        return

    df = df.copy()
    print(df["Class"].value_counts(normalize=True))

    # Feature Engineering
    df["Amount_log"] = np.log1p(df["Amount"])

    # Padronização dos valores para média = 0 e desvio padrão = 1
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    # Preparando os dados para o modelo
    x = df.drop("Class", axis=1)
    y = df["Class"]

    # Divisão dos dados em treino e teste
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        stratify=y,
        test_size=0.3,
        random_state=42,
    )

    # Treinamento do modelo de regressão logística
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("")
    print("Aplicação do Logistic Regression")

    #Verificação do desempenho do modelo
    print(classification_report(y_test, y_pred))


    # Avaliação do modelo com curva ROC e AUC
    y_probs = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr)
    plt.title("Roc Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    print("AUC: ", roc_auc_score(y_test, y_probs))

    # Avaliação do modelo com curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision)
    plt.title("Precision-Recall-Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    # Técnicas de balanceamento de classes

    # Undersampling
    fraudes = df[df["Class"] == 1]
    normais = df[df["Class"] == 0].sample(n=len(fraudes), random_state=42)
    df_under = pd.concat([fraudes, normais], axis=0).sample(frac=1, random_state=42)

    print("\nDistribuição após undersampling")
    print(df_under["Class"].value_counts(normalize=True))

    x_under = df_under.drop("Class", axis=1)
    y_under = df_under["Class"]

    x_train_under, x_test_under, y_train_under, y_test_under = train_test_split(
        x_under,
        y_under,
        stratify=y_under,
        test_size=0.3,
        random_state=42,
    )

    model_under = LogisticRegression(max_iter=1000)
    model_under.fit(x_train_under, y_train_under)
    y_pred_under = model_under.predict(x_test_under)
    print("")
    print("Aplicação do Logistic Regression com undersampling")
    print(classification_report(y_test_under, y_pred_under))

    # Oversampling com SMOTE (Synthetic Minority Over-sampling Technique)
    smote = SMOTE()
    x_res, y_res = smote.fit_resample(x, y)

    # Treinamento do modelo de Random Forest
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    rf.fit(x_train_under, y_train_under)
    y_pred_rf = rf.predict(x_test_under)
    print()
    print("Aplicação do Random Forest com undersampling")
    print(classification_report(y_test_under, y_pred_rf))

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(x_train_under, y_train_under)
    y_pred_pipeline = pipeline.predict(x_test_under)

    

    threshold = 0.3
    y_pred_custom = (y_probs > threshold).astype(int)
    print(classification_report(y_test, y_pred_custom))

    print()
    print("fim do fluxo de execução do pipeline")
