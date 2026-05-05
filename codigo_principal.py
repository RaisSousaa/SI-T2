import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from normalizacao import normalizar_zscore


def carregar_dados():
    df = pd.read_csv("insurance.csv")

    df_preparado = pd.get_dummies(
        df,
        columns=['sex', 'smoker', 'region'],
        drop_first=True
    )

    X = df_preparado.drop('charges', axis=1).values.astype(float)
    Y = df_preparado['charges'].values

    # Adiciona coluna de 1 para o intercepto
    X = np.c_[np.ones(X.shape[0]), X]

    return X, Y


def descida_gradiente(X, Y, taxa_aprendizado, quantidade_iteracoes):
    m = len(Y)
    quantidade_colunas = X.shape[1]

    pesos = np.zeros(quantidade_colunas)
    lista_custos = []

    for i in range(quantidade_iteracoes):
        previsao = np.dot(X, pesos)
        erro = previsao - Y

        gradiente = (1 / m) * np.dot(X.T, erro)
        pesos = pesos - taxa_aprendizado * gradiente

        custo = (1 / (2 * m)) * np.sum(erro ** 2)
        lista_custos.append(custo)

    return pesos, lista_custos


def calcular_metricas(Y_teste, Y_previsto):
    ss_total = np.sum((Y_teste - np.mean(Y_teste)) ** 2)
    ss_residual = np.sum((Y_teste - Y_previsto) ** 2)

    r2 = 1 - (ss_residual / ss_total)
    mae = np.mean(np.abs(Y_teste - Y_previsto))

    return r2, mae

def grafico_custo(custos, titulo):
    plt.figure(figsize=(10, 4))
    plt.plot(custos, color='red', linewidth=2)

    plt.title(titulo)
    plt.xlabel("Número de Iterações")
    plt.ylabel("Custo")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def grafico_real_previsto(Y_teste, Y_previsto, r2, titulo):
    plt.figure(figsize=(10, 6))

    plt.scatter(
        Y_teste,
        Y_previsto,
        alpha=0.5,
        color='royalblue',
        label='Previsões do Modelo'
    )

    limite = max(Y_teste.max(), Y_previsto.max())

    plt.plot(
        [0, limite],
        [0, limite],
        color='red',
        linewidth=3,
        label='Linha de Perfeição'
    )

    plt.xlabel("Valores Reais (Charges)")
    plt.ylabel("Previsões do Modelo (Charges)")
    plt.title(f"{titulo} | R² = {r2:.2f}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# Parte principal
X, Y = carregar_dados()

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42
)

# Cópias para o modelo sem normalização
X_train_sem_norm = X_train.copy()
X_test_sem_norm = X_test.copy()

# Cópias para o modelo com normalização
X_train_com_norm = X_train.copy()
X_test_com_norm = X_test.copy()

# Aplica normalização Z-score manual
X_train_com_norm, X_test_com_norm = normalizar_zscore(
    X_train_com_norm,
    X_test_com_norm
)

taxa_aprendizado = 0.001
quantidade_iteracoes = 1000

# Modelo SEM normalização
pesos_sem_norm, custos_sem_norm = descida_gradiente(
    X_train_sem_norm,
    Y_train,
    taxa_aprendizado,
    quantidade_iteracoes
)

Y_pred_sem_norm = np.dot(X_test_sem_norm, pesos_sem_norm)

r2_sem_norm, mae_sem_norm = calcular_metricas(
    Y_test,
    Y_pred_sem_norm
)

# Modelo COM normalização
pesos_com_norm, custos_com_norm = descida_gradiente(
    X_train_com_norm,
    Y_train,
    taxa_aprendizado,
    quantidade_iteracoes
)

Y_pred_com_norm = np.dot(X_test_com_norm, pesos_com_norm)

r2_com_norm, mae_com_norm = calcular_metricas(
    Y_test,
    Y_pred_com_norm
)

print("RESULTADOS SEM NORMALIZAÇÃO")
print(f"R²: {r2_sem_norm:.4f}")
print(f"MAE: {mae_sem_norm:.2f}")

print("\nRESULTADOS COM NORMALIZAÇÃO")
print(f"R²: {r2_com_norm:.4f}")
print(f"MAE: {mae_com_norm:.2f}")

# Gráficos exigidos
grafico_custo(
    custos_sem_norm,
    "Curva de Perda - Sem Normalização"
)

grafico_custo(
    custos_com_norm,
    "Curva de Perda - Com Normalização Z-score"
)

grafico_real_previsto(
    Y_test,
    Y_pred_sem_norm,
    r2_sem_norm,
    "Predito vs Real - Sem Normalização"
)

grafico_real_previsto(
    Y_test,
    Y_pred_com_norm,
    r2_com_norm,
    "Predito vs Real - Com Normalização Z-score"
)