import numpy as np

def normalizar_zscore(X_train, X_test):
    quantidade_colunas = X_train.shape[1]

    for coluna in range(1, quantidade_colunas):
        media = np.mean(X_train[:, coluna])
        desvio = np.std(X_train[:, coluna])

        if desvio == 0:
            desvio = 1

        X_train[:, coluna] = (X_train[:, coluna] - media) / desvio
        X_test[:, coluna] = (X_test[:, coluna] - media) / desvio

    return X_train, X_test