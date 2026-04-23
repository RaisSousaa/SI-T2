import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Carregar e preparar a base
df = pd.read_csv("insurance.csv")

# Transformar variáveis categóricas em numéricas
df_preparado = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# 2. Separar X e Y
X = df_preparado.drop('charges', axis=1).values.astype(float)
Y = df_preparado['charges'].values

# Adicionar coluna de 1s para o intercepto
X = np.c_[np.ones(X.shape[0]), X]

# 3. Dividir treino e teste (70/30)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

# 4. Normalização manual Min-Max
# Não normalizamos a coluna do intercepto, por isso usamos [:, 1:]
x_min = np.min(X_train[:, 1:], axis=0)
x_max = np.max(X_train[:, 1:], axis=0)

intervalo = x_max - x_min
intervalo[intervalo == 0] = 1  # evita divisão por zero

X_train[:, 1:] = (X_train[:, 1:] - x_min) / intervalo
X_test[:, 1:] = (X_test[:, 1:] - x_min) / intervalo

# 5. Descida do gradiente manual
def descida_gradiente(X, y, learning_rate, n_iteracoes):
    m = len(y)
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    historico_custo = []

    for _ in range(n_iteracoes):
        previsao = np.dot(X, theta)
        erro = previsao - y
        gradiente = (1 / m) * np.dot(X.T, erro)

        # Atualização dos pesos
        theta = theta - (learning_rate * gradiente)

        # Cálculo do custo
        custo = (1 / (2 * m)) * np.sum(erro ** 2)
        historico_custo.append(custo)

    return theta, historico_custo

# 6. Treinamento
E_FIXO = 0.01
N_ITERACAO = 1000

theta_final, custos = descida_gradiente(X_train, Y_train, E_FIXO, N_ITERACAO)

# 7. Previsões e métricas
Y_pred = np.dot(X_test, theta_final)

ss_total = np.sum((Y_test - np.mean(Y_test)) ** 2)
ss_residual = np.sum((Y_test - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

mae = np.mean(np.abs(Y_test - Y_pred))

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f}")

# Gráfico 1: Real vs Previsto
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5, color='royalblue', label='Previsões do Modelo')

limite = max(Y_test.max(), Y_pred.max())
plt.plot([0, limite], [0, limite], color='red', linewidth=3, label='Linha de Perfeição')

plt.xlabel("Valores Reais (Charges)")
plt.ylabel("Previsões do Modelo (Charges)")
plt.title(f"Regressão Múltipla: Real vs Previsto | R² = {r2:.2f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Gráfico 2: Curva de aprendizado
plt.figure(figsize=(10, 4))
plt.plot(custos, color='red', linewidth=2)

plt.title("Curva de Aprendizado (Descida do Gradiente)")
plt.xlabel("Número de Iterações")
plt.ylabel("Custo (Erro Quadrático Médio)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()