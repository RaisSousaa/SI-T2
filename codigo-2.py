import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Carregar e preparar a base
df = pd.read_csv("insurance.csv")

# Transformar variáveis categóricas em numéricas (One-Hot Encoding)
df_preparado = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# 2. Separar X (todas as características) e Y (charges)
X = df_preparado.drop('charges', axis=1).values
Y = df_preparado['charges'].values

# Adicionar coluna de 1s para o intercepto (bias)
X = np.c_[np.ones(X.shape[0]), X]

# 3. Dividir treino e teste (70/30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 4. Escalonamento (Crucial para o Gradiente Descendente)
scaler = StandardScaler()
X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])
X_test[:, 1:] = scaler.transform(X_test[:, 1:])

# 5. Implementação da Descida do Gradiente Manual
def descida_gradiente(X, y, learning_rate, n_iteracoes):
    m = len(y)
    n_features = X.shape[1]
    theta = np.zeros(n_features) # Inicializa pesos com zero
    historico_custo = []
    
    for _ in range(n_iteracoes):
        previsao = np.dot(X, theta)
        erro = previsao - y
        gradiente = (1/m) * np.dot(X.T, erro)
        
        # Atualização dos pesos (E fixo / Learning Rate)
        theta = theta - (learning_rate * gradiente)
        
        # Cálculo da perda (MSE) para histórico
        custo = (1/(2*m)) * np.sum(erro**2)
        historico_custo.append(custo)
        
    return theta, historico_custo

# 6. Treinamento com Hiperparâmetros
E_FIXO = 0.01 # Taxa de aprendizado
N_ITERACAO = 1000 # Número de iterações

theta_final, custos = descida_gradiente(X_train, Y_train, E_FIXO, N_ITERACAO)

# 7. Previsões e Métricas
Y_pred = np.dot(X_test, theta_final)

# Cálculo do R² (Média de acertos/Qualidade do modelo)
ss_total = np.sum((Y_test - np.mean(Y_test)) ** 2)
ss_residual = np.sum((Y_test - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

print(f"R² (Acurácia do Modelo): {r2:.4f}")
print(f"MAE: {np.mean(np.abs(Y_test - Y_pred)):.2f}")

# --- GRÁFICO 1: REAL VS PREVISTO (IGUAL AO ANTERIOR) ---
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5, color='royalblue', label='Previsões do Modelo')

# Linha de perfeição
limite = max(Y_test.max(), Y_pred.max())
plt.plot([0, limite], [0, limite], color='red', linewidth=3, label='Linha de Perfeição')

plt.xlabel("Valores Reais (Charges)")
plt.ylabel("Previsões do Modelo (Charges)")
plt.title(f"Regressão Múltipla: Real vs Previsto | R² = {r2:.2f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# --- GRÁFICO 2: CURVA DE APRENDIZADO (HISTÓRICO DE CUSTO) ---
plt.figure(figsize=(10, 4))
plt.plot(custos, color='red', linewidth=2) # Vermelho para representar o 'esforço' do modelo

plt.title("Curva de Aprendizado (Descida do Gradiente)")
plt.xlabel("Número de Iterações")
plt.ylabel("Custo (Erro Quadrático Médio)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()