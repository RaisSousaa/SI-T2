import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Carregar a base
df = pd.read_csv("insurance.csv")

# 2. Selecionar colunas
df = df[['bmi', 'charges']]

# 3. Remover valores nulos
df = df.dropna()

# 4. Separar X e Y
X = df['bmi'].values
Y = df['charges'].values

# 5. Dividir treino e teste (70/30)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, shuffle=True, random_state=42
)

# 6. Calcular médias (treino)
x_mean = np.mean(X_train)
y_mean = np.mean(Y_train)

# 7. Calcular coeficiente angular (a)
numerador = np.sum((X_train - x_mean) * (Y_train - y_mean))
denominador = np.sum((X_train - x_mean) ** 2)
a = numerador / denominador

# 8. Calcular coeficiente linear (b)
b = y_mean - a * x_mean

# 9. Fazer previsões
Y_pred = a * X_test + b

# 10. Calcular métricas

# Erro Médio Absoluto (MAE)
mae = np.mean(np.abs(Y_test - Y_pred))

# Coeficiente de Determinação (R²)
# Representa quanto da variação de 'charges' é explicada pelo 'bmi'
media_y_teste = np.mean(Y_test)
soma_total_quadrados = np.sum((Y_test - media_y_teste) ** 2)
soma_residuos = np.sum((Y_test - Y_pred) ** 2)
r2 = 1 - (soma_residuos / soma_total_quadrados)

# 11. Mostrar resultados
print(f"--- Relatório do Modelo ---")
print(f"Registros (Treino/Teste): {len(X_train)} / {len(X_test)}")
print(f"Modelo: Y = {a:.4f} * X + {b:.4f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f} ({r2*100:.2f}% de variância explicada)")

# 12. Plotar gráfico
plt.figure(figsize=(10,6))

# Dados reais do teste
plt.scatter(X_test, Y_test, alpha=0.5, color='royalblue', label="Dados reais (Teste)")

# Reta da regressão - Usando apenas os extremos para uma linha limpa
x_reta = np.array([X_test.min(), X_test.max()])
y_reta = a * x_reta + b
plt.plot(x_reta, y_reta, color='red', linewidth=3, label="Modelo de Regressão")

plt.xlabel("BMI (IMC)")
plt.ylabel("Charges (Custos)")
plt.title("Impacto do IMC nos Custos Médios de Saúde")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()