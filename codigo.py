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

# MAE
mae = np.mean(np.abs(Y_test - Y_pred))

# R²
ss_total = np.sum((Y_test - np.mean(Y_test)) ** 2)
ss_residual = np.sum((Y_test - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

# 11. Mostrar resultados
print("Quantidade total de dados:", len(df))
print("Quantidade de treino:", len(X_train))
print("Quantidade de teste:", len(X_test))
print()

print("Coeficiente angular (a):", a)
print("Coeficiente linear (b):", b)
print()

print("MAE:", mae)
print("R²:", r2)

# 12. Plotar gráfico (ordenado para ficar bonito)
plt.scatter(X_test, Y_test, label="Dados reais")

indices = np.argsort(X_test)
plt.plot(X_test[indices], Y_pred[indices], label="Reta da regressão")

plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("Regressão Linear Simples: BMI x Charges")
plt.legend()

plt.show()