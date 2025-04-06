from sklearn.datasets import load_diabetes
import pandas as pd

diabetes = load_diabetes()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
df.head()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X_bmi = df[['bmi']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X_bmi, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

y_pred_simple = model_simple.predict(X_test)

r2_simple = r2_score(y_test, y_pred_simple)
mae_simple = mean_absolute_error(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)

print("Basit Lineer Regresyon")
print("R² Skoru:", r2_simple)
print("MAE:", mae_simple)
print("MSE:", mse_simple)

X_multi = df.drop(columns='target')

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

y_pred_multi = model_multi.predict(X_test_m)

r2_multi = r2_score(y_test_m, y_pred_multi)
mae_multi = mean_absolute_error(y_test_m, y_pred_multi)
mse_multi = mean_squared_error(y_test_m, y_pred_multi)

print("\nÇoklu Lineer Regresyon")
print("R² Skoru:", r2_multi)
print("MAE:", mae_multi)
print("MSE:", mse_multi)
