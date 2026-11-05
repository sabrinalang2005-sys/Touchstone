import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize

url_main = "https://raw.githubusercontent.com/sophiaAcademics/BDA_Excel/main/StudentData/Assessments/Unit6/TechGearTouchstone/techgear_sales_data.xlsx"
url_forecast = "https://raw.githubusercontent.com/sophiaAcademics/BDA_Excel/main/StudentData/Assessments/Unit6/TechGearTouchstone/techgear_sales_data_monthly.xlsx"

# Note: In an actual environment, make sure these URLs point to valid Excel files or mock the dataframes.
sales_df = pd.read_excel(url_main)
forecast_df = pd.read_excel(url_forecast)

print("=== QUESTION 1: DATA STRUCTURES & AVERAGES ===")
print(f"Dataset Shape (Rows, Columns): {sales_df.shape}")
print(f"Average Facebook Spend: ${sales_df['Ad_Spend_Facebook'].mean():.2f}")
print(f"Average Instagram Spend: ${sales_df['Ad_Spend_Instagram'].mean():.2f}")




print("=== QUESTION 2: DATA VISUALIZATION ===")
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.regplot(data=sales_df, x='Ad_Spend_Facebook', y='Sales', scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.title("Sales vs. Facebook Ad Spend")
plt.xlabel("Facebook Ad Spend ($)")
plt.ylabel("Units Sold")
plt.grid(True)
plt.savefig('facebook_vs_sales.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 4))
sns.regplot(data=sales_df, x='Ad_Spend_Instagram', y='Sales', scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.title("Sales vs. Instagram Ad Spend")
plt.xlabel("Instagram Ad Spend ($)")
plt.ylabel("Units Sold")
plt.grid(True)
plt.savefig('instagram_vs_sales.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 4))
sns.regplot(data=sales_df, x='Discount_Rate', y='Sales', scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.title("Sales vs. Discount Rate")
plt.xlabel("Discount Rate (%)")
plt.ylabel("Units Sold")
plt.grid(True)
plt.savefig('discount_rate_vs_sales.png', bbox_inches='tight')
plt.show()



print("=== QUESTION 3 & 4: SIMPLE LINEAR REGRESSION ===")
X_simple = sm.add_constant(sales_df['Ad_Spend_Facebook'])
y = sales_df['Sales']
model_simple = sm.OLS(y, X_simple).fit()
print(f"Intercept: {model_simple.params.iloc[0]:.2f}")
print(f"Facebook Coefficient (Slope): {model_simple.params.iloc[1]:.2f}")
print(f"R-squared Value: {model_simple.rsquared:.4f}\n")


print("=== QUESTION 5: MULTIPLE LINEAR REGRESSION ===")
X_multi = sm.add_constant(sales_df[['Ad_Spend_Facebook', 'Ad_Spend_Instagram', 'Discount_Rate']])
model_multi = sm.OLS(y, X_multi).fit()
print(f"Intercept: {model_multi.params.iloc[0]:.2f}")
print(f"Facebook Coef: {model_multi.params.iloc[1]:.2f}")
print(f"Instagram Coef: {model_multi.params.iloc[2]:.2f}")
print(f"Discount Rate Coef: {model_multi.params.iloc[3]:.2f}")
print(f"R-squared: {model_multi.rsquared:.4f}")
print(f"adjusted R-squared: {model_multi.rsquared_adj:.4f}\n")


print("=== QUESTION 6: FORECASTING (JANUARY 2025) ===")
last_3_months_avg = forecast_df['Sales'].iloc[-4:-1].mean()
sales_series = forecast_df['Sales'].dropna().tolist()
smoothed_values = [sales_series[0]]
alpha = 0.80
for i in range(1, len(sales_series)):
    next_forecast = (alpha * sales_series[i]) + ((1 - alpha) * smoothed_values[-1])
    smoothed_values.append(next_forecast)

print(f"3-Month Moving Average Forecast: {last_3_months_avg:.2f}")
print(f"Exponential Smoothing Forecast: {smoothed_values[-1]:.2f}\n")


print("=== QUESTION 7: MACHINE LEARNING (RMSE) ===")
X_ml = sales_df[['Ad_Spend_Facebook', 'Ad_Spend_Instagram', 'Discount_Rate']]
y_ml = sales_df['Sales']
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lr_rmse = np.sqrt(-cross_val_score(LinearRegression(), X_ml, y_ml, cv=kf, scoring='neg_mean_squared_error')).mean()
dt_rmse = np.sqrt(-cross_val_score(DecisionTreeRegressor(random_state=42), X_ml, y_ml, cv=kf, scoring='neg_mean_squared_error')).mean()
print(f"Linear Regression RMSE: {lr_rmse:.2f}")
print(f"Decision Tree RMSE: {dt_rmse:.2f}\n")


print("=== QUESTION 8: MONTE CARLO SIMULATION ===")
np.random.seed(42)
simulated_data = np.random.uniform(2299, 7702, 1000)
print(f"Simulated Mean: {np.mean(simulated_data):.2f}")
print(f"Simulated Median: {np.median(simulated_data):.2f}")
print(f"Simulated Standard Deviation: {np.std(simulated_data):.2f}\n")


print("=== QUESTION 9: LINEAR PROGRAMMING OPTIMIZATION ===")
result = minimize(lambda x: -(5 * x[0] + 3 * x[1]), [2000, 1000], method='SLSQP',
                  bounds=((2000, None), (1000, 7000)),
                  constraints=({'type': 'ineq', 'fun': lambda x: 10000 - (x[0] + x[1])},
                               {'type': 'ineq', 'fun': lambda x: x[0] - (0.5 * (x[0] + x[1]))}))

if result.success:
    print(f"Optimal Facebook Budget: ${result.x[0]:.2f}")
    print(f"Optimal Instagram Budget: ${result.x[1]:.2f}")
    print(f"Maximum Expected Sales Units: {-result.fun:.2f}")
