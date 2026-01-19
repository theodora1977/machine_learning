import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("gig_intl_prices_cleaned.csv")

# Single variable: KG -> Zone1
X = data[["KG"]]
y = data["Zone 1"]

model = LinearRegression()
model.fit(X, y)

predicted = model.predict(X)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

# Plot
plt.scatter(X, y)
plt.plot(X, predicted)
plt.xlabel("KG")
plt.ylabel("Zone 1 Price")
plt.title("Single Variable Linear Regression (Zone 1)")
plt.show()
