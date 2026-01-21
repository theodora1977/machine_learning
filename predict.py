import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("gig_intl_prices_cleaned.csv")

#help user predict Zone 8 price based on KG and other zones
X = data[["KG", "Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6", "Zone 7"]]
y = data["Zone 8"]
#allow user to input values for prediction
def predict_zone_8(kg, zone1, zone2, zone3, zone4, zone5, zone6, zone7):
    model = LinearRegression()
    model.fit(X, y)
    input_data = np.array([[kg, zone1, zone2, zone3, zone4, zone5, zone6, zone7]])
    predicted_price = model.predict(input_data)
    return predicted_price[0]
# Example usage
kg = float(input("Enter weight in KG: "))
zone1 = float(input("Enter price in Zone 1: "))
zone2 = float(input("Enter price in Zone 2: "))
zone3 = float(input("Enter price in Zone 3: "))
zone4 = float(input("Enter price in Zone 4: "))
zone5 = float(input("Enter price in Zone 5: "))
zone6 = float(input("Enter price in Zone 6: "))
zone7 = float(input("Enter price in Zone 7: "))

predicted_price = predict_zone_8(kg, zone1, zone2, zone3, zone4, zone5, zone6, zone7)
print(f"Predicted price for Zone 8 is: {predicted_price}")
#show user the range of price that can be inputed for getting zone 8 price
print("Price ranges for other zones based on dataset:")
print("Zone 1:", data["Zone 1"].min(), "-", data["Zone 1"].max())
print("Zone 2:", data["Zone 2"].min(), "-", data["Zone 2"].max())
print("Zone 3:", data["Zone 3"].min(), "-", data["Zone 3"].max())
print("Zone 4:", data["Zone 4"].min(), "-", data["Zone 4"].max())
print("Zone 5:", data["Zone 5"].min(), "-", data["Zone 5"].max())
print("Zone 6:", data["Zone 6"].min(), "-", data["Zone 6"].max())
print("Zone 7:", data["Zone 7"].min(), "-", data["Zone 7"].max())
#else raise error if input is out of range
if not (data["Zone 1"].min() <= zone1 <= data["Zone 1"].max()):
    raise ValueError("Zone 1 price out of range")
if not (data["Zone 2"].min() <= zone2 <= data["Zone 2"].max()):
    raise ValueError("Zone 2 price out of range")
if not (data["Zone 3"].min() <= zone3 <= data["Zone 3"].max()):
    raise ValueError("Zone 3 price out of range")
if not (data["Zone 4"].min() <= zone4 <= data["Zone 4"].max()):
    raise ValueError("Zone 4 price out of range")
if not (data["Zone 5"].min() <= zone5 <= data["Zone 5"].max()):
    raise ValueError("Zone 5 price out of range")
if not (data["Zone 6"].min() <= zone6 <= data["Zone 6"].max()):
    raise ValueError("Zone 6 price out of range")
if not (data["Zone 7"].min() <= zone7 <= data["Zone 7"].max()):
    raise ValueError("Zone 7 price out of range")
if kg <= 0:
    raise ValueError("KG must be positive")
print("All inputs are within valid ranges.")
