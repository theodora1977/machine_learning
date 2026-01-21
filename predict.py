import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("gig_intl_prices_cleaned.csv")

#help user predict Zone 8 price based on KG and other zones
X = data[["KG", "Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6", "Zone 7"]]
y = data["Zone 8"]

# Train the model once
model = LinearRegression()
model.fit(X, y)

#allow user to input values for prediction
def predict_zone_8(kg, zone1, zone2, zone3, zone4, zone5, zone6, zone7):
    input_data = np.array([[kg, zone1, zone2, zone3, zone4, zone5, zone6, zone7]])
    predicted_price = model.predict(input_data)
    return predicted_price[0]

if __name__ == "__main__":
    try:
        kg = float(input("Enter KG: "))
        zone1 = float(input("Enter price in Zone 1: "))
        zone2 = float(input("Enter price in Zone 2: "))
        zone3 = float(input("Enter price in Zone 3: "))
        zone4 = float(input("Enter price in Zone 4: "))
        zone5 = float(input("Enter price in Zone 5: "))
        zone6 = float(input("Enter price in Zone 6: "))
        zone7 = float(input("Enter price in Zone 7: "))

        # Validate inputs
        if kg <= 0:
            raise ValueError("KG must be positive")
        
        zones = [zone1, zone2, zone3, zone4, zone5, zone6, zone7]
        for i, val in enumerate(zones, 1):
            col = f"Zone {i}"
            if not (data[col].min() <= val <= data[col].max()):
                raise ValueError(f"{col} price out of range ({data[col].min()} - {data[col].max()})")

        predicted_price = predict_zone_8(kg, zone1, zone2, zone3, zone4, zone5, zone6, zone7)
        print(f"Predicted price for Zone 8 is: {predicted_price}")

    except ValueError as e:
        print(f"Error: {e}")
