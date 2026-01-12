import pandas as pd
import os

class ShippingCalculator:
    def __init__(self, prices_csv='gig_intl_prices_cleaned.csv', zones_csv='gig_intl_zones_cleaned.csv'):
        """
        Initialize the ShippingCalculator with paths to the CSV files.
        """
        self.prices_csv = prices_csv
        self.zones_csv = zones_csv
        self.price_df = None
        self.zone_df = None
        self._load_data()

    def _load_data(self):
        """
        Loads the CSV data into pandas DataFrames.
        """
        if not os.path.exists(self.prices_csv) or not os.path.exists(self.zones_csv):
            print(f"Error: CSV files not found. Please ensure '{self.prices_csv}' and '{self.zones_csv}' exist.")
            return

        self.price_df = pd.read_csv(self.prices_csv)
        self.zone_df = pd.read_csv(self.zones_csv)
        
        # Pre-process country names for case-insensitive matching
        # Create a lower-case column for lookups
        self.zone_df['Country_Lower'] = self.zone_df['Country'].astype(str).str.strip().str.lower()

    def get_countries(self):
        """
        Returns a sorted list of available countries.
        """
        if self.zone_df is None:
            return []
        return sorted(self.zone_df['Country'].tolist())

    def get_shipping_price(self, country, kg):
        """
        Calculates the shipping price based on the country and weight (KG).
        Handles case-insensitive country names.
        """
        if self.price_df is None or self.zone_df is None:
            # Try reloading if data is missing
            self._load_data()
            if self.price_df is None or self.zone_df is None:
                return "Error: Data not loaded."

        # Normalize input country
        country_norm = str(country).strip().lower()

        # Find zone
        zone_row = self.zone_df[self.zone_df['Country_Lower'] == country_norm]

        if zone_row.empty:
            return f"Error: Country '{country}' not found in zone data."

        zone_num = zone_row.iloc[0]['Zone']
        zone_col = f"Zone {zone_num}"

        # Find price
        # Ensure KG is float
        try:
            kg = float(kg)
        except ValueError:
             return f"Error: Invalid weight '{kg}'."

        # Get all available weights
        available_weights = sorted(self.price_df['KG'].unique())
        
        # Find the appropriate weight bracket
        target_weight = None
        for w in available_weights:
            if w >= kg:
                target_weight = w
                break
        
        if target_weight is None:
             return f"Error: Weight {kg} KG exceeds the maximum available weight of {available_weights[-1]} KG."

        price_row = self.price_df[self.price_df['KG'] == target_weight]
        
        # Get the price from the specific zone column
        try:
            price = price_row.iloc[0][zone_col]
            # Return both the price and the weight used for calculation if it differs
            if target_weight != kg:
                return f"{price} (Based on next weight tier: {target_weight} KG)"
            return price
        except KeyError:
            return f"Error: Zone column '{zone_col}' not found in price data."

if __name__ == "__main__":
    # Example usage when running the script directly
    calculator = ShippingCalculator()
    
    print("Available countries (first 5):", calculator.get_countries()[:5])
    
    print(f"Shipping to Ghana (2.0 KG): {calculator.get_shipping_price('Ghana', 2.0)}")
    print(f"Shipping to ghana (2.0 KG): {calculator.get_shipping_price('ghana', 2.0)}")
    print(f"Shipping to GHANA (2.0 KG): {calculator.get_shipping_price('GHANA', 2.0)}")
    print(f"Shipping to Afghanistan (1.0 KG): {calculator.get_shipping_price('Afghanistan', 1.0)}")
