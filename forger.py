import random
import pandas as pd

# Read the CSV file
charge_df = pd.read_csv('/home/neeko/Scripts/serialParser/data/charge.csv')
discharge_df = pd.read_csv('/home/neeko/Scripts/serialParser/data/discharge.csv')

# Apply random shift to each value
charge_df = charge_df.map(lambda x: x + random.uniform(-0.05, 0.05))
discharge_df = discharge_df.map(lambda x: x + random.uniform(-0.05, 0.05))

# Write the modified DataFrame to a new CSV file
charge_df.to_csv('/home/neeko/Scripts/serialParser/data/forged_charge.csv', index=False)
discharge_df.to_csv('/home/neeko/Scripts/serialParser/data/forged_discharge.csv', index=False)