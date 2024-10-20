import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Step 1: Load the healthcare data
data = pd.read_csv(r'C:\Users\Dinesh\OneDrive\Desktop\maps\data\health_data.csv')  # Replace with your CSV file

# Print the column names to verify if 'Target (Healthcare Access)' exists
print(data.columns)

# Step 2: Rank the healthcare data into 5 bins (0 to 4)
if 'Target (Healthcare Access)' in data.columns:
    # Create 5 bins ranging from 0 to 4
    data['rank'] = pd.cut(data['Target (Healthcare Access)'], bins=5, labels=False)  # Bins from 0 to 4
    data['rank'] = data['rank'].astype(int)  # Ensure rank is an integer
else:
    print("Column 'Target (Healthcare Access)' does not exist in the DataFrame.")
    exit()

# Step 3: Create a GeoDataFrame from Latitude and Longitude
geometry = [Point(xy) for xy in zip(data['Longitude'], data['Latitude'])]
gdf = gpd.GeoDataFrame(data, geometry=geometry)

# Step 4: Plot the outline map for each state
states = gdf['State Names'].unique()  # Get unique state names

for state in states:
    state_gdf = gdf[gdf['State Names'] == state]

    if not state_gdf.empty:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the points for the specific state
        state_gdf.plot(column='rank', ax=ax, legend=True, cmap='coolwarm', markersize=100, edgecolor='black')

        # Add annotations for each point and print to console
        for x, y, district, rank in zip(state_gdf.geometry.x, state_gdf.geometry.y, state_gdf['District'], state_gdf['rank']):
            ax.text(x, y, f'{district}\nRank: {rank}', fontsize=8, ha='center', va='center')
            print(f'District: {district}, Rank: {rank}')  # Print actual district names and ranks

        # Customize the title and axes
        plt.title(f'Healthcare Access Ranking in {state}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)  # Optional: Add a grid for better visibility
        plt.show()
    else:
        print(f"No data available for {state}.")
