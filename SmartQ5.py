#%%[markdown]
## Is there any significant difference in average order profit between regions?

#%%
import pandas as pd



#%%
# Reading dataset
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')
#%%
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())

print(df['Order Region'].value_counts())
# %%

# Check unique values in 'Order Status' to understand how completed orders are labeled
print(df['Order Status'].value_counts())

# Filter the dataset for completed orders
df = df[df['Order Status'] == 'COMPLETE']

print(df.head())


# %%
df[['Latitude', 'Longitude']].head()


# %%
df_stores_location = df[['Latitude', 'Longitude']].drop_duplicates().dropna()

# %%
import folium
from folium.plugins import MarkerCluster
from IPython.display import display


# Filter relevant columns for store locations
store_locations_df = df[['Customer City', 'Latitude', 'Longitude']].drop_duplicates()

# Create a base map
map_center = [store_locations_df['Latitude'].mean(), store_locations_df['Longitude'].mean()]
store_map = folium.Map(location=map_center, zoom_start=5)

# Add markers for each store location
marker_cluster = MarkerCluster().add_to(store_map)

for index, row in store_locations_df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Customer City'],
        icon=folium.Icon(color='blue')
    ).add_to(marker_cluster)

# Display the map
display(store_map)

# %%
#store_map.save("store_locations_map.html")
# %%[markdown]
#OBSERVATION: Most of the stores are located in the United States

#%%
import folium
from folium.plugins import MarkerCluster

# Filter relevant columns for delivery locations
delivery_locations_df = df[['Order City', 'Order Country', 'Latitude', 'Longitude']].drop_duplicates().dropna().reset_index(drop=True)
delivery_locations_df.head()

#%%
# Create a dark-themed base map
map_center = [delivery_locations_df['Latitude'].mean(), delivery_locations_df['Longitude'].mean()]
delivery_map = folium.Map(location=map_center, zoom_start=5, tiles='CartoDB dark_matter')

# Add markers for each delivery location
marker_cluster = MarkerCluster().add_to(delivery_map)

for index, row in delivery_locations_df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Order City']}, {row['Order Country']}",
        icon=folium.Icon(color='blue')
    ).add_to(marker_cluster)

# Save the map to an HTML file
#delivery_map.save("delivery_locations_map.html")
display(delivery_map)

import folium

# Calculate average profit per city and the overall average profit
average_profit_per_city = df.groupby('Order City')['Order Profit Per Order'].mean()
overall_avg_profit = df['Order Profit Per Order'].mean()

# Merge the average profit data with the original DataFrame to get coordinates
city_profit_with_coords = df[['Order City', 'Latitude', 'Longitude']].drop_duplicates().merge(average_profit_per_city, on='Order City')

# Create a base map
map = folium.Map(location=[0, 0], zoom_start=2)

# Add markers to the map
for idx, row in city_profit_with_coords.iterrows():
    color = 'green' if row['Order Profit Per Order'] >= overall_avg_profit else 'red'
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        tooltip=f"{row['Order City']}: {'Above' if color == 'green' else 'Below'} Average"
    ).add_to(map)

# Add legends
legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:1000; font-size:14px;">
    &nbsp; <i class="fa fa-circle" style="color:green"></i> High Profit &nbsp; 
    &nbsp; <i class="fa fa-circle" style="color:red"></i> Low Profit &nbsp;
    </div>
    """
map.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
#map.save('map.html')
overall_avg_profit
# %%
# Filter the dataset for completed orders
df_completed = df[df['Order Status'] == 'COMPLETE']

# Filter relevant columns for the analysis
profit_region_df = df_completed[['Order Profit Per Order', 'Order Region']]

# Perform a groupby operation to calculate the average profit for each region
average_profit_by_region = profit_region_df.groupby('Order Region')['Order Profit Per Order'].mean().reset_index()

# Display the average profit for each region
print(average_profit_by_region)

# Example of using ANOVA
from scipy.stats import f_oneway

# Extract profit data for each region
region_profits = [profit_region_df[profit_region_df['Order Region'] == region]['Order Profit Per Order'] for region in average_profit_by_region['Order Region']]

# Perform ANOVA
f_stat, p_value = f_oneway(*region_profits)

# Display the results
print(f"F-Statistic: {f_stat}")
print(f"P-Value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference in average order profit between regions.")
else:
    print("There is no significant difference in average order profit between regions.")
#%%[markdown]
## OBSERVATION: Even though each region shows different average profits from orders, these differences are not significant enough to say for sure that they are due to the region itself. It could just be random chance that some regions have higher or lower average profits. So, we can't confidently say that the region has a real effect on how much profit is made from orders.
#%%
std_dev_profit_per_region = df.groupby('Order Region')['Order Profit Per Order'].std()
std_dev_profit_per_region.sort_values(ascending=False)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Filter the dataset for completed orders
df_completed = df[df['Order Status'] == 'COMPLETE']

profit_region_df = df_completed[['Order Profit Per Order', 'Order Region']]
overall_avg_profit = df_completed['Order Profit Per Order'].mean()

average_profit_by_region_sorted = average_profit_by_region.sort_values(by='Order Profit Per Order', ascending=False)
overall_avg_profit = df_completed['Order Profit Per Order'].mean()

import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
sns.barplot(x='Order Region', y='Order Profit Per Order', data=average_profit_by_region_sorted, palette="viridis")
plt.axhline(y=overall_avg_profit, color='red', linestyle='--', label='Overall Average')
plt.title('Average Order Profit Across Regions')
plt.xlabel('Order Region')
plt.ylabel('Average Order Profit Per Order')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.legend()
plt.ylim(0, 40)

plt.show()



# %%
