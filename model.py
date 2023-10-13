from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Load the data
batter_data = pd.read_csv('datasets/batter_data.csv')
bowler_data = pd.read_csv('datasets/bowler_data.csv')

# Define attributes for batter and bowler
batter_attributes = ['strike_rate', 'sixes', 'fours']
bowler_attributes = ['total_wickets', 'bowled', 'caught', 'economy_rate']

# Normalize the data
scaler_batter = MinMaxScaler()
scaler_bowler = MinMaxScaler()

batter_data_normalized = scaler_batter.fit_transform(batter_data[batter_attributes])
bowler_data_normalized = scaler_bowler.fit_transform(bowler_data[bowler_attributes])

# Initialize the KNN models
knn_batter = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn_bowler = NearestNeighbors(n_neighbors=5, metric='euclidean')

# Fit the models to the normalized data
knn_batter.fit(batter_data_normalized)
knn_bowler.fit(bowler_data_normalized)

# Validate the models by providing a query point
# For simplicity, using mean values of normalized data as query point
# query_point_batter = np.mean(batter_data_normalized, axis=0).reshape(1, -1)
# query_point_bowler = np.mean(bowler_data_normalized, axis=0).reshape(1, -1)

# # Find the k-nearest neighbors for the query points
# _, indices_batter = knn_batter.kneighbors(query_point_batter)
# _, indices_bowler = knn_bowler.kneighbors(query_point_bowler)

# # Retrieve the original player data for the recommended indices
# recommended_batters = batter_data.iloc[indices_batter[0]]
# recommended_bowlers = bowler_data.iloc[indices_bowler[0]]

# print(recommended_batters, recommended_bowlers)

# Assume the user gives the following weights in [0,1] range
user_preferences_batter = {
    'Striker': 0.1,  # Example weight for 'strike_rate'
    'Hitman': 0.2,   # Example weight for 'sixes'
    'Smasher': 0.7   # Example weight for 'fours'
}

# Convert user preferences to array
user_point_batter = np.array(list(user_preferences_batter.values())).reshape(1, -1)

# Get nearest neighbors based on user preferences
_, indices_batter = knn_batter.kneighbors(user_point_batter)

# Retrieve and print recommended players
recommended_batters = batter_data.iloc[indices_batter[0]]
print(recommended_batters)
