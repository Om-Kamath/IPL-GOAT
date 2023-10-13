import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and Normalize Data
batter_data = pd.read_csv('datasets/batter_data.csv')
bowler_data = pd.read_csv('datasets/bowler_data.csv')

batter_attributes = ['strike_rate', 'sixes', 'fours']
bowler_attributes = ['total_wickets', 'bowled', 'caught', 'economy_rate']

scaler_batter = MinMaxScaler()
scaler_bowler = MinMaxScaler()

batter_data_normalized = scaler_batter.fit_transform(batter_data[batter_attributes])
bowler_data_normalized = scaler_bowler.fit_transform(bowler_data[bowler_attributes])

# KNN Models
knn_batter = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(batter_data_normalized)
knn_bowler = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(bowler_data_normalized)

# Streamlit UI
st.title("Cricket GOAT Predictor")
st.sidebar.header("Attributes Weightage")
player_type = st.sidebar.selectbox("Select Player Type", ["Batter", "Bowler"])

# Define and label sliders based on player type
if player_type == 'Batter':
    weights = {
        'Striker': st.sidebar.slider("Striker Importance", 0.0, 1.0, 0.5),
        'Hitman': st.sidebar.slider("Hitman Importance", 0.0, 1.0, 0.5),
        'Smasher': st.sidebar.slider("Smasher Importance", 0.0, 1.0, 0.5),
    }
    knn_model = knn_batter
    data = batter_data
    normalized_data = batter_data_normalized
elif player_type == 'Bowler':
    weights = {
        'Wicket-Taker': st.sidebar.slider("Wicket-Taker Importance", 0.0, 1.0, 0.5),
        'Uprooter': st.sidebar.slider("Uprooter Importance", 0.0, 1.0, 0.5),
        'Fielder-Dependent': st.sidebar.slider("Fielder-Dependent Importance", 0.0, 1.0, 0.5),
        'Economizer': st.sidebar.slider("Economizer Importance", 0.0, 1.0, 0.5),
    }
    knn_model = knn_bowler
    data = bowler_data
    normalized_data = bowler_data_normalized

# Button to get recommendations
if st.sidebar.button("Get Recommendations"):
    user_point = np.array([weights[feature] for feature in weights.keys()]).reshape(1, -1)
    _, indices = knn_model.kneighbors(user_point)
    recommended_players = data.iloc[indices[0]]
    st.dataframe(recommended_players)

    attributes = batter_attributes if player_type == 'Batter' else bowler_attributes
    # Visualization
    fig = px.scatter_3d(data, x=attributes[0], y=attributes[1], z=attributes[2])
    fig.add_scatter3d(x=user_point[:, 0], y=user_point[:, 1], z=user_point[:, 2], mode='markers', marker=dict(color='red', size=10), name='User Point')
    fig.add_scatter3d(x=normalized_data[indices[0], 0], y=normalized_data[indices[0], 1], z=normalized_data[indices[0], 2], mode='markers', marker=dict(color='blue', size=10), name='Recommended Players')

    fig.update_layout(title="3D K-Nearest Neighbors Visualization", scene=dict(xaxis_title=list(weights.keys())[0], yaxis_title=list(weights.keys())[1], zaxis_title=list(weights.keys())[2]))
    st.plotly_chart(fig)
else:
    st.info("Select the attributes' importance using the sliders and click 'Get Recommendations'")
