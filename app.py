import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load Data
batter_data = pd.read_csv('datasets/batter_agg.csv')
bowler_data = pd.read_csv('datasets/bowler_agg.csv')
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
     [data-testid="stForm"] {border: 0px; background: #262730; border-radius: 10px} 
     [data-baseweb="select"] {margin-top: -30px;}
    </style>
    """,
    unsafe_allow_html=True
)


# Define and label sliders based on player type
col1, col2 = st.columns([4,8], gap="medium")
with col1:
    st.subheader("Player Type ")
    player_type = st.selectbox("",options =("Batter", "Bowler"))
    with st.form("My Form"):
        st.subheader("Attributes Weightage", )
        if player_type == "Batter":
            batter_weights = {
                'Blazing Batsman': st.slider("Blazing Batsman Importance", 0.0, 1.0, 0.5, help="Values half-centuries and centuries made by the player"),
                'Thunderous Titan': st.slider("Thunderous Titan Importance", 0.0, 1.0, 0.5, help="Rewards boundaries scored"), 
                'Consistent Conqueror': st.slider("Consistent Conqueror Importance", 0.0, 1.0, 0.5, help="Encourages a higher strike-rate and average score per innings while penalizing losing one's wicket"),
                'Accolades': st.checkbox("Accolades Boost", help="Rewards awards won by the player like Player of the Match and Orange Cap")
            }
            
            # Define the metrics and their respective weights
            

            # Allow users to select a range for innings
            innings = st.slider("Number of Innings", 1, 200, (50,200), help="The minimum and the maximum no. of innings played by the player")  # (min_value, max_value, (default_min_range, default_max_range))

            num_recommendations = st.slider("Number of Recommendations", 1, 10, 5, help="No. of recommendations you want the application to generate")

            # accolades_boost = st.checkbox("Accolades Boost", help="Boosts awards won by the player like Player of the Match and Orange Cap")
            
            if not batter_weights['Accolades']:
                batter_metric_weights = {
                    'Blazing Batsman': {'total_100s': 0.7, 'total_50s': 0.4},
                    'Thunderous Titan': {'sixes': 0.6, 'fours': 0.4},
                    'Consistent Conqueror': {'average': 0.2, 'avg_strike_rate': 0.5, 'outs': -0.3},
                    'Accolades': {'orange_caps' : 0.1, "motm_count" : 0.1}
                }
            else:
                batter_metric_weights = {
                    'Blazing Batsman': {'total_100s': 0.7, 'total_50s': 0.4},
                    'Thunderous Titan': {'sixes': 0.6, 'fours': 0.4},
                    'Consistent Conqueror': {'average': 0.2, 'avg_strike_rate': 0.5, 'outs': -0.3},
                    'Accolades': {'orange_caps' : 0.7, "motm_count" : 0.5}
                }


            # Filter data based on user input
            filtered_batter_data = batter_data[
                (batter_data['innings'] >= innings[0]) &
                (batter_data['innings'] <= innings[1])
            ]
            # Compute the score for each player by summing the weighted attributes
            for category in batter_metric_weights.keys():
                filtered_batter_data[category] = filtered_batter_data.apply(
                    lambda row: np.dot(row[batter_metric_weights[category].keys()], list(batter_metric_weights[category].values())), axis=1)
                filtered_batter_data[category] = filtered_batter_data[category] * batter_weights[category]

            filtered_batter_data['score'] = filtered_batter_data[list(batter_weights.keys())].sum(axis=1)

        elif player_type == "Bowler":
            bowler_weights = {
                'Mystical Magician': st.slider("Mystical Magician Importance", 0.0, 1.0, 0.5, help="Encourages lower economy and rewards wickets taken via clean bowled"),
                'Tenacious Terminator': st.slider("Tenacious Terminator Importance", 0.0, 1.0, 0.5, help="Rewards higher wicket count and penalizes total runs given up"),
                'Consistent Challenger': st.slider("Consistent Challenger Importance", 0.0, 1.0, 0.5, help="Penalizes wides and no-balls and encourages discipline"),
                'Accolades': st.checkbox("Accolades Boost", help="Rewards awards won by the player like Player of the Match and Purple Cap")
            }

            # Allow users to select a range for innings
            innings = st.slider("Number of Innings", 1, 200,(50,200))  # (min_value, max_value, (default_min_range, default_max_range))

            num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)
            
            
            # Define the metrics and their respective weights
            if not bowler_weights['Accolades']:
                bowler_metric_weights = {
                    'Mystical Magician': {'economy': -0.3, 'bowled': 0.4},
                    'Tenacious Terminator': {'total_wickets': 0.3, 'total_runs_conceded': -0.1, "caught": 0.3, "stumped": 0.1, "lbw": 0.2},
                    'Consistent Challenger': {'total_wides': -0.3, 'total_no_balls': -0.3, 'total_legal_deliveries': 0.4},
                    'Accolades': {'purple_caps' : 0.1, "motm_count" : 0.1}
                }
            else:
                bowler_metric_weights = {
                    'Mystical Magician': {'economy': -0.3, 'bowled': 0.4},
                    'Tenacious Terminator': {'total_wickets': 0.3, 'total_runs_conceded': -0.1, "caught": 0.3, "stumped": 0.1, "lbw": 0.2},
                    'Consistent Challenger': {'total_wides': -0.3, 'total_no_balls': -0.3, 'total_legal_deliveries': 0.4},
                    'Accolades': {'purple_caps' : 0.7, "motm_count" : 0.5}
                }


            # Filter data based on user input
            filtered_bowler_data = bowler_data[
                (bowler_data['matches'] >= innings[0]) &
                (bowler_data['matches'] <= innings[1])
            ]
            # Compute the score for each player by summing the weighted attributes
            for category in bowler_metric_weights.keys():
                filtered_bowler_data[category] = filtered_bowler_data.apply(
                    lambda row: np.dot(row[bowler_metric_weights[category].keys()], list(bowler_metric_weights[category].values())), axis=1)
                filtered_bowler_data[category] = filtered_bowler_data[category] * bowler_weights[category]

            filtered_bowler_data['score'] = filtered_bowler_data[list(bowler_weights.keys())].sum(axis=1)
            

        submit = st.form_submit_button("Get Recommendations")
            

# Button to get recommendations
with col2:
    if submit:
        if player_type == "Batter": 
            # batter,innings,total_runs,total_balls,outs,total_50s,total_100s,avg_strike_rate,motm_count,orange_caps,average,fours,sixes
            recommended_players = filtered_batter_data.nlargest(num_recommendations, 'score')
            st.dataframe(recommended_players[["batter", "Blazing Batsman", "Thunderous Titan", "Consistent Conqueror", "score"]], hide_index=True)
           
            # Visualization
            fig = px.scatter_3d(filtered_batter_data, x='Blazing Batsman', y='Thunderous Titan', z='Consistent Conqueror',
                                color='score', hover_name='batter',
                                hover_data={"Blazing Batsman": False, "Thunderous Titan": False, "Consistent Conqueror": False,'score': False, 'total_runs': True, 'avg_strike_rate': True, 'fours': True, 'sixes': True})

            # Highlight recommended players
            for i, player in recommended_players.iterrows():
                fig.add_trace(go.Scatter3d(x=[player['Blazing Batsman']], y=[player['Thunderous Titan']], z=[player['Consistent Conqueror']],
                                            mode='markers', marker=dict(color='red', size=10), showlegend=False,
                                            hovertemplate=f"<b>{player['batter']}</b><br>Total Runs: {player['total_runs']}<br>Strike Rate: {player['avg_strike_rate']:.2f}<br>Fours: {player['fours']}<br>Sixes: {player['sixes']}<extra></extra>"))

            fig.update_layout(title="3D Visualization", scene=dict(xaxis_title='Blazing Batsman', yaxis_title='Thunderous Titan', zaxis_title='Consistent Conqueror'))
            st.plotly_chart(fig,__loader__ = "hide")
        else:

            # bowler,matches,total_runs_conceded,total_balls_bowled,total_illegal_deliveries,total_wides,total_no_balls,total_legal_deliveries,total_wickets,bowled,lbw,caught,stumped,others,total_overs,economy,purple_caps,motm_count
            recommended_players = filtered_bowler_data.nlargest(num_recommendations, 'score')
            st.dataframe(recommended_players[["bowler", "Mystical Magician", "Tenacious Terminator", "Consistent Challenger", "score"]], hide_index=True)

            # Visualization
            fig = px.scatter_3d(filtered_bowler_data, x='Mystical Magician', y='Tenacious Terminator', z='Consistent Challenger',
                                color='score', hover_name='bowler',
                                hover_data={'Mystical Magician': False,'Tenacious Terminator': False,'Consistent Challenger': False,"score": False,'total_wickets': True, 'matches': True, 'economy': True})

            # Highlight recommended players
            for i, player in recommended_players.iterrows():
                fig.add_trace(go.Scatter3d(x=[player['Mystical Magician']], y=[player['Tenacious Terminator']], z=[player['Consistent Challenger']],
                                            mode='markers', marker=dict(color='red', size=10), showlegend=False,
                                            hovertemplate=f"<b>{player['bowler']}</b><br>Economy: {player['economy']:.2f}<br>Total Wickets: {player['total_wickets']:.2f}<br>Total Matches: {player['matches']:.2f}<extra></extra>"))

            fig.update_layout(title="3D Visualization", scene=dict(xaxis_title='Mystical Magician', yaxis_title='Tenacious Terminator', zaxis_title='Consistent Challenger'))
            st.plotly_chart(fig)

        

    # Only display the info message when the button hasn't been clicked
    if not submit:
        st.info("Select the attributes' importance using the sliders and click 'Get Recommendations'")
