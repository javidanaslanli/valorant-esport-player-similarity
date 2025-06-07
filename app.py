import warnings
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import gradio as gr
from fuzzywuzzy import process

warnings.filterwarnings('ignore')

# --- Load and preprocess data ---
df = pd.read_csv('players_stats.csv')

df['Average Combat Score'] = df['Average Combat Score'].fillna(method='bfill')

def clean_percentage_column(df, col_name):
    df[col_name] = df[col_name].str.replace('%', '', regex=False)
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce') / 100

percentage_columns = ['Kill, Assist, Trade, Survive %', 'Headshot %', 'Clutch Success %']
for col in percentage_columns:
    clean_percentage_column(df, col)

df[['clutches_won', 'clutches_played']] = df['Clutches (won/played)'].str.split('/', expand=True)
df['clutches_won'] = pd.to_numeric(df['clutches_won'], errors='coerce')
df['clutches_played'] = pd.to_numeric(df['clutches_played'], errors='coerce')
df['clutch_success_rate'] = (df['clutches_won'] / df['clutches_played']).round(2)
df.drop(['clutches_won', 'clutches_played', 'Clutches (won/played)'], axis=1, inplace=True)

def rnwpm(df, column_name, other_column='Player'):
    player_mean = df.groupby(other_column)[column_name].mean()
    df['player_mean'] = df[other_column].map(player_mean)
    overall_mean = df[column_name].mean()
    df[column_name] = (df[column_name].fillna(df['player_mean']).fillna(overall_mean)).round(2)
    df.drop(columns=['player_mean'], inplace=True)

cols = ['First Kills Per Round', 'Rating', 'Average Damage Per Round', 
        'First Deaths Per Round', 'Kill, Assist, Trade, Survive %', 
        'Headshot %', 'Clutch Success %', 'clutch_success_rate']
for col in cols:
    rnwpm(df, column_name=col)

features = df.drop(['Tournament','Stage','Match Type','Teams','Agents'], axis=1)
features_meaned = features.groupby('Player').mean(numeric_only=True).reset_index()
player_features = features_meaned.drop('Player', axis=1)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(player_features)
features_df = pd.DataFrame(features_scaled, columns=player_features.columns)
features_df['Player'] = features_meaned['Player'].values

similarity_matrix = cosine_similarity(features_df.drop('Player', axis=1))
similarity_df = pd.DataFrame(similarity_matrix, index=features_df['Player'], columns=features_df['Player'])


def get_similar_players(player_name, top_n=5):
    all_players = similarity_df.index.tolist()
    best_match, score = process.extractOne(player_name, all_players)
    if score < 60:
        return f"Player '{player_name}' not found. Did you mean '{best_match}'?"
    sim_scores = similarity_df[best_match].drop(best_match)
    return f"Top {top_n} players similar to '{best_match}':\n" + '\n'.join(sim_scores.sort_values(ascending=False).head(top_n).index)


def recommend(player_input, top_n):
    return get_similar_players(player_input, top_n)

gradio_app = gr.Interface(fn=recommend,
                    inputs=[
                        gr.Textbox(label="Enter Player Name"),
                        gr.Slider(1, 10, value=5, step=1, label="Top N Similar Players")
                    ],
                    outputs="text",
                    title="Valorant Player Similarity Recommender",
                    description="Enter a player name to find similar players based on VCT 2025 data.")


if __name__ == "__main__":
    gradio_app.launch()