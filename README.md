# Valorant Player Similarity Recommender

This project is a player similarity recommender system for Valorant players based on Valorant Champion Tour (VCT) 2021-2023 data.  
It uses player statistics and machine learning techniques (cosine similarity on standardized features) to find players with similar playstyles.

The app is built with **Python**, **Pandas**, **Scikit-learn**, and **Gradio** for the interactive UI.

---

## Features

- Clean and preprocess player stats data
- Calculate player similarity using cosine similarity
- Fuzzy matching for player names to handle typos
- Interactive Gradio interface to input player name and get similar players

---

## Files in this repo

- `app.py` — main Python script that runs the recommender and Gradio app
- `players_stats.csv` — Valorant player stats dataset used for recommendations
- `requirements.txt` — Python dependencies for the project

---

## How to run locally

1. **Clone this repo**

```bash
git clone https://github.com//javidanaslanli/valorant-esport-player-similarity.git
cd valorant-esport-player-similarity
