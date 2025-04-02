import pandas as pd
import numpy as np
import joblib
import os

def predict_match(team1, team2, venue, toss_winner=None, toss_decision=None):
    """
    Predict the outcome of an IPL match
    
    Parameters:
    -----------
    team1: str
        First team name
    team2: str
        Second team name
    venue: str
        Match venue
    toss_winner: str, optional
        Team that won the toss (if known)
    toss_decision: str, optional
        Toss decision ('bat' or 'field') if toss is known
    
    Returns:
    --------
    dict
        Prediction results including predicted winner and probability
    """
    # Load required files
    try:
        venue_stats = pd.read_csv('data/processed/venue_stats.csv', index_col=0)
        team_venue_stats = pd.read_csv('data/processed/team_venue_stats.csv')
        h2h_records = pd.read_csv('data/processed/h2h_records.csv')
        team_strengths = pd.read_csv('data/processed/team_strengths.csv')
    except FileNotFoundError as e:
        print(f"Required data file not found: {e}")
        print("Please run data_integration.py first.")
        return None
    
    # Load the trained model
    if not os.path.exists('models/best_model.pkl'):
        print("Model not found! Please run model_training.py first.")
        return None
        
    model = joblib.load('models/best_model.pkl')
    
    # Convert team strengths to dictionary
    team_strength_dict = {}
    for _, row in team_strengths.iterrows():
        team_strength_dict[row['team']] = {
            'win_rate': row['win_rate'] if 'win_rate' in row else 0.5,
            'form_score': row['form_score'] if 'form_score' in row else 0.5
        }
    
    # Get venue home teams mapping
    venue_mapping = {}
    for _, row in team_venue_stats.groupby('venue').apply(lambda x: x.loc[x['played'].idxmax()]).iterrows():
        venue_mapping[row['venue']] = row['team']
    
    # Prepare feature vector
    features = {}
    
    # Get current season
    current_season = 2024
    
    # Team info
    features['season'] = current_season
    features['venue'] = venue
    
    # Home advantage
    home_team = venue_mapping.get(venue, None)
    features['team1_is_home'] = 1 if home_team == team1 else 0
    features['team2_is_home'] = 1 if home_team == team2 else 0
    
    # Venue stats
    if venue in venue_stats.index and 'batting_first_win_rate' in venue_stats.columns:
        features['venue_batting_first_win_rate'] = venue_stats.loc[venue]['batting_first_win_rate']
    else:
        features['venue_batting_first_win_rate'] = 0.5
    
    # Team venue stats
    team1_venue_stats_row = team_venue_stats[(team_venue_stats['team'] == team1) & 
                                            (team_venue_stats['venue'] == venue)]
    team2_venue_stats_row = team_venue_stats[(team_venue_stats['team'] == team2) & 
                                            (team_venue_stats['venue'] == venue)]
    
    features['team1_venue_win_rate'] = team1_venue_stats_row['win_rate'].values[0] if not team1_venue_stats_row.empty else 0.5
    features['team2_venue_win_rate'] = team2_venue_stats_row['win_rate'].values[0] if not team2_venue_stats_row.empty else 0.5
    
    # Team form
    features['team1_recent_win_rate'] = team_strength_dict.get(team1, {}).get('win_rate', 0.5)
    features['team2_recent_win_rate'] = team_strength_dict.get(team2, {}).get('win_rate', 0.5)
    features['team1_form_score'] = team_strength_dict.get(team1, {}).get('form_score', 0.5)
    features['team2_form_score'] = team_strength_dict.get(team2, {}).get('form_score', 0.5)
    
    # Head-to-head records
    teams_sorted = sorted([team1, team2])
    h2h_row = h2h_records[(h2h_records['team1'] == teams_sorted[0]) & 
                         (h2h_records['team2'] == teams_sorted[1])]
    
    if not h2h_row.empty:
        if teams_sorted[0] == team1:
            features['team1_h2h_win_rate'] = h2h_row['team1_win_pct'].values[0]
            features['team2_h2h_win_rate'] = h2h_row['team2_win_pct'].values[0]
        else:
            features['team1_h2h_win_rate'] = h2h_row['team2_win_pct'].values[0]
            features['team2_h2h_win_rate'] = h2h_row['team1_win_pct'].values[0]
    else:
        features['team1_h2h_win_rate'] = 0.5
        features['team2_h2h_win_rate'] = 0.5
    
    # Toss-related features
    if toss_winner and toss_decision:
        features['team1_won_toss'] = 1 if toss_winner == team1 else 0
        features['team2_won_toss'] = 1 if toss_winner == team2 else 0
        features['toss_decision_bat'] = 1 if toss_decision.lower() == 'bat' else 0
    else:
        features['team1_won_toss'] = 0.5  # Unknown
        features['team2_won_toss'] = 0.5  # Unknown
        features['toss_decision_bat'] = 0.5  # Unknown
    
    # Convert to DataFrame
    X = pd.DataFrame([features])
    
    # Make prediction
    try:
        win_probability = model.predict_proba(X)[0][1]
    except Exception as e:
        print(f"Error making prediction: {e}")
        win_probability = 0.5
    
    team1_win_prob = float(win_probability)
    predicted_winner = team1 if team1_win_prob > 0.5 else team2
    prediction_prob = team1_win_prob if team1_win_prob > 0.5 else 1 - team1_win_prob
    
    # Determine confidence level
    if prediction_prob >= 0.8:
        confidence = "High"
    elif prediction_prob >= 0.65:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Prepare result
    result = {
        'team1': team1,
        'team2': team2,
        'predicted_winner': predicted_winner,
        'win_probability': prediction_prob * 100,  # Convert to percentage
        'confidence': confidence,
        'team1_win_probability': team1_win_prob * 100,  # Convert to percentage
        'team2_win_probability': (1 - team1_win_prob) * 100,  # Convert to percentage
        'match_details': {
            'venue': venue,
            'team1_is_home': features['team1_is_home'],
            'team2_is_home': features['team2_is_home'],
            'team1_recent_form': features['team1_form_score'],
            'team2_recent_form': features['team2_form_score'],
            'team1_h2h_win_rate': features['team1_h2h_win_rate'],
            'team2_h2h_win_rate': features['team2_h2h_win_rate'],
            'toss_winner': toss_winner,
            'toss_decision': toss_decision
        }
    }
    
    return result

if __name__ == "__main__":
    # Example usage
    prediction = predict_match(
        "Chennai Super Kings", 
        "Mumbai Indians",
        "M.A. Chidambaram Stadium",
        "Chennai Super Kings",  # Toss winner
        "bat"  # Toss decision
    )
    
    if prediction:
        print("\nMatch Prediction")
        print("================")
        print(f"Team 1: {prediction['team1']}")
        print(f"Team 2: {prediction['team2']}")
        print(f"Predicted Winner: {prediction['predicted_winner']}")
        print(f"Win Probability: {prediction['win_probability']:.2f}%")
        print(f"Confidence: {prediction['confidence']}")
        
        print(f"\nTeam 1 Win Probability: {prediction['team1_win_probability']:.2f}%")
        print(f"Team 2 Win Probability: {prediction['team2_win_probability']:.2f}%")
        
        print("\nMatch Details:")
        print(f"Venue: {prediction['match_details']['venue']}")
        print(f"Team 1 is Home Team: {'Yes' if prediction['match_details']['team1_is_home'] else 'No'}")
        print(f"Team 1 Recent Form: {prediction['match_details']['team1_recent_form']:.2f}")
        print(f"Team 2 Recent Form: {prediction['match_details']['team2_recent_form']:.2f}")
        print(f"Team 1 H2H Win Rate: {prediction['match_details']['team1_h2h_win_rate']:.2f}")
        print(f"Team 2 H2H Win Rate: {prediction['match_details']['team2_h2h_win_rate']:.2f}")
        
        if prediction['match_details']['toss_winner']:
            print(f"Toss Winner: {prediction['match_details']['toss_winner']}")
            print(f"Toss Decision: {prediction['match_details']['toss_decision']}")