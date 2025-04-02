import pandas as pd
import os
from predict_match import predict_match

def get_available_teams():
    """Return a list of available teams"""
    try:
        team_venue_stats = pd.read_csv('data/processed/team_venue_stats.csv')
        return sorted(team_venue_stats['team'].unique())
    except:
        return [
            "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore", 
            "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad", 
            "Punjab Kings", "Rajasthan Royals", "Gujarat Titans", "Lucknow Super Giants"
        ]

def get_venues():
    """Return available venues"""
    try:
        team_venue_stats = pd.read_csv('data/processed/team_venue_stats.csv')
        return sorted(team_venue_stats['venue'].unique())
    except:
        return [
            "M.A. Chidambaram Stadium", "Wankhede Stadium", "M. Chinnaswamy Stadium",
            "Eden Gardens", "Arun Jaitley Stadium", "Rajiv Gandhi International Stadium",
            "Punjab Cricket Association Stadium", "Sawai Mansingh Stadium", 
            "Narendra Modi Stadium", "BRSABV Ekana Cricket Stadium"
        ]

def print_menu(options):
    """Print a numbered menu of options"""
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")

def get_user_choice(prompt, options):
    """Get user choice from a list of options"""
    while True:
        print_menu(options)
        try:
            choice = int(input(prompt))
            if 1 <= choice <= len(options):
                return options[choice-1]
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    """Main function for the prediction CLI"""
    print("=" * 50)
    print("IPL Match Prediction System")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('models/best_model.pkl'):
        print("Error: Model files not found. Please run model_training.py first.")
        return
    
    # Get available teams and venues
    teams = get_available_teams()
    venues = get_venues()
    
    # Select teams
    print("\nSelect the first team:")
    team1 = get_user_choice("Enter team number: ", teams)
    
    remaining_teams = [team for team in teams if team != team1]
    print("\nSelect the second team:")
    team2 = get_user_choice("Enter team number: ", remaining_teams)
    
    # Select venue
    print("\nSelect the venue:")
    venue = get_user_choice("Enter venue number: ", venues)
    
    # Toss information
    print("\nDo you know the toss result?")
    know_toss = get_user_choice("Enter your choice: ", ["Yes", "No"])
    
    toss_winner = None
    toss_decision = None
    
    if know_toss == "Yes":
        print("\nWhich team won the toss?")
        toss_winner = get_user_choice("Enter team: ", [team1, team2])
        
        print("\nWhat did they decide?")
        toss_decision = get_user_choice("Enter decision: ", ["bat", "field"])
    
    # Make prediction
    prediction = predict_match(team1, team2, venue, toss_winner, toss_decision)
    
    if prediction:
        # Display prediction
        print("\n" + "=" * 50)
        print("Match Prediction Results")
        print("=" * 50)
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

if __name__ == "__main__":
    main()