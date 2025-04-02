import pandas as pd
import numpy as np
import os

print("Starting data integration process...")

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load datasets
try:
    player_data = pd.read_csv('data/raw/cricket_data_2025.csv')
    match_detail_data = pd.read_csv('data/raw/ipl_matches_detail_2008_2024.csv')
    print("Player data shape:", player_data.shape)
    print("Match detail data shape:", match_detail_data.shape)
    
    # Try to load ball by ball data if available
    try:
        ball_by_ball_data = pd.read_csv('data/raw/ipl_matches_2008_2024.csv')
        print("Ball-by-ball data loaded successfully. Shape:", ball_by_ball_data.shape)
    except FileNotFoundError:
        ball_by_ball_data = None
        print("Ball-by-ball data not found. Proceeding with match details only.")
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print("Please make sure all CSV files are in the data/raw directory")
    exit(1)

# Clean player data
def clean_player_data(df):
    # Convert numeric columns
    numeric_cols = [
        'Matches_Batted', 'Not_Outs', 'Runs_Scored', 'Balls_Faced', 
        'Batting_Average', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries', 
        'Fours', 'Sixes', 'Catches_Taken', 'Stumpings', 'Matches_Bowled', 
        'Balls_Bowled', 'Runs_Conceded', 'Wickets_Taken', 'Bowling_Average', 
        'Economy_Rate', 'Bowling_Strike_Rate', 'Four_Wicket_Hauls', 'Five_Wicket_Hauls'
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean player names to ensure consistency
    df['Player_Name'] = df['Player_Name'].str.strip()
    
    return df

# Calculate player impact metrics
def calculate_player_metrics(df):
    # Calculate batting impact (0-10)
    def batting_impact(row):
        if pd.isna(row['Batting_Average']) or pd.isna(row['Batting_Strike_Rate']) or row['Matches_Batted'] < 5:
            return 0
        
        avg_impact = min(10, row['Batting_Average'] / 50 * 10) * 0.4
        sr_impact = min(10, row['Batting_Strike_Rate'] / 150 * 10) * 0.4
        
        matches = max(1, row['Matches_Batted'])
        milestone_ratio = (row['Centuries'] * 2 + row['Half_Centuries']) / matches
        consistency_impact = min(10, milestone_ratio * 10) * 0.2
        
        return min(10, avg_impact + sr_impact + consistency_impact)
    
    # Calculate bowling impact (0-10)
    def bowling_impact(row):
        if pd.isna(row['Bowling_Average']) or pd.isna(row['Economy_Rate']) or row['Matches_Bowled'] < 5:
            return 0
        
        avg_impact = min(10, (40 - min(40, row['Bowling_Average'])) / 20 * 10) * 0.4
        eco_impact = min(10, (12 - min(12, row['Economy_Rate'])) / 6 * 10) * 0.3
        
        matches = max(1, row['Matches_Bowled'])
        wickets_per_match = row['Wickets_Taken'] / matches
        wicket_impact = min(10, wickets_per_match / 2 * 10) * 0.3
        
        return min(10, avg_impact + eco_impact + wicket_impact)
    
    # Apply impact calculations
    df['Batting_Impact'] = df.apply(batting_impact, axis=1)
    df['Bowling_Impact'] = df.apply(bowling_impact, axis=1)
    df['Total_Impact'] = df['Batting_Impact'] + df['Bowling_Impact']
    
    # Determine player role
    def player_role(row):
        if row['Batting_Impact'] >= 6 and row['Bowling_Impact'] >= 6:
            return "All-rounder"
        elif row['Batting_Impact'] >= 6:
            return "Batsman"
        elif row['Bowling_Impact'] >= 6:
            return "Bowler"
        elif row['Batting_Impact'] >= 4 and row['Bowling_Impact'] >= 4:
            return "All-rounder"
        elif row['Batting_Impact'] >= 4:
            return "Batsman"
        elif row['Bowling_Impact'] >= 4:
            return "Bowler"
        else:
            return "Unknown"
    
    df['Player_Role'] = df.apply(player_role, axis=1)
    
    return df

# Clean and process match data
def process_match_data(match_data):
    # Clean column names and values
    match_data.columns = match_data.columns.str.lower()
    
    # Ensure consistent team names
    def standardize_team_name(name):
        if pd.isnull(name):
            return name
            
        name = str(name).strip()
        # Standard mappings for team names
        team_mappings = {
            'delhi daredevils': 'Delhi Capitals',
            'deccan chargers': 'Sunrisers Hyderabad',
            'gujarat lions': 'Gujarat Titans',
            'rising pune supergiant': 'Rising Pune Supergiants',
            'rising pune supergiants': 'Rising Pune Supergiants',
            'pune warriors': 'Pune Warriors India',
            'pune warriors india': 'Pune Warriors India',
            'kings xi punjab': 'Punjab Kings'
        }
        # Fix capitalization
        if name.lower() in team_mappings:
            return team_mappings[name.lower()]
        else:
            # Capitalize each word
            return ' '.join(word.capitalize() for word in name.split())
    
    # Standardize team names
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        if col in match_data.columns:
            match_data[col] = match_data[col].apply(lambda x: standardize_team_name(x) if pd.notnull(x) else x)
    
    # Extract key match information
    essential_columns = [
        'id', 'season', 'city', 'date', 'venue', 'team1', 'team2', 
        'toss_winner', 'toss_decision', 'winner', 'result', 'result_margin',
        'player_of_match', 'target_runs', 'target_overs'
    ]
    
    # Keep only columns that exist in the DataFrame
    columns_to_keep = [col for col in essential_columns if col in match_data.columns]
    matches = match_data[columns_to_keep].copy()
    
    # Convert date to datetime
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    
    # Create features
    matches['home_advantage'] = 0  # Will update this later with venue-team mapping
    matches['toss_advantage'] = matches['toss_winner'] == matches['winner']
    
    # Extract batting first info
    try:
        matches['batting_first'] = matches.apply(
            lambda x: x['team1'] if (x['toss_winner'] == x['team1'] and x['toss_decision'] == 'bat') or 
                                (x['toss_winner'] == x['team2'] and x['toss_decision'] == 'field') 
                    else x['team2'], axis=1
        )
        
        # Create target variable: winning_team_batted_first
        matches['winning_team_batted_first'] = matches.apply(
            lambda x: x['batting_first'] == x['winner'] if pd.notnull(x['winner']) else np.nan, axis=1
        )
    except KeyError as e:
        print(f"Warning: Could not create batting first features: {e}")
    
    # Calculate venue stats - win rate when batting first
    try:
        venue_stats = matches.groupby('venue').agg(
            matches_played=('id', 'count'),
            batting_first_wins=('winning_team_batted_first', 'sum')
        )
        venue_stats['batting_first_win_rate'] = venue_stats['batting_first_wins'] / venue_stats['matches_played']
    except:
        print("Warning: Could not calculate venue batting first statistics")
        venue_stats = pd.DataFrame(columns=['venue', 'matches_played', 'batting_first_wins', 'batting_first_win_rate'])
    
    # Calculate team performance at each venue
    team_venue_stats = pd.DataFrame()
    for team in matches['team1'].dropna().unique():
        team_home_matches = matches[(matches['team1'] == team) | (matches['team2'] == team)]
        team_home_wins = team_home_matches[team_home_matches['winner'] == team]
        
        if not team_home_matches.empty:
            team_venues = team_home_matches.groupby('venue').agg(
                played=('id', 'count')
            )
            
            team_wins = pd.DataFrame(index=team_venues.index, columns=['won'])
            if not team_home_wins.empty:
                team_wins_grp = team_home_wins.groupby('venue').agg(
                    won=('id', 'count')
                )
                team_wins.update(team_wins_grp)
            
            team_results = pd.merge(team_venues, team_wins, left_index=True, right_index=True, how='left')
            team_results['won'] = team_results['won'].fillna(0)
            team_results['win_rate'] = team_results['won'] / team_results['played']
            team_results['team'] = team
            
            team_venue_stats = pd.concat([team_venue_stats, team_results.reset_index()])
    
    return matches, venue_stats, team_venue_stats

# Team strength calculation
def calculate_team_strengths(matches_df, seasons_to_use=None):
    """Calculate team strengths based on recent performance"""
    # Get recent seasons if specified
    if seasons_to_use is None:
        # Use all seasons
        filtered_matches = matches_df.copy()
    else:
        # Get most recent n seasons
        max_season = matches_df['season'].max()
        recent_seasons = list(range(max_season - seasons_to_use + 1, max_season + 1))
        filtered_matches = matches_df[matches_df['season'].isin(recent_seasons)].copy()
    
    # Calculate team win rates
    team_stats = {}
    all_teams = set(filtered_matches['team1'].dropna().unique()) | set(filtered_matches['team2'].dropna().unique())
    
    for team in all_teams:
        team_matches = filtered_matches[(filtered_matches['team1'] == team) | (filtered_matches['team2'] == team)]
        team_wins = team_matches[team_matches['winner'] == team]
        
        total = len(team_matches)
        wins = len(team_wins)
        win_rate = wins / total if total > 0 else 0
        
        team_stats[team] = {
            'matches_played': total,
            'wins': wins,
            'win_rate': win_rate,
            'recent_form': [] # Will populate with recent results
        }
    
    # Add recent form (last 5 matches)
    for team in all_teams:
        team_matches = filtered_matches[(filtered_matches['team1'] == team) | (filtered_matches['team2'] == team)]
        team_matches = team_matches.sort_values('date', ascending=False).head(5)
        
        form = []
        for _, match in team_matches.iterrows():
            if match['winner'] == team:
                form.append(1) # Win
            else:
                form.append(0) # Loss
        
        team_stats[team]['recent_form'] = form
        team_stats[team]['form_score'] = sum(form) / len(form) if form else 0
    
    return team_stats

# Create venue home teams mapping
def create_venue_mapping(match_data):
    venue_mapping = {}
    
    # For each venue, find the team that plays there most often
    for venue in match_data['venue'].dropna().unique():
        venue_matches = match_data[match_data['venue'] == venue]
        team1_counts = venue_matches['team1'].value_counts()
        team2_counts = venue_matches['team2'].value_counts()
        
        # Combine counts
        team_counts = pd.concat([team1_counts, team2_counts]).groupby(level=0).sum()
        
        if not team_counts.empty:
            home_team = team_counts.idxmax()
            venue_mapping[venue] = home_team
    
    return venue_mapping

# Calculate head-to-head records
def calculate_head_to_head(matches_data):
    h2h_stats = {}
    
    for _, match in matches_data.iterrows():
        team1 = match['team1']
        team2 = match['team2']
        winner = match['winner']
        
        # Skip if any team is NaN
        if pd.isnull(team1) or pd.isnull(team2):
            continue
            
        # Ensure teams are in alphabetical order for consistent key
        teams = sorted([team1, team2])
        key = f"{teams[0]}_vs_{teams[1]}"
        
        if key not in h2h_stats:
            h2h_stats[key] = {'team1': teams[0], 'team2': teams[1], 'matches': 0, 'team1_wins': 0, 'team2_wins': 0, 'no_result': 0}
        
        h2h_stats[key]['matches'] += 1
        
        if pd.isnull(winner):
            h2h_stats[key]['no_result'] += 1
        elif winner == teams[0]:
            h2h_stats[key]['team1_wins'] += 1
        elif winner == teams[1]:
            h2h_stats[key]['team2_wins'] += 1
        else:
            h2h_stats[key]['no_result'] += 1
    
    # Calculate win percentages
    for key in h2h_stats:
        stats = h2h_stats[key]
        total_results = stats['team1_wins'] + stats['team2_wins']
        
        if total_results > 0:
            stats['team1_win_pct'] = stats['team1_wins'] / total_results
            stats['team2_win_pct'] = stats['team2_wins'] / total_results
        else:
            stats['team1_win_pct'] = 0.5
            stats['team2_win_pct'] = 0.5
    
    return pd.DataFrame(list(h2h_stats.values()))

# Create a composite match dataset for modeling
def prepare_modeling_dataset(matches_data, venue_stats, team_venue_stats, venue_mapping, team_strengths):
    modeling_data = []
    
    for _, match in matches_data.iterrows():
        # Skip matches with missing key data
        if pd.isnull(match['team1']) or pd.isnull(match['team2']) or pd.isnull(match['venue']):
            continue
            
        team1 = match['team1']
        team2 = match['team2']
        venue = match['venue']
        
        # Skip matches with no winner (if not NaN)
        if 'winner' in matches_data.columns and pd.isnull(match['winner']):
            continue
            
        # Get venue statistics
        venue_win_rate = venue_stats.loc[venue]['batting_first_win_rate'] if venue in venue_stats.index else 0.5
        
        # Get team venue statistics
        team1_venue_stats = team_venue_stats[(team_venue_stats['team'] == team1) & (team_venue_stats['venue'] == venue)]
        team2_venue_stats = team_venue_stats[(team_venue_stats['team'] == team2) & (team_venue_stats['venue'] == venue)]
        
        team1_venue_win_rate = team1_venue_stats['win_rate'].values[0] if not team1_venue_stats.empty else 0.5
        team2_venue_win_rate = team2_venue_stats['win_rate'].values[0] if not team2_venue_stats.empty else 0.5
        
        # Get team strengths and form
        team1_strength = team_strengths.get(team1, {'win_rate': 0.5, 'form_score': 0.5})
        team2_strength = team_strengths.get(team2, {'win_rate': 0.5, 'form_score': 0.5})
        
        # Get home team advantage
        home_team = venue_mapping.get(venue)
        team1_is_home = home_team == team1
        team2_is_home = home_team == team2
        
        # Get head-to-head
        teams_sorted = sorted([team1, team2])
        h2h_record = h2h_records[(h2h_records['team1'] == teams_sorted[0]) & (h2h_records['team2'] == teams_sorted[1])]
        
        if not h2h_record.empty:
            team1_h2h = h2h_record['team1_win_pct'].values[0] if teams_sorted[0] == team1 else h2h_record['team2_win_pct'].values[0]
            team2_h2h = h2h_record['team1_win_pct'].values[0] if teams_sorted[0] == team2 else h2h_record['team2_win_pct'].values[0]
        else:
            team1_h2h = 0.5
            team2_h2h = 0.5
        
        # Create feature record
        record = {
            'match_id': match['id'],
            'season': match['season'],
            'date': match['date'],
            'venue': venue,
            'team1': team1,
            'team2': team2,
            'toss_winner': match['toss_winner'] if 'toss_winner' in match and pd.notnull(match['toss_winner']) else None,
            'toss_decision': match['toss_decision'] if 'toss_decision' in match and pd.notnull(match['toss_decision']) else None,
            'team1_is_home': team1_is_home,
            'team2_is_home': team2_is_home,
            'venue_batting_first_win_rate': venue_win_rate,
            'team1_venue_win_rate': team1_venue_win_rate,
            'team2_venue_win_rate': team2_venue_win_rate,
            'team1_recent_win_rate': team1_strength.get('win_rate', 0.5),
            'team2_recent_win_rate': team2_strength.get('win_rate', 0.5),
            'team1_form_score': team1_strength.get('form_score', 0.5),
            'team2_form_score': team2_strength.get('form_score', 0.5),
            'team1_h2h_win_rate': team1_h2h,
            'team2_h2h_win_rate': team2_h2h,
        }
        
        # Add toss-related features if available
        if 'toss_winner' in match and pd.notnull(match['toss_winner']):
            record['team1_won_toss'] = match['toss_winner'] == team1
            record['team2_won_toss'] = match['toss_winner'] == team2
            if 'toss_decision' in match and pd.notnull(match['toss_decision']):
                record['toss_decision_bat'] = match['toss_decision'] == 'bat'
        
        # Add result if available
        if 'winner' in match and pd.notnull(match['winner']):
            record['winner'] = match['winner']
            record['team1_won'] = match['winner'] == team1
        
        modeling_data.append(record)
    
    return pd.DataFrame(modeling_data)

print("Processing data...")

# Process the data
cleaned_player_data = clean_player_data(player_data)
player_metrics = calculate_player_metrics(cleaned_player_data)
matches, venue_stats, team_venue_stats = process_match_data(match_detail_data)

# Calculate team strengths - all seasons and recent 2 seasons
all_team_strengths = calculate_team_strengths(matches)
recent_team_strengths = calculate_team_strengths(matches, seasons_to_use=2)

# Create venue to home team mapping
venue_mapping = create_venue_mapping(matches)

# Update home advantage based on venue mapping
matches['home_team'] = matches['venue'].map(venue_mapping)
matches['home_advantage'] = matches.apply(
    lambda x: 1 if pd.notnull(x['home_team']) and x['home_team'] == x['team1'] else (
        -1 if pd.notnull(x['home_team']) and x['home_team'] == x['team2'] else 0
    ), axis=1
)

# Calculate head-to-head records
h2h_records = calculate_head_to_head(matches)

# Save processed data
player_metrics.to_csv('data/processed/player_metrics.csv', index=False)
matches.to_csv('data/processed/matches.csv', index=False)
venue_stats.to_csv('data/processed/venue_stats.csv')
team_venue_stats.to_csv('data/processed/team_venue_stats.csv', index=False)
h2h_records.to_csv('data/processed/h2h_records.csv', index=False)

# Create a team strengths dataframe
team_strength_df = pd.DataFrame.from_dict(recent_team_strengths, orient='index')
team_strength_df.reset_index(inplace=True)
team_strength_df.rename(columns={'index': 'team'}, inplace=True)
team_strength_df.to_csv('data/processed/team_strengths.csv', index=False)

# Create modeling dataset
modeling_data = prepare_modeling_dataset(matches, venue_stats, team_venue_stats, venue_mapping, recent_team_strengths)
modeling_data.to_csv('data/processed/modeling_data.csv', index=False)

print("Data integration complete.")
print(f"Processed {player_metrics.shape[0]} player records")
print(f"Processed {matches.shape[0]} match records")
print(f"Created modeling dataset with {modeling_data.shape[0]} records and {modeling_data.shape[1]} features")