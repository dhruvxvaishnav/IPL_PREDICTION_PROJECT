# 🏏 IPL Match Prediction System

An AI-powered system that predicts IPL cricket match outcomes with 73.85% accuracy using machine learning and historical data analysis.

## 🚀 Overview

This project analyzes 16+ years of IPL data (2008-2024) to predict match winners using multiple ML algorithms. The system considers team performance, venue statistics, head-to-head records, recent form, and toss decisions to make intelligent predictions.

## ✨ Key Features

- **High Accuracy Models**: Neural Networks (73.85%), Gradient Boosting (72.94%)
- **Interactive CLI**: User-friendly prediction interface
- **Smart Analytics**: Venue advantage analysis and team form tracking
- **Multiple Algorithms**: Random Forest, XGBoost, Neural Networks, Gradient Boosting
- **Real-time Predictions**: Input teams, venue, and toss details for instant results

## 🏗️ Project Structure

```
📁 ipl-prediction/
├── 📊 data/
│   ├── raw/           # Original CSV datasets
│   └── processed/     # Cleaned feature data
├── 🤖 models/         # Trained ML models (.pkl files)
├── 📈 results/        # Performance metrics & visualizations
├── 🎯 main.py         # Complete pipeline runner
├── 🔧 data_integration.py    # Data preprocessing
├── 🏋️ model_training.py      # ML model training
├── 🧠 neural_network_sklearn.py  # Neural network training
├── 💬 predict_cli.py         # Interactive prediction tool
└── ⚡ predict_match.py       # Core prediction engine
```

## 🎯 Quick Start

### 1. Setup Data Files
Place these CSV files in `data/raw/`:
- `cricket_data_2025.csv` (player statistics)
- `ipl_matches_detail_2008_2024.csv` (match details)
- `ipl_matches_2008_2024.csv` (ball-by-ball data)

### 2. Run Complete Pipeline
```bash
python main.py
```
Choose option 1 for full setup: data processing → model training → predictions

### 3. Make Predictions
```bash
python predict_cli.py  # Traditional ML
python predict_with_nn.py  # Neural Network
```

## 📊 Model Performance

| Model | Accuracy | ROC-AUC | Best For |
|-------|----------|---------|----------|
| **Neural Network** | **73.85%** | **0.78** | **Overall best** |
| Gradient Boosting | 72.94% | 0.82 | Interpretability |
| XGBoost | 68.81% | 0.80 | Feature importance |
| Random Forest | 67.43% | 0.75 | Baseline |

## 🔍 Key Prediction Features

1. **Team venue performance** (78% importance)
2. **Season trends** (9.4% importance)  
3. **Head-to-head records** (4.5% importance)
4. **Home ground advantage** (4.2% importance)
5. **Recent team form** (3.3% importance)

## 💻 Usage Examples

### Basic Prediction
```python
from predict_match import predict_match

result = predict_match(
    team1="Chennai Super Kings",
    team2="Mumbai Indians", 
    venue="M.A. Chidambaram Stadium",
    toss_winner="Chennai Super Kings",
    toss_decision="bat"
)

print(f"🏆 Winner: {result['predicted_winner']}")
print(f"📈 Confidence: {result['win_probability']:.1f}%")
```

### Interactive Menu
```bash
python main.py
# Choose from:
# 1. Complete pipeline
# 2. Data processing only  
# 3. Train models
# 4. Make predictions
# 5. Compare model performance
```

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

## 📋 Available Teams (2024)
- Chennai Super Kings
- Mumbai Indians  
- Royal Challengers Bangalore
- Kolkata Knight Riders
- Delhi Capitals
- Sunrisers Hyderabad
- Punjab Kings
- Rajasthan Royals
- Gujarat Titans
- Lucknow Super Giants

## 🏟️ Supported Venues
- M.A. Chidambaram Stadium (Chennai)
- Wankhede Stadium (Mumbai)
- M. Chinnaswamy Stadium (Bangalore)
- Eden Gardens (Kolkata)
- Arun Jaitley Stadium (Delhi)
- And 5+ more major IPL venues

## 🎲 Prediction Output
```
Match Prediction Results
========================
Team 1: Chennai Super Kings
Team 2: Mumbai Indians
🏆 Predicted Winner: Chennai Super Kings
📊 Win Probability: 67.23%
🎯 Confidence: Medium

Team Probabilities:
CSK: 67.23% | MI: 32.77%

Match Factors:
🏠 Home Advantage: Yes (CSK)
📈 Recent Form: CSK (0.75) vs MI (0.60)
⚔️ Head-to-Head: CSK leads 52%
🎲 Toss Impact: Won toss, chose to bat
```

## 🚀 Future Enhancements
- Player impact analysis
- Weather condition integration
- Live match probability updates
- Web dashboard interface
- API endpoints for external use

---

*Built with ❤️ for cricket analytics and machine learning enthusiasts*
