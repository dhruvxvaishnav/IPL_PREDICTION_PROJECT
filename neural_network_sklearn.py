import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

print("Starting neural network training using scikit-learn...")

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load modeling data
try:
    modeling_data = pd.read_csv('data/processed/modeling_data.csv')
    print(f"Loaded modeling data: {modeling_data.shape[0]} rows, {modeling_data.shape[1]} columns")
except FileNotFoundError:
    print("Modeling data not found! Please run data_integration.py first.")
    exit(1)

# Select features (excluding non-predictive columns)
drop_cols = ['match_id', 'date', 'team1', 'team2', 'winner', 'toss_winner', 'toss_decision']
drop_cols = [col for col in drop_cols if col in modeling_data.columns]

# Separate features and target
if 'team1_won' in modeling_data.columns:
    y = modeling_data['team1_won'].astype(int)
    X = modeling_data.drop(columns=['team1_won'] + drop_cols)
else:
    print("Error: target variable 'team1_won' not found.")
    exit(1)

# Identify numerical and categorical features
categorical_features = ['venue', 'season']
categorical_features = [col for col in categorical_features if col in X.columns]
numerical_features = [col for col in X.columns if col not in categorical_features]

print(f"Selected {X.shape[1]} features, including {len(categorical_features)} categorical and {len(numerical_features)} numerical features")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# This preprocessor handles both numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Apply preprocessing to get transformed data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

print(f"Preprocessed training data shape: {X_train_preprocessed.shape}")

# Define neural networks with different architectures
nn_configs = {
    'small_nn': {
        'hidden_layer_sizes': (50, 25),
        'max_iter': 1000,
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'learning_rate': 'adaptive',
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10
    },
    'medium_nn': {
        'hidden_layer_sizes': (100, 50),
        'max_iter': 1000,
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'learning_rate': 'adaptive',
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10
    },
    'large_nn': {
        'hidden_layer_sizes': (150, 75, 30),
        'max_iter': 1000,
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'learning_rate': 'adaptive',
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10
    }
}

# Train and evaluate neural networks
results = {}

for name, config in nn_configs.items():
    print(f"\nTraining {name}...")
    
    # Create and train the model
    nn = MLPClassifier(**config)
    nn.fit(X_train_preprocessed, y_train)
    
    # Evaluate on test set
    y_pred = nn.predict(X_test_preprocessed)
    y_pred_proba = nn.predict_proba(X_test_preprocessed)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Save model and results
    joblib.dump(nn, f'models/{name}_model.pkl')
    
    results[name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'architecture': config['hidden_layer_sizes']
    }

# Save preprocessor for prediction
joblib.dump(preprocessor, 'models/nn_preprocessor.pkl')

# Find best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest neural network: {best_model_name} with accuracy {best_accuracy:.4f}")

# Save the best model as the default neural network model
best_model = joblib.load(f'models/{best_model_name}_model.pkl')
joblib.dump(best_model, 'models/best_nn_model.pkl')

# Create wrapper pipeline to include preprocessing
best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])
joblib.dump(best_pipeline, 'models/best_nn_pipeline.pkl')

# Save results to a file
with open('results/nn_model_evaluation.txt', 'w') as f:
    f.write("Neural Network Model Evaluation\n")
    f.write("============================\n\n")
    
    for model_name, result in results.items():
        f.write(f"{model_name}:\n")
        f.write(f"  Architecture: {result['architecture']}\n")
        f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
        f.write(f"  ROC AUC: {result['roc_auc']:.4f}\n\n")
    
    f.write(f"Best Model: {best_model_name} with accuracy {best_accuracy:.4f}\n")

# Plot loss curve for the best model
if hasattr(best_model, 'loss_curve_'):
    plt.figure(figsize=(10, 6))
    plt.plot(best_model.loss_curve_)
    plt.title(f'Loss Curve for {best_model_name}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('results/nn_loss_curve.png')
    print(f"Loss curve saved to results/nn_loss_curve.png")

# Compare with traditional ML models if available
if os.path.exists('results/model_evaluation.txt'):
    with open('results/model_evaluation.txt', 'r') as f:
        ml_results = f.read()
    
    print("\nComparing with traditional ML models:")
    print(ml_results)

print("\nNeural network training complete!")