import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Try to import CatBoost
try:
    from catboost import CatBoostClassifier
    catboost_available = True
    print("CatBoost successfully imported!")
except ImportError:
    catboost_available = False
    print("CatBoost not available, skipping...")

# Load modeling data
try:
    modeling_data = pd.read_csv('data/processed/modeling_data.csv')
    print(f"Loaded modeling data with {modeling_data.shape[0]} matches and {modeling_data.shape[1]} features")
except FileNotFoundError:
    print("Modeling data file not found. Please run data_integration.py first.")
    exit(1)

# Feature selection
def select_features(df):
    """Select and prepare features for modeling"""
    
    # Drop non-predictive columns
    drop_cols = ['match_id', 'date', 'team1', 'team2', 'winner', 'toss_winner', 'toss_decision']
    drop_cols = [col for col in drop_cols if col in df.columns]
    features = df.drop(columns=drop_cols)
    
    # Define categorical features
    cat_features = ['venue', 'season'] 
    cat_features = [col for col in cat_features if col in features.columns]
    
    # Target variable
    if 'team1_won' in features.columns:
        target = features['team1_won'].astype(int)
        features = features.drop(columns=['team1_won'])
    else:
        print("Warning: target variable 'team1_won' not found. Model training cannot proceed.")
        exit(1)
    
    return features, cat_features, target

# Select features
features, categorical_features, target = select_features(modeling_data)
print(f"Selected {features.shape[1]} features, including {len(categorical_features)} categorical features")

# Split the data - stratify to ensure balanced classes
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
print(f"Training set class distribution: {pd.Series(y_train).value_counts(normalize=True)}")
print(f"Testing set class distribution: {pd.Series(y_test).value_counts(normalize=True)}")

# Preprocessing pipeline for the data
numerical_features = [col for col in features.columns if col not in categorical_features]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Define models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
}

# Add CatBoost if available
if catboost_available:
    models['CatBoost'] = CatBoostClassifier(random_state=42, verbose=0)

# Define hyperparameter grids
param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5]
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9]
    }
}

# Add CatBoost parameters if available
if catboost_available:
    param_grids['CatBoost'] = {
        'classifier__iterations': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__depth': [4, 6, 8]
    }

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Train and evaluate models
results = {}
best_models = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Initial cross-validation score
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Initial 5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Hyperparameter tuning with grid search
    grid_search = GridSearchCV(
        pipeline, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Best model after tuning
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV accuracy: {best_cv_score:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:,1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"ROC-AUC score: {roc_auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # Save results
    results[model_name] = {
        'initial_cv_score': cv_scores.mean(),
        'best_cv_score': best_cv_score,
        'test_accuracy': test_accuracy,
        'roc_auc': roc_auc,
        'best_params': best_params
    }
    
    # Save the best model
    best_models[model_name] = best_model
    joblib.dump(best_model, f'models/{model_name}_model.pkl')

# Determine the best overall model
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
best_accuracy = results[best_model_name]['test_accuracy']

print(f"\nBest overall model: {best_model_name} with test accuracy of {best_accuracy:.4f}")

# Save the best model with a special name
joblib.dump(best_models[best_model_name], 'models/best_model.pkl')

# Save results to a text file
with open('results/model_evaluation.txt', 'w') as f:
    f.write(f"Model Evaluation Results\n")
    f.write(f"======================\n\n")
    
    for model_name, result in results.items():
        f.write(f"{model_name}:\n")
        f.write(f"  Initial CV Score: {result['initial_cv_score']:.4f}\n")
        f.write(f"  Best CV Score: {result['best_cv_score']:.4f}\n")
        f.write(f"  Test Accuracy: {result['test_accuracy']:.4f}\n")
        f.write(f"  ROC-AUC Score: {result['roc_auc']:.4f}\n")
        f.write(f"  Best Parameters: {result['best_params']}\n\n")
    
    f.write(f"Best Overall Model: {best_model_name} with test accuracy of {best_accuracy:.4f}\n")

# Plot feature importances if possible
def get_feature_importances(model, preprocessor, feature_names):
    """Extract feature importances from the pipeline model"""
    if hasattr(model, 'feature_importances_'):
        # Get the feature names after preprocessing
        cat_cols = preprocessor.transformers_[1][2]  # Categorical columns
        one_hot_encoder = preprocessor.transformers_[1][1]  # OneHotEncoder
        
        # Get one-hot encoded feature names
        cat_features = []
        for i, col in enumerate(cat_cols):
            categories = one_hot_encoder.categories_[i]
            for category in categories:
                cat_features.append(f"{col}_{category}")
        
        # Get all feature names
        all_features = numerical_features + cat_features
        
        # Return feature importances with names
        return list(zip(all_features, model.feature_importances_))
    
    return None

# Try to get feature importances from the best model
if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'CatBoost']:
    try:
        # Get preprocessor and classifier from pipeline
        preprocessor = best_models[best_model_name].named_steps['preprocessor']
        classifier = best_models[best_model_name].named_steps['classifier']
        
        # Get feature importances
        if hasattr(classifier, 'feature_importances_'):
            importances = get_feature_importances(classifier, preprocessor, features.columns)
            
            if importances:
                # Sort by importance
                importances = sorted(importances, key=lambda x: x[1], reverse=True)
                
                # Plot top 20 features
                plt.figure(figsize=(12, 8))
                feature_names = [x[0] for x in importances[:20]]
                feature_values = [x[1] for x in importances[:20]]
                
                plt.barh(range(len(feature_names)), feature_values, align='center')
                plt.yticks(range(len(feature_names)), feature_names)
                plt.xlabel('Feature Importance')
                plt.ylabel('Feature')
                plt.title(f'Top 20 Features ({best_model_name})')
                plt.tight_layout()
                
                # Save plot
                plt.savefig('results/feature_importance.png')
                print("\nFeature importance plot saved to results/feature_importance.png")
                
                # Save feature importances to CSV
                feature_importance_df = pd.DataFrame(importances, columns=['Feature', 'Importance'])
                feature_importance_df.to_csv('results/feature_importances.csv', index=False)
    except Exception as e:
        print(f"\nCould not extract feature importances: {e}")

print("\nModel training complete!")