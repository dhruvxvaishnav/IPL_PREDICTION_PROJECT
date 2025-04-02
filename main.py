import os
import sys
import subprocess
import pandas as pd

def check_file_exists(filepath):
    """Check if a file exists"""
    return os.path.exists(filepath)

def run_script(script_name):
    """Run a Python script and capture output"""
    print(f"\nRunning {script_name}...")
    try:
        result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully ran {script_name}")
            # Print first 10 lines of output
            output_lines = result.stdout.split('\n')
            for line in output_lines[:10]:
                print(line)
            if len(output_lines) > 10:
                print("...")
        else:
            print(f"Error running {script_name}")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Exception running {script_name}: {e}")
        return False

def create_project_structure():
    """Create the project directory structure"""
    print("Creating project directory structure...")
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("Project directory structure created successfully!")

def check_data_files():
    """Check if the raw data files exist and copy them if needed"""
    player_data_path = 'data/raw/cricket_data_2025.csv'
    match_data_path = 'data/raw/ipl_matches_detail_2008_2024.csv'
    ball_data_path = 'data/raw/ipl_matches_2008_2024.csv'
    
    missing_files = []
    if not check_file_exists(player_data_path):
        missing_files.append('cricket_data_2025.csv')
    
    if not check_file_exists(match_data_path):
        missing_files.append('ipl_matches_detail_2008_2024.csv')
    
    if not check_file_exists(ball_data_path):
        missing_files.append('ipl_matches_2008_2024.csv')
    
    if missing_files:
        print(f"The following data files are missing: {', '.join(missing_files)}")
        print("Please place these files in the data/raw/ directory.")
        return False
    
    return True

def main():
    """Main function to run the IPL prediction model pipeline"""
    print("=" * 60)
    print("IPL CRICKET MATCH PREDICTION MODEL - PIPELINE")
    print("=" * 60)
    
    # Create project structure
    create_project_structure()
    
    # Check for data files
    if not check_data_files():
        return
    
    while True:
        print("\n" + "=" * 60)
        print("MENU")
        print("=" * 60)
        print("1. Run complete pipeline (data integration → training → prediction)")
        print("2. Data integration only")
        print("3. Traditional ML model training")
        print("4. Neural network training (scikit-learn)")
        print("5. Run traditional ML prediction interface")
        print("6. Run neural network prediction interface")
        print("7. Compare model performance")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == '1':
            # Run complete pipeline
            print("\nRunning complete pipeline...")
            
            run_script('data_integration.py')
            run_script('model_training.py')
            run_script('neural_network_sklearn.py')
            
            print("\nPipeline completed successfully!")
            print("Now you can run the prediction interfaces or compare model performance.")
            
            sub_choice = input("\nWhat would you like to do next? (1: Run predictions, 2: Compare models, 3: Return to menu): ")
            if sub_choice == '1':
                model_choice = input("Which model to use? (1: Traditional ML, 2: Neural Network): ")
                if model_choice == '1':
                    run_script('predict_cli.py')
                elif model_choice == '2':
                    run_script('predict_with_nn.py')
            elif sub_choice == '2':
                compare_model_performance()
        
        elif choice == '2':
            run_script('data_integration.py')
        
        elif choice == '3':
            if not check_file_exists('data/processed/modeling_data.csv'):
                print("Modeling data not found. Running data integration first...")
                run_script('data_integration.py')
            run_script('model_training.py')
        
        elif choice == '4':
            if not check_file_exists('data/processed/modeling_data.csv'):
                print("Modeling data not found. Running data integration first...")
                run_script('data_integration.py')
            run_script('neural_network_sklearn.py')
        
        elif choice == '5':
            if not check_file_exists('models/best_model.pkl'):
                print("Traditional ML model not found. Running model training first...")
                if not check_file_exists('data/processed/modeling_data.csv'):
                    print("Modeling data not found. Running data integration first...")
                    run_script('data_integration.py')
                run_script('model_training.py')
            run_script('predict_cli.py')
        
        elif choice == '6':
            if not check_file_exists('models/best_nn_pipeline.pkl'):
                print("Neural network model not found. Running neural network training first...")
                if not check_file_exists('data/processed/modeling_data.csv'):
                    print("Modeling data not found. Running data integration first...")
                    run_script('data_integration.py')
                run_script('neural_network_sklearn.py')
            run_script('predict_with_nn.py')
        
        elif choice == '7':
            compare_model_performance()
        
        elif choice == '8':
            print("\nExiting the pipeline. Thank you!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number between 1 and 8.")

def compare_model_performance():
    """Compare performance of traditional ML models and neural network"""
    print("\nComparing Model Performance")
    print("=========================")
    
    # Check if evaluation files exist
    ml_eval_path = 'results/model_evaluation.txt'
    nn_eval_path = 'results/nn_model_evaluation.txt'
    
    if not check_file_exists(ml_eval_path):
        print("Traditional ML model evaluation not found. Please run model training first.")
        return
    
    # Read and display traditional ML results
    print("\nTraditional ML Models:")
    with open(ml_eval_path, 'r') as f:
        ml_eval = f.read()
        print(ml_eval)
    
    # Read and display neural network results if available
    if check_file_exists(nn_eval_path):
        print("\nNeural Network Models:")
        with open(nn_eval_path, 'r') as f:
            nn_eval = f.read()
            print(nn_eval)
    else:
        print("\nNeural network model evaluation not found. Please run neural network training first.")

if __name__ == "__main__":
    main()