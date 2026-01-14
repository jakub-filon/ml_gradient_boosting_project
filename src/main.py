"""
Main execution script for the Churn Prediction Pipeline.
Orchestrates the sequence of:
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing & Feature Engineering
3. Model Training & Hyperparameter Optimization
4. Model Evaluation
"""
import os
import subprocess
import sys

def run_script(script_name):
    """
    Executes a Python script as a subprocess.
    Stops execution if the script fails.
    """
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n{'='*20} Running {script_name} {'='*20}")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"Error running {script_name}")
        sys.exit(result.returncode)

def main():
    """
    Runs the full ML pipeline steps sequentially.
    """
    # 1. EDA
    run_script('eda.py')
    
    # 2. Preprocessing
    run_script('preprocess.py')
    
    # 3. Training
    run_script('train.py')
    
    # 4. Evaluation
    run_script('evaluate.py')
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
