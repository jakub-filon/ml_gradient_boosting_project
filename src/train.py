import pandas as pd
import numpy as np
import optuna
import os
import joblib
import json
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

# Configuration
N_TRIALS = 200
N_WARMUP_TRIALS = 40
RANDOM_STATE = 42

def load_data(data_dir):
    """Loads processed training data."""
    print("Loading data...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    return X_train, y_train

def evaluate_metrics(y_true, y_pred, y_prob):
    """Calculates MCC, F1, and PR-AUC for a given set of predictions."""
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    return mcc, f1, pr_auc

def objective_xgboost(trial, X, y):
    """
    Optuna objective function for XGBoost.
    Optimizes hyperparameters using Stratified K-Fold CV.
    
    Note: 'scale_pos_weight' is set to 4 to handle class imbalance (approx 20% positive class).
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
        'gamma': trial.suggest_float('gamma', 0, 5, step=0.5),
        'scale_pos_weight': 4,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mcc_scores = []
    f1_scores = []
    pr_auc_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = XGBClassifier(**param)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        m, f, p = evaluate_metrics(y_val, preds, probs)
        mcc_scores.append(m)
        f1_scores.append(f)
        pr_auc_scores.append(p)
        
    return np.mean(mcc_scores), np.mean(f1_scores), np.mean(pr_auc_scores)

def objective_lightgbm(trial, X, y):
    """
    Optuna objective function for LightGBM.
    'scale_pos_weight': 4 used for imbalance.
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, step=0.01),
        'num_leaves': trial.suggest_int('num_leaves', 15, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
        'scale_pos_weight': 4,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mcc_scores = []
    f1_scores = []
    pr_auc_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = LGBMClassifier(**param)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        m, f, p = evaluate_metrics(y_val, preds, probs)
        mcc_scores.append(m)
        f1_scores.append(f)
        pr_auc_scores.append(p)
        
    return np.mean(mcc_scores), np.mean(f1_scores), np.mean(pr_auc_scores)

def objective_catboost(trial, X, y):
    """
    Optuna objective function for CatBoost.
    """
    param = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'depth': trial.suggest_int('depth', 3, 10),
        # 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, step=0.01),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, step=0.5),
        'scale_pos_weight': 4,
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'allow_writing_files': False
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mcc_scores = []
    f1_scores = []
    pr_auc_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = CatBoostClassifier(**param)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        m, f, p = evaluate_metrics(y_val, preds, probs)
        mcc_scores.append(m)
        f1_scores.append(f)
        pr_auc_scores.append(p)
        
    return np.mean(mcc_scores), np.mean(f1_scores), np.mean(pr_auc_scores)

def optimize_and_save(objective_func, study_name, db_path, X, y, model_cls, model_save_path, n_warmup_steps):
    """
    Executes the Optuna optimization study and saves the best model.
    """
    print(f"\nStarting {study_name}...")
    
    storage = f"sqlite:///{db_path}"
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_warmup_steps, multivariate=True, seed=RANDOM_STATE)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=['maximize', 'maximize', 'maximize'], # MCC, F1, PR-AUC
        sampler=sampler,
        load_if_exists=False
    )
    
    study.set_metric_names(['MCC', 'F1', 'PR-AUC'])
    
    study.optimize(lambda trial: objective_func(trial, X, y), n_trials=N_TRIALS)
    
    # Select best trial based on MCC (index 0) from the Pareto front
    # We prefer MCC as it handles class imbalance metrics well.
    best_trial = max(study.best_trials, key=lambda t: t.values[0])

    print(f"{study_name} Best Trial Metrics (Chosen by MCC form Pareto):")
    print(f"  MCC: {best_trial.values[0]:.4f}")
    print(f"  F1: {best_trial.values[1]:.4f}")
    print(f"  PR-AUC: {best_trial.values[2]:.4f}")
    
    final_params = best_trial.params.copy()
    
    # Specific fixed params from objectives needed for final model reproduction
    if 'xgboost' in study_name or 'lightgbm' in study_name or 'catboost' in study_name:
        final_params['scale_pos_weight'] = 4
        
    if 'xgboost' in study_name:
        final_params['n_jobs'] = -1
        final_params['random_state'] = RANDOM_STATE
    elif 'lightgbm' in study_name:
        final_params['n_jobs'] = -1
        final_params['random_state'] = RANDOM_STATE
        final_params['verbose'] = -1
    elif 'catboost' in study_name:
        final_params['random_seed'] = RANDOM_STATE
        final_params['verbose'] = 0
        final_params['allow_writing_files'] = False

    best_model = model_cls(**final_params)
    best_model.fit(X, y)
    joblib.dump(best_model, model_save_path)
    
    return best_trial.params

def train_and_optimize():
    """Main training orchestration function."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    X_train, y_train = load_data(data_dir)
    studies = {}
    
    common_db_path = os.path.join(models_dir, 'optuna_studies.db')
    
    # 1. XGBoost
    studies['xgboost'] = optimize_and_save(
        objective_xgboost, 
        "xgboost_study", 
        common_db_path,
        X_train, y_train, 
        XGBClassifier, 
        os.path.join(models_dir, 'xgboost_best.pkl'),
        n_warmup_steps=N_WARMUP_TRIALS
    )
    
    # 2. LightGBM
    studies['lightgbm'] = optimize_and_save(
        objective_lightgbm, 
        "lightgbm_study", 
        common_db_path,
        X_train, y_train, 
        LGBMClassifier, 
        os.path.join(models_dir, 'lightgbm_best.pkl'),
        n_warmup_steps=N_WARMUP_TRIALS
    )
    
    # 3. CatBoost
    studies['catboost'] = optimize_and_save(
        objective_catboost, 
        "catboost_study", 
        common_db_path,
        X_train, y_train, 
        CatBoostClassifier, 
        os.path.join(models_dir, 'catboost_best.pkl'),
        n_warmup_steps=N_WARMUP_TRIALS
    )
    
    # Save best parameters (from the best trial)
    with open(os.path.join(models_dir, 'best_params.json'), 'w') as f:
        json.dump(studies, f, indent=4)
        
    print(f"\nTraining Complete. Models and Studies saved to {models_dir}")

if __name__ == "__main__":
    train_and_optimize()
