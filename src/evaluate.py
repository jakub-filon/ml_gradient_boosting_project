import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, confusion_matrix, matthews_corrcoef, average_precision_score

def load_data(data_dir):
    """Loads transformed test data."""
    print("Loading test data...")
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    return X_test, y_test

def load_models(models_dir):
    """Loads trained models from individual pickel files."""
    models = {}
    for name in ['xgboost', 'lightgbm', 'catboost']:
        path = os.path.join(models_dir, f'{name}_best.pkl')
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

def evaluate_models():
    """
    Evaluates trained models on the test set.
    
    Metrics:
    - ROC-AUC: Area under Reciever Operating Characteristic curve. Good for general ranking quality.
    - MCC: Matthews Correlation Coefficient. Excellent summary metric for imbalanced classes.
    - PR-AUC: Area under Precision-Recall curve. Focuses on positive class performance.
    - Accuracy: Simple correct/total ratio. (Can be misleading for imbalanced data).
    
    Generates:
    - ROC Curves plot
    - Confusion Matrix heatmaps
    - Feature Importance plots
    - CSV Summary of metrics
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    output_dir = os.path.join(base_dir, 'plots', 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    X_test, y_test = load_data(data_dir)
    models = load_models(models_dir)
    
    results = []
    
    # Setup ROC plot
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_prob)
        
        print(f"{name} Results:")
        print(f"AUC: {auc:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        results.append({
            'Model': name,
            'AUC': auc,
            'Accuracy': acc,
            'MCC': mcc,
            'PR-AUC': pr_auc
        })
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        # Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name}.png'))
        plt.close()
        
        # Feature Importance
        plt.figure(figsize=(10, 8))
        if name == 'xgboost':
            importances = model.feature_importances_
            feature_names = X_test.columns
        elif name == 'lightgbm':
            importances = model.feature_importances_
            feature_names = model.feature_name_
        elif name == 'catboost':
            importances = model.feature_importances_
            feature_names = model.feature_names_
            
        # Sort
        indices = np.argsort(importances)[::-1]
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
        plt.title(f'Feature Importances - {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_importance_{name}.png'))
        plt.close()

    # Finalize ROC Plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    
    # Save metrics summary
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    evaluate_models()
