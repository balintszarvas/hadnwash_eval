import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

# Add ML4QS Python3 modules to path
sys.path.insert(0, 'ML4QS/Python3Code')

from ML4QS.Python3Code.util.VisualizeDataset import VisualizeDataset
from ML4QS.Python3Code.Chapter7.LearningAlgorithms import ClassificationAlgorithms
from ML4QS.Python3Code.Chapter7.Evaluation import ClassificationEvaluation


def main():
    train_path = '/Users/balints/Documents/CLS/MLQS/hadnwash_eval/aggregated_data/merged_features_5s.csv'
    test_path = '/Users/balints/Documents/CLS/MLQS/hadnwash_eval/aggregated_data/test_merged_features_5s.csv'

    print(f'Loading train file {train_path} …')
    df_train = pd.read_csv(train_path)
    print(f'Train shape: {df_train.shape}')

    print(f'Loading test file {test_path} …')
    df_test = pd.read_csv(test_path)
    print(f'Test shape: {df_test.shape}')

    # Basic sanity check
    if 'score' not in df_train.columns or 'score' not in df_test.columns:
        raise ValueError('Target column "score" not found in one of the datasets')

    feature_cols = [c for c in df_train.columns if c not in ('score', 'datetime')]

    y_train = df_train['score'].astype(str)
    X_train = df_train[feature_cols]

    y_test = df_test['score'].astype(str)
    X_test = df_test[feature_cols]

    # Impute missing values based on training medians
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

    viz = VisualizeDataset(__file__)

    # Use ML4QS helper RandomForest (no grid search for speed)
    learner = ClassificationAlgorithms()
    train_pred, test_pred, _, _, rf_model = learner.random_forest(
        X_train, y_train, X_test,
        n_estimators=500,
        min_samples_leaf=2,
        gridsearch=True,
        print_model_details=True,  # prints feature importances in console
        return_model=True
    )

    evaluator = ClassificationEvaluation()
    acc_train = evaluator.accuracy(y_train, train_pred)
    acc_test = evaluator.accuracy(y_test, test_pred)

    print(f'Random-Forest accuracy  train: {acc_train:.4f}  test: {acc_test:.4f}')

    # Confusion matrix plot
    cm = confusion_matrix(y_test, test_pred, labels=sorted(y_test.unique()))
    viz.plot_confusion_matrix(cm, classes=sorted(y_test.unique()), normalize=False)

    # ----------------------------------------------------------
    # Plot top 30 feature importances
    # ----------------------------------------------------------
    import matplotlib.pyplot as plt
    import numpy as np

    importances = rf_model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1][:30]
    top_features = [feature_cols[i] for i in idx_sorted]
    top_importances = importances[idx_sorted]

    plt.figure(figsize=(8, 10))
    plt.barh(range(len(top_features))[::-1], top_importances, align='center')
    plt.yticks(range(len(top_features))[::-1], top_features)
    plt.xlabel('Importance')
    plt.title('Random Forest – Top 30 Feature Importances', loc='left')
    plt.tight_layout()

    fig_dir = 'figures/train_random_forest'
    import os; os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'feature_importances_top30.png'))
    plt.savefig(os.path.join(fig_dir, 'feature_importances_top30.pdf'))
    plt.close()
    print('Feature-importance plot saved to', fig_dir)


if __name__ == '__main__':
    main()