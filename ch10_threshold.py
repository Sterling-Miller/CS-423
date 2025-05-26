import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def threshold_results(thresh_list, actuals, predicted):
    """
    Computes classification metrics for a range of probability thresholds and returns results as DataFrames.

    This function evaluates a binary classifier's performance across multiple thresholds by:
    - Converting predicted probabilities to binary predictions at each threshold.
    - Calculating precision, recall, F1-score, accuracy, and AUC for each threshold.
    - Returning both a plain DataFrame and a styled DataFrame for display.

    Parameters
    ----------
    thresh_list : list or array-like
        List of thresholds to evaluate (e.g., [0.3, 0.4, 0.5, 0.6]).
    actuals : array-like
        True binary labels (0 or 1).
    predicted : array-like
        Predicted probabilities or scores from a classifier.

    Returns
    -------
    result_df : pd.DataFrame
        DataFrame with columns ['threshold', 'precision', 'recall', 'f1', 'accuracy', 'auc'] for each threshold.
    fancy_df : pd.io.formats.style.Styler
        Styled DataFrame for display in Jupyter notebooks, highlighting the best metric values.

    Examples
    --------
    >>> result_df, fancy_df = threshold_results([0.4, 0.5, 0.6], y_true, y_pred_proba)
    >>> print(result_df)
       threshold  precision  recall    f1  accuracy   auc
    0        0.4       0.80    0.85  0.82      0.81  0.90
    1        0.5       0.78    0.80  0.79      0.79  0.90
    2        0.6       0.75    0.70  0.72      0.76  0.90
    """
    result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy', 'auc'])

    for t in thresh_list:
        yhat = [1 if v >= t else 0 for v in predicted]
        precision = precision_score(actuals, yhat, zero_division=0)
        recall = recall_score(actuals, yhat, zero_division=0)
        f1 = f1_score(actuals, yhat)
        accuracy = accuracy_score(actuals, yhat)
        auc = roc_auc_score(actuals, predicted)
        result_df.loc[len(result_df)] = {
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc': auc
        }

    result_df = result_df.round(2)

    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #800000; color: white; text-align: center"
    }
    properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

    fancy_df = result_df.style.highlight_max(color='pink', axis=0).format(precision=2).set_properties(**properties).set_table_styles([headers])
    return (result_df, fancy_df)
