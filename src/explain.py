import pandas as pd
import shap


def get_drivers(model, row, top_n=6):
    """Return the top factors behind one customer's churn score."""
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(row)[0]
    s = pd.Series(vals, index=row.columns)
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)
    total = s.abs().sum()
    return pd.DataFrame({
        "Factor": [f.replace("_", ": ") for f in top.index],
        "Share of impact (%)": (top.abs() / total * 100).round(1).values,
        "Effect": ["Increases risk" if v > 0 else "Reduces risk" for v in top.values],
    })