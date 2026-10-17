import numpy as np
import pandas as pd

_SOURCE = pd.read_csv("data/telco_churn.csv")


def new_customers(n=5):
    """Create n synthetic 'incoming' customers by remixing real records."""
    df = _SOURCE.sample(n).drop(columns="Churn").reset_index(drop=True)
    df["customerID"] = ["SIM-" + str(np.random.randint(10000, 99999))
                        for _ in range(n)]
    df["tenure"] = (df["tenure"] + np.random.randint(-3, 4, n)).clip(0, 72)
    df["MonthlyCharges"] = (df["MonthlyCharges"]
                            * np.random.uniform(0.9, 1.1, n)).round(2).clip(18, 120)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    return df