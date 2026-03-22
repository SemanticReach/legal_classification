import pandas as pd

df = pd.read_csv("legal_eval_results.csv")
failures = df[
    (df.clause_type == "Cap On Liability") & 
    (df.predicted_type == "Uncapped Liability")
]
print(failures[["contract","text_preview"]].to_string())