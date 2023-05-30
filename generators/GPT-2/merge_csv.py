import pandas as pd

abstract = pd.read_csv("gpt2_from_title_to_abstract.csv")
introducion = pd.read_csv("gpt2_from_title_to_introduction.csv")
conclusion = pd.read_csv("gpt2_from_title_to_conclusion.csv")

merged = abstract.merge(introducion, on="title")
merged = merged.merge(conclusion, on="title")
merged.to_csv("GPT2_output.csv", index=False)
