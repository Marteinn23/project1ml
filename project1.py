import pandas as pd

data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

data = data.drop(columns=["FILENAME","URLLength","DomainLength",""])

print(data.keys())
