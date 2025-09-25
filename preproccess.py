"""
Authors: Marteinn, Teitur, Tryggvi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

## classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def main():
    new_data_set = pd.read_csv("dataset/new_data_urls.csv")

    data = make_dict(new_data_set)
    data = IsHTTPS_Process(data)
    data = TLD_process(data)
    print(data)


def make_dict(new_data_set: dict):
    """Creates the dictionary we're going to be turning into a csv with. It's a dictionary of dictionaries with the url as the key"""
    data = {}
    url_list = new_data_set["url"].copy()
    for i, url in enumerate(url_list):
        url = str(url)
        if url[0] == '"':
            print(url)
            url_list[i] = url[1:]
        if url[-1] == '"':
            url = url[:-1]
        url_list[i] = url

        url_data = {
            "TLD": None,
            "TLDLength": None,
            "URLLength": None,
            "IsDomainIP": None,
            "NoOfSubDomain": None,
            "IsHTTPS": None,
            "NoOfDegitsInURL": None,
            "NoOfEqualsInURL": None,
            "NoOfQMarkInURL": None,
            "NoOfAmpersandInURL": None,
            "NoOfOtherSpecialCharsInURL": None,
            "Label": None,
        }
        data[url] = url_data
        data[url]["Label"] = new_data_set["status"][i]
        data[url]["URLLength"] = len(url)
    return data


def IsHTTPS_Process(data: dict):
    """Checks whether a URL has HTTP, HTTPS or is missing that data. Stores the info into the data dictionary."""
    for url in data.keys():
        check = url.lower()
        if check.startswith("https"):
            data[url]["IsHTTPS"] = 1
        elif check.startswith("http"):
            data[url]["IsHTTPS"] = 0
        else:
            data[url]["IsHTTPS"] = 2
    return data


def TLD_process(data: dict):
    """Checks the Top Level Domain (TLD), stores that and the length of the TLD into our dictionary."""
    for url in data.keys():
        check = url.split(".")
        data[url]["TLD"] = check[-1]
        data[url]["TLDLength"] = len(check[-1])
    return data


if __name__ == "__main__":
    main()
