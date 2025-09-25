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

## preprocessing
import tldextract
import ipaddress


def main():
    # # Load the CSV file
    # df = pd.read_csv("dataset\PhiUSIIL_Phishing_URL_Dataset.csv")

    # # Print URL and number of special characters
    # for _, row in df.iterrows():
    #     print(row["URL"], row["NoOfSubDomain"])

    new_data_set = pd.read_csv("dataset/new_data_urls.csv")
    data = make_dict(new_data_set)
    data = is_domain_ip(data)
    data = IsHTTPS_Process(data)
    data = TLD_process(data)
    data = No_of_digits_equal_qmark_amp(data)
    data = no_of_sub_domain(data)

    print_data(data)


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
            "DomainLength": None,
            "TLD": None,
            "TLDLength": None,
            "URLLength": None,
            "IsDomainIP": None,
            "NoOfSubDomain": 0,
            "IsHTTPS": None,
            "NoOfDegitsInURL": 0,
            "NoOfEqualsInURL": 0,
            "NoOfQMarkInURL": 0,
            "NoOfAmpersandInURL": 0,
            "NoOfOtherSpecialCharsInURL": 0,
            "Label": None,
        }
        data[url] = url_data
        data[url]["Label"] = new_data_set["status"][i]
        data[url]["URLLength"] = len(url)
        data[url]["DomainLength"] = len(tldextract.extract(url).domain)
    return data


def is_domain_ip(data: dict):
    """Takes in the domain of the url from the TLDExtract library and parses them through the IP address library to check if the domain is an IP."""
    for url in data.keys():
        tld_check = tldextract.extract(url)
        if is_valid_ip(tld_check.domain):
            data[url]["IsDomainIP"] = 1
        else:
            data[url]["IsDomainIP"] = 0
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
    """Checks the Top Level Domain (TLD) by using the TLDExtract library, taking the suffix and length."""
    for url in data.keys():
        check = tldextract.extract(url)
        tld = check.suffix
        data[url]["TLD"] = tld
        data[url]["TLDLength"] = len(tld)
    return data


def No_of_digits_equal_qmark_amp(data: dict):
    """Counts the number of digits, equal signs, question marks, ampersand and other special characters in the URL."""
    for url in data.keys():
        url = str(url)
        for char in url:
            if char.isdigit():
                data[url]["NoOfDegitsInURL"] += 1
            elif char == "?":
                data[url]["NoOfQMarkInURL"] += 1
            elif char == "&":
                data[url]["NoOfAmpersandInURL"] += 1
            elif char == "=":
                data[url]["NoOfEqualsInURL"] += 1
            else:
                data[url]["NoOfOtherSpecialCharsInURL"] += 1
    return data


def no_of_sub_domain(data: dict):
    """Counts the number of sub domains by getting the subdomain string from TLDExtract, splitting the string by . and counting the length."""
    index = 0
    for url in data.keys():
        tld_check = tldextract.extract(url)
        if tld_check.subdomain != "":
            data[url]["NoOfSubDomain"] = len(tld_check.subdomain.split("."))
            if index == 0:
                print(url)
                print(tld_check.subdomain)
                print(data[url]["NoOfSubDomain"])
                index = 1
    return data


def print_data(data: dict):
    # Clear the file and optionally write new content
    with open("print.txt", "w") as file:
        for url in data.keys():
            file.write("\n")
            file.write(f"{url}: ")
            for values in data[url].keys():
                file.write(f"      {values}: {data[url][values]}")
        file.write("\n")


def is_valid_ip(ip_string):
    """
    Checks if a given string is a valid IPv4 or IPv6 address.

    Args:
        ip_string: The string to validate.

    Returns:
        True if the string is a valid IP address, False otherwise.
    """
    try:
        ipaddress.ip_address(ip_string)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
