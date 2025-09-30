"""
Authors: Marteinn, Teitur, Tryggvi
"""

import pandas as pd

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
    new_data_set = pd.read_csv("dataset/new_data_urls.csv")
    data = make_dict(new_data_set)
    data = is_domain_ip(data)
    data = IsHTTPS_Process(data)
    data = TLD_process(data)
    data = No_of_digits_equal_qmark_amp(data)
    data = no_of_sub_domain(data)

    # Convert dictionary to CSV
    csv_filename = "dataset/proccessed_urls.csv"
    dict_to_csv(data, csv_filename)


def make_dict(new_data_set: dict):
    """Creates the dictionary we're going to be turning into a csv with. It's a dictionary of dictionaries with the url as the key"""
    data = {}
    url_list = new_data_set["url"].copy()
    for i, url in enumerate(url_list):
        url = str(url)
        if url[0] == '"' or url[0] == "'":
            url_list[i] = url[1:]
        if url[-1] == '"' or url[-1] == "'":
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
            data[url]["IsHTTPS"] = 2  # unsure
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
    for url in data.keys():
        tld_check = tldextract.extract(url)
        if tld_check.subdomain != "":
            data[url]["NoOfSubDomain"] = len(tld_check.subdomain.split("."))
    return data


def is_valid_ip(ip_string):
    try:
        ipaddress.ip_address(ip_string)
        return True
    except ValueError:
        return False


def dict_to_csv(data: dict, filename: str):
    records = []
    for url, features in data.items():
        record = {"URL": url}
        record.update(features)
        records.append(record)

    df = pd.DataFrame(records)

    columns = ["URL"] + [col for col in df.columns if col != "URL" and col != "Label"]
    if "Label" in df.columns:
        columns.append("Label")

    df = df[columns]
    df.to_csv(filename, index=False)

    return df


if __name__ == "__main__":
    main()
