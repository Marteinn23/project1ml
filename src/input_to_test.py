import tldextract
import ipaddress
import pandas as pd


class URLPreprocessor:
    def __init__(self):
        self.feature_names = [
            "DomainLength",
            "TLD",
            "TLDLength",
            "URLLength",
            "IsDomainIP",
            "NoOfSubDomain",
            "IsHTTPS",
            "NoOfDegitsInURL",
            "NoOfEqualsInURL",
            "NoOfQMarkInURL",
            "NoOfAmpersandInURL",
            "NoOfOtherSpecialCharsInURL",
        ]

    def preprocess_url(self, url: str) -> dict:
        url = str(url).strip()
        if url.startswith('"'):
            url = url[1:]
        if url.endswith('"'):
            url = url[:-1]

        features = {
            "URL": url,
            "DomainLength": 0,
            "TLD": "",
            "TLDLength": 0,
            "URLLength": 0,
            "IsDomainIP": 0,
            "NoOfSubDomain": 0,
            "IsHTTPS": 2,
            "NoOfDegitsInURL": 0,
            "NoOfEqualsInURL": 0,
            "NoOfQMarkInURL": 0,
            "NoOfAmpersandInURL": 0,
            "NoOfOtherSpecialCharsInURL": 0,
        }

        features["URLLength"] = len(url)

        tld_info = tldextract.extract(url)
        features["DomainLength"] = len(tld_info.domain)
        features["TLD"] = tld_info.suffix
        features["TLDLength"] = len(tld_info.suffix)

        features["IsDomainIP"] = 1 if self._is_valid_ip(tld_info.domain) else 0

        if tld_info.subdomain:
            features["NoOfSubDomain"] = len(tld_info.subdomain.split("."))

        url_lower = url.lower()
        if url_lower.startswith("https"):
            features["IsHTTPS"] = 1
        elif url_lower.startswith("http"):
            features["IsHTTPS"] = 0

        for char in url:
            if char.isdigit():
                features["NoOfDegitsInURL"] += 1
            elif char == "?":
                features["NoOfQMarkInURL"] += 1
            elif char == "&":
                features["NoOfAmpersandInURL"] += 1
            elif char == "=":
                features["NoOfEqualsInURL"] += 1
            else:
                features["NoOfOtherSpecialCharsInURL"] += 1

        return features

    def preprocess_for_model(self, url: str) -> pd.DataFrame:
        features = self.preprocess_url(url)
        model_features = {feature: features[feature] for feature in self.feature_names}
        df = pd.DataFrame([model_features])
        return df

    def get_feature_names(self) -> list:
        return self.feature_names.copy()

    def _is_valid_ip(self, ip_string):
        try:
            ipaddress.ip_address(ip_string)
            return True
        except ValueError:
            return False


def preprocess_single_url(url: str) -> pd.DataFrame:
    preprocessor = URLPreprocessor()
    return preprocessor.preprocess_for_model(url)
