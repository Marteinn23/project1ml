from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

class PHISURL_NeuralNetwork(MLPClassifier):
    def __init__(self, hidden_layer_sizes=(50,), activation="relu", 
                 alpha=0.0001, max_iter=200, random_state=42):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state
        )

class PHISURL_NaiveBayes(BernoulliNB):
    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)

class PHISURL_RandomForest(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )