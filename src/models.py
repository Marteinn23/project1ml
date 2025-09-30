from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

# Simple wrappers around the base models we'll be using
class PHISURL_NeuralNetwork(MLPClassifier):
    def __init__(
            self, 
            hidden_layer_sizes=(50,), 
            activation="relu", 
            alpha=0.0001,
            learning_rate_init=0.001,
        ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            random_state=42
        )

class PHISURL_NaiveBayes(BernoulliNB):
    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)

class PHISURL_RandomForest(RandomForestClassifier):
    def __init__(
            self, 
            n_estimators=100, 
            max_depth=None, 
            min_samples_split=2,
            min_samples_leaf=2,
        ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
