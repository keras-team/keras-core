class NumpyTrainer:
    def __init__(self, **kwargs):
        raise NotImplementedError("Numpy Backend should not have a trainer")

    def fit(self):
        raise NotImplementedError("Numpy Backend should not have a trainer")

    def predict(self):
        raise NotImplementedError("Numpy Backend should not have a trainer")

    def evaluate(self):
        raise NotImplementedError("Numpy Backend should not have a trainer")

    def train_on_batch(self):
        raise NotImplementedError("Numpy Backend should not have a trainer")

    def test_on_batch(self):
        raise NotImplementedError("Numpy Backend should not have a trainer")

    def predict_on_batch(self):
        raise NotImplementedError("Numpy Backend should not have a trainer")
