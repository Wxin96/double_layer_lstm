from keras.models import Sequential, load_model


class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def build_model(self):
        self.model

    def train(self, x, y, epochs, batch_size, save_dir):
        pass



