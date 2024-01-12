class Vectorizer(object):
    def __init__(self):
        pass
    @abstractmethod
    def vectorize_data(self, data):
        pass
    @abstractmethod
    def vectorize_dataset(self, dataset):
        pass

    def callback_vectorize_data(self, data):
        try:
            self.vectorize_data(data)
        except:
            logging.error("callback vectorize data error: " + str(traceback.format_exc()))

    def callback_vectorize_dataset(self, dataset):
        try:
            self.vectorize_dataset(dataset)
        except:
            logging.error("callback vectorize dataset error: " + str(traceback.format_exc()))
