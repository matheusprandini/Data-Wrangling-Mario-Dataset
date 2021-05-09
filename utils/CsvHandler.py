import pandas as pd

class CsvHandler():

    def __init__(self, path):
        self.dataframe = self.read_file(path)

    def read_file(self, path):
        dataframe = pd.read_csv(path)
        dataframe.drop("Unnamed: 275", axis=1, inplace=True)
        return dataframe

    def get_columns(self, columnsList):
        return self.dataframe[columnsList]

    def show_stats(self):
        print(self.dataframe.describe())