import sys

from services.MarioDatasetController import MarioDatasetController
from utils.JsonHandler import JsonHandler
from os import path

sys.path.append(path.join(path.dirname(__file__), '..'))


def main():

    configFile = JsonHandler.read_json("../conf/config.json")

    inputDir = configFile["datasetInfo"]["inputDataset"]
    outputDir = configFile["datasetInfo"]["outputDataset"]
    classesList = configFile["datasetInfo"]["classes"]
    chunkSize = int(configFile["datasetInfo"]["chunkSize"])
    imageSize = tuple(
        (configFile["datasetInfo"]["imageSize"], configFile["datasetInfo"]["imageSize"]))
    featureExtractorName = configFile["featureExtractor"]["name"]

    marioDataset = MarioDatasetController(
        inputDir, outputDir, classesList, chunkSize, imageSize, featureExtractorName)
    marioDataset.create_data()


main()
