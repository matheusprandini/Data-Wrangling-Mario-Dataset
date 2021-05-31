import os
import cv2
import glob
import numpy as np

from feature_extractor.factory.FeatureExtractorFactory import FeatureExtractorFactory
from utils.Utils import Utils
from utils.CsvHandler import CsvHandler


class MarioDatasetController():

    def __init__(self, inputDataset, outputDataset, classes, chunkSize, imageSize, featureExtractorName):
        self.inputDataset = inputDataset # Input directory
        self.outputDataset = outputDataset # Output directory
        self.classes = classes # Classes to extract information
        self.chunkSize = chunkSize # Chunk size
        self.imageSize = imageSize # Image size
        self.featureExtractorModel = FeatureExtractorFactory.get_model(featureExtractorName) # Feature extractor model
        self.classChunkCounter = {} # Count to save chunk files

    def create_data(self):
        # Get all sub directories in the "self.inputDataset" directory
        subDirs = glob.glob(self.inputDataset  + "*")

        # Get all files in the "subDirs" directory
        allFiles = []
        for subDir in subDirs:
            allFiles.extend(Utils.find_all_csv_files(subDir))

        for className in self.classes:
            # Variable to Save Path
            savePath = self.outputDataset + className + "/"
            os.makedirs(savePath, exist_ok=True)

            # Start Counter
            self.classChunkCounter[className] = 0

        ## Processing each csv file
        for marioFile in allFiles:
            print("Processing", marioFile, "file...")
            ## Get dataframe with all activated events
            marioDataframe = CsvHandler(marioFile)
            allClassesDataframe = marioDataframe.get_columns(self.classes)
            allClassesEventsDataframe = allClassesDataframe[(allClassesDataframe == 1).any(axis=1)]
            oneClassesDataframe = {}

            if not allClassesEventsDataframe.empty:
                ## Removing rows with two or more activated events
                for row in allClassesEventsDataframe.iterrows():
                    index = row[0]
                    values = row[1]
                    if np.count_nonzero(values == 1) > 1:
                        continue
                    oneClassesDataframe[str(index)] = values

                indexes = [int(index) for index in list(oneClassesDataframe.keys())]

                ## Processing each activated event
                for index in indexes:
                    chunkIndexes = self.get_chunk_indexes_mean(index)
                    className = oneClassesDataframe[str(index)].idxmax(axis=1)
                    filename = marioFile.split('/')[-1]
                    imgPath = marioFile.split(filename)[0] + "Images/Mario"
                    frames = []
                    for chunkIndex in chunkIndexes:
                        if chunkIndex <= len(allClassesDataframe):
                            framePath = imgPath + str(chunkIndex) + ".png"
                            frame = cv2.imread(framePath)
                            frame = cv2.resize(frame, self.imageSize)
                            frame = frame / 255.0
                            frames.append(frame)
                    
                    savePath = self.outputDataset + className + "/"
                    os.makedirs(savePath, exist_ok=True)
                    self.process_chunk_data(savePath, frames, className)

                    # Update Chunk count
                    self.classChunkCounter[className] += 1
                    
    def process_chunk_data(self, path, frames, className):
        if className not in self.classChunkCounter:
            self.classChunkCounter[className] = 0
        chunkCount = self.classChunkCounter[className]

        for i in range(0, len(frames), self.chunkSize):

            # Filename
            if chunkCount < 10:
                chunkName = "chunk_" + "0" + str(chunkCount)
            else:
                chunkName = "chunk_" + str(chunkCount)

            # Get data (frames)
            data = frames[i:i+self.chunkSize]

            # Checks if the number of frames is less than the chunk size. 
            # If so, normalizes chunk with the last frame.
            if len(data) < self.chunkSize:
                missingFrames = self.chunkSize - len(data)
                for i in range(missingFrames):
                    data.append(data[-1])

            ## Extract features
            data = np.array(data)
            if self.featureExtractorModel:
                preprocessedData = self.featureExtractorModel.predict(data)
            else:
                preprocessedData = data
            
            # Save preprocessed data
            np.save(path + chunkName, preprocessedData)

    def get_chunk_indexes_mean(self, index):
        initialPos = int(index - (self.chunkSize / 2) + 1)
        finalPos = int(index + (self.chunkSize / 2) + 1)
        return list(range(initialPos, finalPos))
