import os
import zipfile
from pathlib import Path

class Utils():

    @staticmethod
    def unzip_file(zippedFilePath):
        with zipfile.ZipFile(zippedFilePath, 'r') as zippedFile:
            zippedFile.extractall()

    @staticmethod
    def get_file_name(file):
        name = os.path.splitext(file)[0]
        return name

    @staticmethod
    def find_all_csv_files(path):
        files = []
        for path in Path(path).rglob('*.csv'):
            filePath = str(path) 
            files.append(filePath)
        return files

    @staticmethod
    def find_all_subfolders_name(csvFiles):
        subfoldersName = set()
        for csvFilename in csvFiles:
            subfolderName = csvFilename.split("\\")[1]
            subfoldersName.add(subfolderName)
        return subfoldersName