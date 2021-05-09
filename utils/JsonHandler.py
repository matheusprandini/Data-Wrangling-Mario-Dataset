import json

class JsonHandler(object):

    @staticmethod
    def read_json(jsonFilePath):
        with open(jsonFilePath) as jsonFile:    
            return json.load(jsonFile)