from ..models.MobileNetV2 import MobileNetV2Model
from ..models.ResNet50V2Model import ResNet50V2Model
from ..models.VGG16Model import VGG16Model

class FeatureExtractorFactory():
    modelNameToFeatureExtractorModel = {
        "MobileNetV2": MobileNetV2Model(),
        "ResNet50V2": ResNet50V2Model(),
        "VGG16": VGG16Model()
    }

    @staticmethod
    def get_model(name, weights="imagenet", includeTop=True, inputShape=(224,224,3)):
        if name not in FeatureExtractorFactory.modelNameToFeatureExtractorModel:
            return None
        return FeatureExtractorFactory.modelNameToFeatureExtractorModel[name].build_model(weights, includeTop, inputShape)