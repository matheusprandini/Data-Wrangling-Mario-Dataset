import keras as K
from keras.applications.resnet_v2 import ResNet50V2

class ResNet50V2Model():

    def build_model(self, weights, includeTop, inputShape):
        baseModel = ResNet50V2(weights=weights, include_top=includeTop, input_shape=inputShape)

        model = K.models.Model(
            inputs=baseModel.input,
            outputs=baseModel.get_layer('avg_pool').output
        )
        
        return model