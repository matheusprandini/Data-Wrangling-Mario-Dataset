import keras as K
from keras.applications.mobilenet_v2 import MobileNetV2

class MobileNetV2Model():

    def build_model(self, weights, includeTop, inputShape):
        baseModel = MobileNetV2(weights=weights, include_top=includeTop, input_shape=inputShape)

        model = K.models.Model(
            inputs=baseModel.input,
            outputs=baseModel.get_layer('global_average_pooling2d').output
        )
        
        return model