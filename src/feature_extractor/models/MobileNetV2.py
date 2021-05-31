from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2


class MobileNetV2Model():

    def build_model(self, weights, includeTop, inputShape):
        baseModel = MobileNetV2(weights=weights, include_top=includeTop, input_shape=inputShape)

        model = Model(
            inputs=baseModel.input,
            outputs=baseModel.get_layer('global_average_pooling2d').output
        )
        
        return model