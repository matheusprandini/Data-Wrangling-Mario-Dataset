from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50V2


class ResNet50V2Model():

    def build_model(self, weights, includeTop, inputShape):
        baseModel = ResNet50V2(weights=weights, include_top=includeTop, input_shape=inputShape)

        model = Model(
            inputs=baseModel.input,
            outputs=baseModel.get_layer('avg_pool').output
        )
        
        return model