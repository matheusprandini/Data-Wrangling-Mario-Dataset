from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16


class VGG16Model():

    def build_model(self, weights, includeTop, inputShape):
        baseModel = VGG16(weights=weights, include_top=includeTop, input_shape=inputShape)

        model = Model(
            inputs=baseModel.input,
            outputs=baseModel.get_layer('fc2').output
        )
        
        return model
