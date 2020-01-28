from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


def get_model_body(trainable=False):
    input_tensor = Input(shape=(567, 567, 3))
    vgg16_model = VGG16(input_tensor=input_tensor, include_top=False)
    if not trainable:
        for layer in vgg16_model.layers:
            # 让其不可训练
            layer.trainable = False
    model = Model(inputs=vgg16_model.input, outputs=vgg16_model.get_layer('block5_conv3').output)
    # model = Model(inputs=vgg16_model.input, outputs=vgg16_model.output)
    return model

#
# model = get_model_body()
# model.summary()