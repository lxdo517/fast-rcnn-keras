from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model


def get_model_body(trainable=False):
    vgg16_model = VGG16(include_top=False)
    if not trainable:
        for layer in vgg16_model.layers:
            # 让其不可训练
            layer.trainable = False
    vgg16_model.layers.pop()
    model = Model(inputs=vgg16_model.input, outputs=vgg16_model.output)
    return model
