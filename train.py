from voc_annotation import VOCAnnotation
from voc_data import VocData
from models.model import FastRCNN
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD
import config as cfg

if __name__ == '__main__':
    log_dir = 'logs/000/'

    voc_train_annotation = VOCAnnotation(2007, 'train', '/Users/lx/segment_data', './data/voc_classes.txt')
    voc_train_data = VocData('./data/2007_train.txt', voc_train_annotation)

    voc_val_annotation = VOCAnnotation(2007, 'val', '/Users/lx/segment_data', './data/voc_classes.txt')
    voc_val_data = VocData('./data/2007_val.txt', voc_val_annotation)

    # pascal voc 20个类别
    model = FastRCNN(20)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    model.compile(optimizer=SGD(lr=1e-3), loss=lambda y_true, y_pred: y_pred)
    model.build(input_shape=[(None, cfg.DEFAUTL_IMAGE_SIZE, cfg.DEFAUTL_IMAGE_SIZE, 3),
                             (None, None, 1), (None, None, 5), (None, None, 80),
                             (None, None, 80), (None, None, 80)])
    batch_size = 1
    model.fit_generator(voc_train_data.data_generator_wrapper(),
                        steps_per_epoch=max(1, voc_train_data.example_nums // batch_size),
                        validation_data=voc_val_data.data_generator_wrapper(),
                        validation_steps=max(1, voc_val_data.example_nums // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint])
    model.save_weights(log_dir + 'trained_weights_stage_1.h5')
