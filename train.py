import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from keras.utils import print_summary
from os.path import join
from config import args
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from custom_losses import dice_hard, dice_hard_intersection, dice_hard_union
from capsNet import build_arch, get_splits, build_arch2, build_arch3
import numpy as np


def train(model, train_list, val_list):

    (x_train, y_train) = train_list
    (x_test, y_test) = val_list

    # Compile the loaded model
    opt = Adam(lr=args.lr, beta_1=0.99, beta_2=0.999, decay=1e-6)
    metrics = {'out_seg': [dice_hard, dice_hard_intersection, dice_hard_union]}
    loss = {'out_seg': 'binary_crossentropy', 'out_recon': 'mse'}
    loss_weighting = {'out_seg': 1., 'out_recon': args.recon_wei}

    training_model.compile(optimizer=opt, loss=loss, metrics=metrics)

    #######################################
    #               CallBacks             #
    #######################################

    monitor_name = 'val_out_seg_dice_hard_intersection'

    csv_logger = CSVLogger(join(args.logs_dir, 'epoch_logs_'+str(args.time)+'.csv'))
    tb = TensorBoard(join(args.logs_dir, 'tensorflow_logs'), batch_size=args.batch_size, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(join(args.weights_dir, 'weights_{epoch:04d}.h5'),
                                       monitor=monitor_name, save_weights_only=True,
                                       verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=5, verbose=1, mode='max')
    # early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=25, verbose=0, mode='max')

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size):
        train_datagen = ImageDataGenerator()
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[model_checkpoint, csv_logger, lr_reducer, tb], verbose=1)

    model.save_weights(join(args.models_dir, "model_"+str(args.time)+".h5"))
    print("=" * 50)
    print("Model saved to --> " + str(join(args.models_dir, "model_"+str(args.time)+".h5")))
    print("=" * 50)


if __name__ == "__main__":
    
    (x_train, y_train), (x_val, y_val) = get_splits(verbose=args.debug)

    net_input_shape = x_train.shape[1:]
    training_model, _ = build_arch2(net_input_shape, n_class=args.num_class)
    if args.restore_model:
        training_model.load_weights(join(args.models_dir, args.weights))
        if args.debug:
            print("="*51)
            print("Model loaded from", str(join(args.models_dir, args.weights)))
            print("="*51)
    print_summary(model=training_model, positions=[.38, .65, .75, 1.])

    train(training_model, (x_train, y_train), (x_val, y_val))