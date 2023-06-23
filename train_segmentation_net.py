import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os
import cv2 
import numpy as np 
import pdb 
from tensorflow.keras import layers
import random 
import json 

# train_masks[train_masks == 15] = 2
# val_masks[val_masks == 15] = 2

# train_masks_one_hot = to_categorical(train_masks, num_classes=3)
# val_masks_one_hot = to_categorical(val_masks, num_classes=3)

def simplified_unet_model(input_size=(256, 256, 3), num_classes = 3):
    inputs = tf.keras.Input(input_size)
    
    # Encoder (contracting) path
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
    
    u8 = UpSampling2D((2, 2))(c5)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    # Create the U-Net model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class CustomMeanIoU(MeanIoU):
    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
    
def iou_loss(y_true, y_pred):
    def f_score(y_true, y_pred, beta=1):
        smooth = 1e-15
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        true_positive = tf.reduce_sum(y_true * y_pred)
        false_positive = tf.reduce_sum(y_pred) - true_positive
        false_negative = tf.reduce_sum(y_true) - true_positive

        return (1 + beta**2) * true_positive + smooth / ((1 + beta**2) * true_positive + beta**2 * false_negative + false_positive + smooth)

    return 1 - f_score(y_true, y_pred)

def load_images(img_list_path, folder, size=256, color_space='BGR'):
    
    img_list = np.loadtxt(img_list_path, dtype='str')
    images_array = np.zeros((len(img_list), size, size, 3))
    for j, img_name in enumerate(img_list):
        img_path = os.path.join(folder, img_name)
        imgcv = cv2.imread(img_path)
        if color_space == 'RGB':
            imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
        elif color_space == 'HSV':
            imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2HSV)
        imgcv_resized = cv2.resize(imgcv, (size, size))
        imgcv_resized = imgcv_resized / 255.0
        images_array[j, :, :, :] = imgcv_resized
    return images_array

def load_masks(img_list_path, folder, size=256):

    img_list = np.loadtxt(img_list_path, dtype='str')
    images_array = np.zeros((len(img_list), size, size))
    for j, img_name in enumerate(img_list):
        img_path = os.path.join(folder, img_name)
        imgcv = cv2.imread(img_path, 0)
        imgcv_resized = cv2.resize(imgcv, (size, size))
        images_array[j, :, :] = imgcv_resized
    return images_array

def load_images_from_folder(folder_path, size=256):

    #pdb.set_trace()
    images_path = os.listdir(folder_path)
    images_array = np.zeros((len(images_path), size, size, 3))

    for j, img in enumerate(images_path):
        img_path = os.path.join(folder_path, img)
        imgcv = cv2.imread(img_path)
        imgcv_resized = cv2.resize(imgcv, (size, size))
        imgcv_resized = imgcv_resized / 255.0
        images_array[j, :, :, :] = imgcv_resized
    
    return images_array

def load_masks_from_folder(folder_path, size=256):

    #pdb.set_trace()
    images_path = os.listdir(folder_path)
    images_array = np.zeros((len(images_path), size, size))

    for j, img in enumerate(images_path):
        img_path = os.path.join(folder_path, img)
        imgcv = cv2.imread(img_path, 0)
        imgcv_resized = cv2.resize(imgcv, (size, size))
        images_array[j, :, :] = imgcv_resized
    
    return images_array

def main():

    ## Parameters 
    IMG_SIZE = 256 
    EPOCHS = 10
    BATCH_SIZE = 32
    AUGMENT = False
    COLOR_SPACE = 'RGB'
    par = {
        'img':IMG_SIZE,
        'epochs':EPOCHS,
        'batch_size':BATCH_SIZE,
        'augment':AUGMENT,
        'color_space':COLOR_SPACE,
    }

    root_folder_MoFF = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/'
    rgb_folder_MoFF = os.path.join(root_folder_MoFF, 'RGB')
    train_images = load_images(os.path.join(root_folder_MoFF, 'train.txt'), rgb_folder_MoFF, size=IMG_SIZE, color_space=COLOR_SPACE)
    valid_images = load_images(os.path.join(root_folder_MoFF, 'validation.txt'), rgb_folder_MoFF, size=IMG_SIZE, color_space=COLOR_SPACE)
    #test_images = load_images(os.path.join(root_folder_MoFF, 'test.txt'), rgb_folder_MoFF, size=IMG_SIZE)
    masks_folder_MoFF = os.path.join(root_folder_MoFF, 'segmap3c')
    train_masks = load_masks(os.path.join(root_folder_MoFF, 'train.txt'), masks_folder_MoFF, size=IMG_SIZE)
    valid_masks = load_masks(os.path.join(root_folder_MoFF, 'validation.txt'), masks_folder_MoFF, size=IMG_SIZE)
    #test_masks = load_masks(os.path.join(root_folder_MoFF, 'test.txt'), masks_folder_MoFF, size=IMG_SIZE)

    train_masks_one_hot = to_categorical(train_masks, num_classes=3)
    valid_masks_one_hot = to_categorical(valid_masks, num_classes=3)
    #pdb.set_trace()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)    
    num_classes = train_masks.shape[-1]
    model = simplified_unet_model()

    # Create a tf.data pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks_one_hot))
    train_dataset = train_dataset.batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices((valid_images, valid_masks_one_hot))
    val_dataset = val_dataset.batch(BATCH_SIZE)#.map(lambda x, y: (resize(x), resize(y)))

    if AUGMENT:
        data_augmentation = tf.keras.Sequential([
            layers.Resizing(IMG_SIZE, IMG_SIZE),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])
        train_dataset.map(lambda x, y: (data_augmentation(x), data_augmentation(y)))

    output_dir = f'run_{str(random.random())[2:]}'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as jp:
        json.dump(par, jp, indent=3)

    # Compile the model
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=[CustomMeanIoU(num_classes)])

    checkpoint = ModelCheckpoint(f'{output_dir}/best_unet_model_da.h5', monitor='loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
    csv_logger = CSVLogger('training_log.csv')

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping, csv_logger]
    )

    # history = model.fit(
    #     x=train_images,
    #     y=train_masks_one_hot,
    #     # sample_weight=train_masks_weight_map,
    #     batch_size=32,
    #     epochs=1,
    #     # validation_data=(val_images, val_masks_one_hot),
    #     callbacks=[checkpoint, early_stopping, csv_logger]
    # )

    # Plot the training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    #plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    # Plot the training and validation mean IoU
    plt.subplot(1, 2, 2)
    plt.plot(history.history['custom_mean_io_u'], label='Training MeanIoU')
    #plt.plot(history.history['val_custom_mean_io_u_1'], label='Validation MeanIoU')
    plt.xlabel('Epoch')
    plt.ylabel('MeanIoU')
    plt.legend()
    plt.title('MeanIoU')

    # Save the plot
    plt.savefig(f'{output_dir}/metrics.png', dpi=300, bbox_inches='tight')
    #pdb.set_trace()

    plt.figure(figsize=(32,24))
    plt.subplot(241)
    plt.imshow(cv2.cvtColor((train_images[0,:,:,:]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.subplot(242)
    plt.title('Ground Truth')
    plt.imshow(train_masks_one_hot[0,:,:,:])
    plt.subplot(243)
    plt.title('Predictions (values)')
    pred = model.predict(np.expand_dims(train_images[0,:,:,:], axis=0), batch_size=1)[0,:,:,:]
    plt.imshow(pred)
    plt.subplot(244)
    plt.title('Predictions (labels)')
    pred_labels = tf.argmax(pred, axis=-1)
    plt.imshow(pred_labels)
    plt.subplot(245)
    plt.imshow(cv2.cvtColor((train_images[100,:,:,:]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.subplot(246)
    plt.title('Ground Truth')
    plt.imshow(train_masks_one_hot[100,:,:,:])
    plt.subplot(247)
    plt.title('Predictions (values)')
    pred = model.predict(np.expand_dims(train_images[100,:,:,:], axis=0), batch_size=1)[0,:,:,:]
    plt.imshow(pred)
    plt.subplot(248)
    plt.title('Predictions (labels)')
    pred_labels = tf.argmax(pred, axis=-1)
    plt.imshow(pred_labels)
    plt.savefig(f'{output_dir}/prediction_0_100.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(32,24))
    plt.subplot(241)
    plt.imshow(cv2.cvtColor((train_images[50,:,:,:]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.subplot(242)
    plt.title('Ground Truth')
    plt.imshow(train_masks_one_hot[50,:,:,:])
    plt.subplot(243)
    plt.title('Predictions (values)')
    pred = model.predict(np.expand_dims(train_images[50,:,:,:], axis=0), batch_size=1)[0,:,:,:]
    plt.imshow(pred)
    plt.subplot(244)
    plt.title('Predictions (labels)')
    pred_labels = tf.argmax(pred, axis=-1)
    plt.imshow(pred_labels)
    plt.subplot(245)
    plt.imshow(cv2.cvtColor((train_images[250,:,:,:]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.subplot(246)
    plt.title('Ground Truth')
    plt.imshow(train_masks_one_hot[250,:,:,:])
    plt.subplot(247)
    plt.title('Predictions (values)')
    pred = model.predict(np.expand_dims(train_images[250,:,:,:], axis=0), batch_size=1)[0,:,:,:]
    plt.imshow(pred)
    plt.subplot(248)
    plt.title('Predictions (labels)')
    pred_labels = tf.argmax(pred, axis=-1)
    plt.imshow(pred_labels)
    plt.savefig(f'{output_dir}/prediction_50_250.png', dpi=300, bbox_inches='tight')
    #pdb.set_trace()

if __name__ == "__main__":
    main()