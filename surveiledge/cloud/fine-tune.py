from keras.preprocessing.image import ImageDataGenerator
from keras_applications import mobilenet_v2
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import time, os
from MobileNet_v2 import MobileNetV2
from keras.layers.core import Activation
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from sys import argv
import shutil

def queryTrain(queryClass,alpha):
    print('model training starts...')
    start_time=time.time()

    img_width, img_height = 224, 224

    train_data_dir1 = '/home/cloud/queryTrainData1_' + queryClass
    train_data_dir2 = '/home/cloud/queryTrainData2_' + queryClass
    weightPath1 = '/home/cloud/queryTrain1_' + queryClass +'.h5'
    weightPath2 = '/home/cloud/queryTrain2_' + queryClass +'.h5'

    classNum = 2
    epochs = 1
    batch_size = 32

    base_model = MobileNetV2(input_shape=(img_width, img_height,3),
                    alpha=alpha,
                    include_top=False,
                    weights= 'imagenet'
                    )

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(classNum,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    # for layer in model.layers:
    #     layer.trainable=False
    # or if we want to set the first 20 layers of the network to be non-trainable
    for layer in model.layers[:20]:
        layer.trainable=False
    for layer in model.layers[20:]:
        layer.trainable=True

    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir1,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    es = EarlyStopping(monitor='acc', patience=1)

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[es])

    
    model.save_weights(weightPath1)

    shutil.rmtree(train_data_dir1)
    shutil.rmtree(train_data_dir2)

    print('done')
    end_time=time.time()
    print("Total training time: ",end_time-start_time)
    return weightPath1,weightPath2