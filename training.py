import keras
from keras.optimizers import Adagrad,Adam,SGD
from keras.callbacks import TensorBoard
from keras import Sequential,backend
from keras.layers import Dense,Conv2D,Conv1D,MaxPooling2D,Dropout,Dense,Flatten,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

train_datagen = ImageDataGenerator(
    height_shift_range=0.2,
    horizontal_flip=True)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(directory='./train',
                                  target_size=(4000,16),
                                  batch_size=64,color_mode="grayscale")

val_generator = val_datagen.flow_from_directory(directory='./valid',
                                  target_size=(4000,16),
                                batch_size=50,color_mode="grayscale")



def define_model():
    model =Sequential()
    model.add(Conv2D(64,kernel_size=[24,1],strides=[2,1],padding="same",activation='relu',input_shape=(4000,16,1)))
    model.add(Conv2D(64,kernel_size=[24,1],strides=[2,1],padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,1),strides=(2, 1)))
    model.add(Conv2D(128,kernel_size=[24,1],strides=[2,1],padding="same",activation='relu'))
    model.add(Conv2D(128,kernel_size=[24,1],strides=[2,1],padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,1),strides=(2, 1)))
    model.add(Conv2D(256,kernel_size=[24,1],strides=[2,1],padding="same",activation='relu'))
    model.add(Conv2D(256,kernel_size=[24,1],strides=[2,1],padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,1),strides=(2, 1)))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    return model


model =define_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)
def setup_to_train(model):
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


setup_to_train(model)
history=model.fit_generator(generator=train_generator, 
                   steps_per_epoch=3000,
                    epochs=30,
                    validation_data=val_generator,
                    validation_steps=100,
                    class_weight='auto',callbacks=[early_stopping])
model.save('deephic_model.h5')






