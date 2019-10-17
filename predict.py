from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
from keras.callbacks import TensorBoard
import keras
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import datetime
model = keras.models.load_model("deephic_model.h5")
test_datagen = ImageDataGenerator()
test_generator = val_datagen.flow_from_directory(directory='./test/',
                                  target_size=(4000,16),
                                  batch_size=1,color_mode="grayscale",shuffle = False)
start_time = datetime.datetime.now()
predict_y = model.predict_generator(test_generator,steps=len(test_generator))
end_time = datetime.datetime.now()

files = test_generator.filenames
f = open('files_test.txt','w')
f.write('\n'.join(files))


f.close()

numpy.savetxt("result_test.txt",predict_y)



