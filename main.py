import numpy as np
import pandas as pd

img_dir = 'images/'
imgs_df = pd.read_csv('train.csv')
imgpaths = pd.Series(img_dir + imgs_df['img_id'].values+'.jpeg', name='img_id')
metastasis_ratios = pd.Series(imgs_df['metastasis_ratio'].values, name='metastasis_ratio')
images = pd.concat([imgpaths, metastasis_ratios], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(images, train_size=0.8, shuffle=True, random_state=1)
import tensorflow as tf

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='img_id',
    y_col='metastasis_ratio',
    target_size=(512, 512),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='img_id',
    y_col='metastasis_ratio',
    target_size=(512, 512),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='img_id',
    y_col='metastasis_ratio',
    target_size=(512, 512),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)
inputs = tf.keras.Input(shape=(512, 512, 3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse'
)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)
model.save('metastasis_ratiodetect_model_10epochs.h5')
from keras.models import load_model
model = load_model('metastasis_ratiodetect_model_10epochs.h5')

import csv

sub = open('sample_submission.csv', 'w')
writer = csv.writer(sub)
data1 = ["img_id", "predicted_metastasis_ratio"]
writer.writerow(data1)


timgdf = pd.read_csv('test.csv')
testimgspaths = pd.Series(img_dir+ timgdf['img_id'].values+'.jpeg', name='img_id')
premetastasis_ratios = pd.Series(timgdf['predicted'].values, name='predicted')
imagess = pd.concat([testimgspaths, premetastasis_ratios], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
testt_images = test_generator.flow_from_dataframe(
    dataframe=imagess,
    x_col='img_id',
    y_col= 'predicted',
    target_size=(512, 512),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=False)

imgnames = []
for j, image_name in enumerate(testimgspaths):
    imgname = image_name.split('.')[0]
    imgnames.append(imgname)

imgnames = np.array(imgnames)
predictions = model.predict(testt_images)
i=0
while i<938 :

    data2 = [imgnames[i].split('/')[1], predictions[i][0]]
    writer.writerow(data2)
    i = i + 1
    data2 = []

