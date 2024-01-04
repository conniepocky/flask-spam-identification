import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("spam_dataset.csv")

y = data.label_num

X = data.text

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"])

result = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

eval = model.evaluate(x_test, y_test)

for name, value in zip(model.metrics_names, eval):
    print("%s: %.3f" % (name, value))


model.save("spam_model.h5")

epoch = result.epoch
val_loss = result.history["val_loss"]
plt.plot(epoch, val_loss)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.show()
