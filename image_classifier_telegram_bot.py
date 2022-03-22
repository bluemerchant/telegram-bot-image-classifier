from telegram.ext import *
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf

TOKEN = 'YOUR BOT TOKEN'

#cifar10 training dataset is used which is already available
#build convolutional NN to classify images

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#normalize data

x_train, x_test = x_train / 255, x_test / 255

#list of classes in cifar10

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#basic sequential model

model = tf.keras.models.Sequential()

#add layers
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


def start(update, context):
    update.message.reply_text("Hey there! It's text classifier bot!")

def help(update, context):
    update.message.reply.text("""
    /start - start chat
    /help - show this message
    /train - train NN
    """)

def train(update, context):
    update.message.reply_text("Model is being trained...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.save('cifar_classifier.model')
    update.message.reply_text("You can now send me photos!")

def handle_message(update, context):
    update.message.reply_text("Please, train the model and send a picture!")

def handle_photo(update, context):
    #preprocesing the image
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

    #get image object
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #swap bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #resize the image
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    #prediction is activation of all neurons
    prediction = model.predict(np.array([img / 255]))
    update.message.reply_text(f"I think, I can see a {class_names[np.argmax(prediction)]}")

updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("train", train))
dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))


updater.start_polling()
updater.idle()
