#!/usr/bin/env python3

import warnings
import os
import cv2
import numpy as np
import string
import random
import argparse
import tensorflow as tf
from tensorflow import keras

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create the captcha model
def create_captcha_model(captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
    input_tensor = keras.Input(input_shape)
    x = input_tensor

    for i, module_length in enumerate([module_size] * model_depth):
        for j in range(module_length):
            x = keras.layers.Conv2D(32 * 2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    outputs = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name=f'char_{i+1}')(x) for i in range(captcha_length)]
    model = keras.Model(inputs=input_tensor, outputs=outputs)

    return model

# Custom data sequence for image data
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(np.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, len(self.captcha_symbols)), dtype=np.uint8) for _ in range(self.captcha_length)]

        for i in range(self.batch_size):
            if not self.files:
                break

            random_image_label, random_image_file = self.get_random_image()
            self.used_files.append(random_image_label)
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            processed_data = np.array(rgb_data) / 255.0
            X[i] = processed_data
            self.update_labels(random_image_label, y, i)

        return X, y

    def get_random_image(self):
        random_image_label = random.choice(list(self.files.keys()))
        random_image_file = self.files[random_image_label]
        return random_image_label, random_image_file

    def update_labels(self, random_image_label, y, i):
        random_image_label = random_image_label.split('_')[0].replace("/", ":")
        for j, ch in enumerate(random_image_label):
            y[j][i, :] = 0
            y[j][i, self.captcha_symbols.find(ch)] = 1

# Main function for training the captcha model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--batch-size', help='How many images in training captcha batches', type=int)
    parser.add_argument('--train-dataset', help='Path to the training image dataset', type=str)
    parser.add_argument('--validate-dataset', help='Path to the validation image dataset', type=str)
    parser.add_argument('--output-model-name', help='Name for the trained model', type=str)
    parser.add_argument('--input-model', help='Path to the input model for continued training', type=str)
    parser.add_argument('--epochs', help='Number of training epochs', type=int)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    # Check for missing arguments
    required_args = ['width', 'height', 'length', 'batch_size', 'train_dataset', 'validate_dataset', 'output_model_name', 'epochs', 'symbols']
    if not all(vars(args)[arg] for arg in required_args):
        print("Please provide all the required arguments")
        exit(1)

    captcha_symbols = None
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()
    captcha_symbols = captcha_symbols + '@'

    # Create and configure the captcha model
    with tf.device('/device:GPU:0'):
        model = create_captcha_model(args.length, len(captcha_symbols), (args.height, args.width, 3))

        # Load pre-trained model if specified
        if args.input_model is not None:
            model.load_weights(args.input_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        model.summary()

        # Create data generators for training and validation
        training_data = ImageSequence(args.train_dataset, args.batch_size, args.length, captcha_symbols, args.width, args.height)
        validation_data = ImageSequence(args.validate_dataset, args.batch_size, args.length, captcha_symbols, args.width, args.height)

        callbacks = [keras.callbacks.EarlyStopping(patience=3),
                     keras.callbacks.ModelCheckpoint(args.output_model_name + '.h5', save_best_only=False)]

        # Save the model architecture to JSON
        with open(args.output_model_name + ".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit(training_data,
                      validation_data=validation_data,
                      epochs=args.epochs,
                      callbacks=callbacks,
                      use_multiprocessing=True)
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + args.output_model_name + '_resume.h5')
            model.save_weights(args.output_model_name + '_resume.h5')

if __name__ == '__main__':
    main()