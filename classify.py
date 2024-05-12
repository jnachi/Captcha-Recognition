#!/usr/bin/env python3

import warnings
import os
import numpy as np
import string
import argparse
import tensorflow as tf
import csv
import keras
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Decode the model's output
def decode(characters, y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])

# Main function for classifying captchas
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    # Check for missing arguments
    if not all(vars(args).values()):
        print("Please provide all the required arguments")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()
    captcha_symbols = captcha_symbols + '@'

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with tf.device('/gpu:0'):
        with open(args.output, 'w', newline='') as output_file:
            file = csv.writer(output_file)
            file.writerow(['janaparn'],)
            json_file = open(args.model_name + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(args.model_name + '.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])
            file_list = sorted(os.listdir(args.captcha_dir))
            for x in (file_list):
                if x.endswith(('.png')):
                    image_path = os.path.join(args.captcha_dir, x)
                    raw_data = Image.open(image_path)
                    rgb_data = raw_data.convert('RGB')
                    image = np.array(rgb_data) / 255.0
                    (h, w, c) = image.shape
                    image = image.reshape([-1, h, w, c])
                    prediction = model.predict(image)
                    file.writerow([x,decode(captcha_symbols, prediction).replace(' ', '').replace('@', '')])
                    print('Classified ' + x)

if __name__ == '__main__':
    main()
