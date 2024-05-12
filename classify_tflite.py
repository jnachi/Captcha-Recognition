#!/usr/bin/env python3

import warnings
import os
import numpy as np
import string
import argparse
import tflite_runtime.interpreter as tflite
from PIL import Image
import csv

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Decode the model's output
def decode(characters, y):
    y = np.argmax(np.array(y), axis=1)
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

    img_files = os.listdir(args.captcha_dir)
    img_files = sorted(img_files)

    # Load the TFLite interpreter
    with open(args.output, 'w', newline='') as output_file:
        file = csv.writer(output_file)
        file.writerow(['janaparn'],)
        interpreter = tflite.Interpreter(args.model_name + '.tflite')
        interpreter.allocate_tensors()

        for x in img_files:
            if x.endswith(('.png')):
            # Load image and preprocess it
                raw_data = Image.open(os.path.join(args.captcha_dir, x))
                image = np.array(raw_data) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w])

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                image = image.astype('float32')
                interpreter.set_tensor(input_details[0]['index'], image)
                interpreter.invoke()
                captcha = ""
                captcha_final = ""

                # Decode the output for each character
                char1 = decode(captcha_symbols, interpreter.get_tensor(output_details[3]['index']))
                char2 = decode(captcha_symbols, interpreter.get_tensor(output_details[5]['index']))
                char3 = decode(captcha_symbols, interpreter.get_tensor(output_details[0]['index']))
                char4 = decode(captcha_symbols, interpreter.get_tensor(output_details[4]['index']))
                char5 = decode(captcha_symbols, interpreter.get_tensor(output_details[2]['index']))
                char6 = decode(captcha_symbols, interpreter.get_tensor(output_details[1]['index']))
                captcha = captcha + char1 + char2 + char3 + char4 + char5 + char6

                # Remove '/' character
                for i in captcha:
                    if i != '/':
                        captcha_final = captcha_final + i
                
                file.writerow([x,captcha_final.replace(' ', '').replace('@', '')])
                print('Classified ' + x)

if __name__ == '__main__':
    main()
