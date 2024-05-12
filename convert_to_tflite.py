#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import tensorflow as tf


def main():
    model_name='v1'
    output_dir=''
        # Loading the model
    with open(f'{model_name}.json') as f:
            model = f.read()
    model = tf.keras.models.model_from_json(model)
    model.load_weights(f'{model_name}.h5')

        # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

        # Save the model.
    with open(f'{output_dir}{model_name}_converted.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"Successfully converted the {model_name} to TFLite Model!")

if __name__ == '__main__':
    main()
