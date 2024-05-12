
# Captcha-Recognition-AI

An AI-based CAPTCHA recognition system using scalable computing techniques. This project employs normalization, denoising, erosion, and convolutional neural networks (CNNs) to accurately recognize CAPTCHA images. Optimized with TensorFlow Lite for efficient inference on resource-constrained devices like Raspberry Pi.

## Installation

### Local System
Install the required Python libraries on your local system using the following commands:
```bash
pip install opencv-python
pip install numpy
pip install captcha
pip install tensorflow
pip install keras
pip install pillow
```

### Raspberry Pi
To run the project on a Raspberry Pi, install these libraries:
```bash
pip install numpy
pip install tflite-runtime
pip install pillow
pip install csv
```

## Pre-trained Model
Download the pre-trained model from [this link](https://drive.google.com/file/d/1adFO7Dy4ro4BqVmzv2f1AKd4_bQmGVV2/view?usp=drive_link).

## Execution Instructions

### Download Test Images
Test images are already available in the `test` folder. To download them again:
```bash
./Getfiles.py
```

### Create Training and Validation Images
Generate training and validation images using these commands:

#### Training Images
```bash
./generate.py --width 128 --height 64 --length 6 --symbols symbol.txt --count 100000 --output-dir training_data --font eamonwide.woff.ttf
```

#### Validation Images
```bash
./generate.py --width 128 --height 64 --length 6 --symbols symbol.txt --count 100000 --output-dir validation_data --font eamonwide.woff.ttf
```

### Model Training
Train the model with:
```bash
./train.py --width 128 --height 64 --length 6 --symbols symbol.txt --batch-size 25 --epochs 10 --output-model-name v1 --train-dataset training_data --validate-dataset validation_data
```

### Local Classification
To classify CAPTCHAs on your local system and generate the CSV output:
```bash
./classify.py --model-name v1 --captcha-dir test --output submit.csv --symbols symbol.txt
```

### Raspberry Pi Classification
1. **Convert the Model:** Convert the trained TensorFlow model to TensorFlow Lite:
    ```bash
    ./convert_to_tflite.py
    ```

2. **Classify on Raspberry Pi:** Navigate to `www/toy_dir/proj2/final` and generate the CSV file:
    ```bash
    ./classify_tflite.py --model-name v1_converted --captcha-dir test --output submit.csv --symbols symbol.txt
    ```

