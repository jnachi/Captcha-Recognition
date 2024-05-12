#!/usr/bin/env python3

import os
import numpy as np
import random
import string
import cv2
import argparse
from captcha.image import ImageCaptcha

def generate_captchas(args):
    if not os.path.exists(args.font):
        print("Font file not found")
        exit(1)

    captcha_generator = ImageCaptcha(fonts=[args.font], width=args.width, height=args.height)

    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    print("Generating captchas with symbol set {" + captcha_symbols + "} using font: " + args.font)

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    captcha_set = set()
    for i in range(args.count):
        random_str = ''.join([random.choice(captcha_symbols) for j in range(random.randint(1, args.length))])
        while random_str in captcha_set:
            random_str = ''.join([random.choice(captcha_symbols) for k in range(random.randint(1, args.length))])
        captcha_set.add(random_str)
        label = random_str + '@' * (args.length - len(random_str))
        image_path = os.path.join(args.output_dir, label + '.png')
        image = np.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--font', help='Font file to use for captchas', type=str)
    args = parser.parse_args()

    if args.width is None or args.height is None or args.length is None or args.count is None or args.output_dir is None or args.symbols is None or args.font is None:
        print("Please provide all the required arguments")
        exit(1)

    generate_captchas(args)

if __name__ == '__main__':
    main()
