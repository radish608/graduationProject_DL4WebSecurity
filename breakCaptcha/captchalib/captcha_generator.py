# -*- coding:utf-8 -*-

from captcha.image import ImageCaptcha
import os
import random
import time
import numpy as np

class Generator:
    width = 100
    height = 60
    def __init__(self, characters, root_dir):
        self.characters = characters
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

    def generate(self, img_count, char_count):
        file_suffix = "png"

        for i in range(img_count):
            text = ""
            for j in range(char_count):
                text += random.choice(self.characters)
            img = ImageCaptcha(self.width, self.height).generate_image(text)
            now_time = str(time.time()).replace(".", "")
            path = os.path.join(self.root_dir, "{}_{}.{}".format(text, now_time, file_suffix))
            img.save(path)

if __name__ == '__main__':
    root_dir = "/home/randolph1997/DL4WebSecurity/breakCaptcha/captchalib/dataset"
    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    generator = Generator(characters, root_dir)
    img_count = 6000
    char_count = 4
    generator.generate(img_count, char_count)
