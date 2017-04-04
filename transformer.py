import random

class RandomRotation(object):
    def __call__(self, img):
        rot_multiplier = random.randint(0, 3)
        return img.rotate(rot_multiplier * 90)
