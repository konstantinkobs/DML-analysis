import torch
from torchvision import transforms
from PIL import Image
from glob import glob

class ConvertToBGR(object):
    """
    Converts a PIL image from RGB to BGR
    """

    def __init__(self):
        pass

    def __call__(self, img):
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        return img

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class Multiplier(object):
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, img):
        return img*self.multiple

    def __repr__(self):
        return "{}(multiple={})".format(self.__class__.__name__, self.multiple)


transform = transforms.Compose([ConvertToBGR(),
                                transforms.Resize(256), 
                                transforms.CenterCrop(227), 
                                transforms.ToTensor(),
                                Multiplier(255),
                                transforms.Normalize(mean = [104, 117, 128], 
                                                     std = [1, 1, 1])])

def load_and_convert(path: str):
    img = Image.open(path).convert('RGB')
    return img, transform(img)


def load_and_convert_multiple(path: str):
    files = glob(path)
    imgs = []
    transformed_imgs = []
    for f in files:
        img, transformed_img = load_and_convert(f)
        imgs.append(img)
        transformed_imgs.append(transformed_img.unsqueeze(0))
    
    return imgs, torch.cat(transformed_imgs)
