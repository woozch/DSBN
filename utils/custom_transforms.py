import torch
import random
import math
import numbers
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms import Lambda, Compose


class NumpyToTensor(object):
    def __call__(self, numpy_pic):
        numpy_pic = numpy_pic.transpose([2, 0, 1])
        numpy_pic = numpy_pic.astype(np.float32)
        return torch.from_numpy(numpy_pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PILToNumpy(object):
    def __call__(self, pil_img):
        numpy_img = np.array(pil_img, dtype=np.float32)
        return numpy_img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RGBtoBGR(object):
    def __call__(self, numpy_pic):
        converted_img = np.flip(numpy_pic, 2)
        return np.array(converted_img.copy())

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SubtractMean(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, numpy_pic):
        """
        Args:
            pic (numpy pic): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        numpy_pic = numpy_pic - self.mean.reshape(1, 1, 3)
        return np.array(numpy_pic.copy())

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float tuple): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float tuple): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float tuple): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float tuple): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        for tup in [self.brightness, self.contrast, self.saturation, self.hue]:
            assert tup[0] <= tup[1]

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """

        transforms = []
        brightness_factor = random.uniform(max(0, brightness[0]), brightness[1])
        transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        contrast_factor = random.uniform(max(0, contrast[0]), contrast[1])
        transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        saturation_factor = random.uniform(max(0, saturation[0]), saturation[1])
        transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        hue_factor = random.uniform(-hue[0], hue[1])
        transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomColorRotation(object):
    def __init__(self, degrees=(0, 0)):
        self.degrees = degrees

    def __call__(self, tensor):
        """
        Args:
            tensor (TorchTensor): Input TorchTensor [C x H x W].

        Returns:
            PIL Image: Color jittered image.
        """
        degree = random.uniform(self.degrees[0], self.degrees[1])
        radian = math.radians(degree)
        cos_val = math.cos(radian)
        sin_val = math.sin(radian)

        rotation_matrices = [
            torch.tensor([[cos_val, -sin_val, 0],
                          [sin_val, cos_val, 0],
                          [0, 0, 1]]),
            torch.tensor([[1, 0, 0],
                          [0, cos_val, -sin_val],
                          [0, sin_val, cos_val]]),
            torch.tensor([[cos_val, 0, sin_val],
                          [0, 1, 0],
                          [-sin_val, 0, cos_val]])
        ]
        rotation_matrix = random.choice(rotation_matrices)
        tensor_shape = tensor.shape
        num_channels = tensor_shape[0]
        assert num_channels == 3, 'num_channels is not 3. Tensor shape is {}'.format(tensor_shape)
        tensor = rotation_matrix.matmul(tensor.view(num_channels, -1)).view(tensor_shape)
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'degrees={0})'.format(self.degrees)
        return format_string


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = degrees
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees[1]

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        # angle = random.uniform(degrees[0], degrees[1])
        angle = np.random.normal(0, math.radians(degrees), 1)[0]
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)


class AddGaussianNoise(object):
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        tensor += torch.randn(list(tensor.shape)) * self.std
        return tensor

    def __repr__(self):
        s = '{name}(std={std})'
        return s.format(name=self.__class__.__name__, std=self.std)
