"""
Factory method for easily getting dataset by name.
written by wgchang
"""
import utils.custom_transforms as custom_transforms
from torchvision import transforms
from numpy import array

__sets = {}
from dataset.datasets import SVHN, MNIST, USPS
from dataset.datasets import OFFICE, OFFICEHOME, OFFICECALTECH
from dataset.datasets import IMAGECLEF, VISDA

# for digit DA
MNIST_DIR = './data/MNIST'
SVHN_DIR = './data/SVHN'
USPS_DIR = './data/USPS'
SPLITS = ['train', 'test', 'val']
JITTERS = ['None', ]
for model_name in ['lenet', 'lenetdsbn']:
    for domain in ['mnist', 'svhn', 'usps']:
        for jitter in JITTERS:
            if domain == 'mnist':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: x.repeat([3, 1, 1])),
                ])
                for split in SPLITS:
                    name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                           split=split, jitter=jitter)
                    train = True if split == 'train' else False
                    __sets[name] = (
                        lambda train=train, transform=transform: MNIST(MNIST_DIR, train=train, download=True,
                                                                       transform=transform))

            elif domain == 'svhn':
                transform = transforms.Compose([
                    transforms.Resize([28, 28]),
                    transforms.Grayscale(),
                    transforms.ToTensor()
                ])
                for split in SPLITS:
                    name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                           split=split, jitter=jitter)
                    split = 'test' if split == 'val' else split
                    __sets[name] = (
                        lambda split=split, transform=transform: SVHN(SVHN_DIR, split=split, download=True,
                                                                      transform=transform))

            elif domain == 'usps':
                transform = transforms.Compose([
                    transforms.Resize([28, 28]),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: x.repeat([3, 1, 1])),
                ])
                for split in SPLITS:
                    name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                           split=split, jitter=jitter)
                    train = True if split == 'train' else False
                    __sets[name] = (
                        lambda train=train, transform=transform: USPS(USPS_DIR, train=train, transform=transform))

# for Office DA
OFFICE_DIR = './data/Office/domain_adaptation_images'
JITTERS = ['None', ]
for model_name in ['alexnet', 'resnet50', 'cresnet50', 'resnet50dsbn', 'cpuanet50', 'cpuanet50dsbn']:
    for split in SPLITS:
        for jitter in JITTERS:
            if model_name == 'alexnet':
                if jitter == 'None':
                    if split == 'train':
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                    else:
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                else:
                    continue
            else:  # for resnet
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])

            for domain in ['amazon', 'webcam', 'dslr']:
                name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                       split=split, jitter=jitter)
                __sets[name] = (
                    lambda domain=domain, transform=transform: OFFICE(OFFICE_DIR, domain=domain, transform=transform))

# for IMAGECLEF-DA
IMAGECLEF_DIR = './data/image-clef/'
JITTERS = ['None', ]
for model_name in ['alexnet', 'resnet50', 'resnet50dsbn', 'cpuanet50', 'cpuanet50dsbn']:
    for split in SPLITS:
        for jitter in JITTERS:
            # transforms
            if model_name == 'alexnet':
                if jitter == 'None':
                    if split == 'train':
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                    else:
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                else:
                    continue

            else:  # for resnet
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])

            for domain in ['imagenet', 'pascal', 'bing', 'caltech']:
                name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                       split=split, jitter=jitter)
                __sets[name] = (
                    lambda domain=domain, transform=transform: IMAGECLEF(IMAGECLEF_DIR, domain=domain,
                                                                         transform=transform))

# for VisDA
VISDA_DIR = './data/VisDA/'
JITTERS = ['None', 'competition', 'rgb160noaspect', 'grey160noaspect', 'rgb224noaspect', 'grey224noaspect', 'rgb224scaleup-noaspect',
           'rndgrey224scaleup', 'rndgrey224scaleup-noaspect', '160', 'rndgrey160scaleup']
for model_name in ['alexnet', 'resnet50', 'resnet50dsbn', 'resnet101', 'resnet101dsbn', 'resnet152', 'resnet152dsbn',
                   'cpuanet50', 'cpuanet50dsbn', 'cpuanet101', 'cpuanet101dsbn', 'cpuanet152', 'cpuanet152dsbn']:
    for split in SPLITS:
        for jitter in JITTERS:
            # transforms
            if model_name == 'alexnet':
                if jitter == 'None':
                    if split == 'train':
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                    else:
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                else:
                    continue

            elif jitter == 'competition':
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.Resize(176),
                        custom_transforms.ColorJitter(brightness=(0.75, 1.333), saturation=(0.0, 1.0)),
                        custom_transforms.RandomAffine(36, scale=(0.75, 1.333)),
                        transforms.RandomCrop(160),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        custom_transforms.RandomColorRotation(degrees=(-9, 9)),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                        custom_transforms.AddGaussianNoise(std=0.1)
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(176),
                        transforms.CenterCrop(160),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])

            elif jitter == 'grey160noaspect':
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.Resize(176),
                        transforms.Grayscale(3),
                        # transforms.RandomResizedCrop(160, scale=(0.75, 1.333), ratio=(1.0, 1.0)),
                        transforms.RandomCrop(160),
                        transforms.RandomAffine(0, scale=(0.75, 1.333)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(176),
                        transforms.Grayscale(3),
                        transforms.CenterCrop(160),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            elif jitter == 'rgb160noaspect':
                if split == 'train':
                    transform = transforms.Compose([
                        # transforms.Resize(176),
                        transforms.RandomResizedCrop(160, scale=(0.08, 1.333), ratio=(1.0, 1.0)),
                        # transforms.RandomCrop(160),
                        transforms.RandomAffine(0, scale=(0.75, 1.333)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(176),
                        transforms.CenterCrop(160),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            elif jitter == 'grey224noaspect':
                if split == 'train':
                    transform = transforms.Compose([
                        # transforms.RandomResizedCrop(224, scale=(0.75, 1.333), ratio=(1.0, 1.0)),
                        transforms.RandomCrop(160),
                        transforms.RandomAffine(0, scale=(0.75, 1.333)),
                        transforms.Grayscale(3),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.Grayscale(3),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            elif jitter == 'rgb224noaspect':
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.08, 1.333), ratio=(1.0, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            elif jitter == 'rgb224scaleup-noaspect':
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.08, 2.0), ratio=(1.0, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            elif jitter == 'rndgrey224scaleup':
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.08, 2.0)),
                        transforms.RandomGrayscale(),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            elif jitter == 'rndgrey224scaleup-noaspect':
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.08, 2.0), ratio=(1.0, 1.0)),
                        transforms.RandomGrayscale(),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            elif jitter == '160':
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(160),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(176),
                        transforms.CenterCrop(160),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            elif jitter == 'rndgrey160scaleup':
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(160, scale=(0.08, 2.0)),
                        transforms.RandomGrayscale(),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(176),
                        transforms.CenterCrop(160),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            else:  # for resnet
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
            for domain in ['train', 'validation', 'test']:
                name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                       split=split, jitter=jitter)
                __sets[name] = (
                    lambda domain=domain, transform=transform: VISDA(VISDA_DIR, domain=domain, transform=transform))

# Office Home datasets
# change "Real World" to "RealWorld" domain_name with space is not allowed!
OFFICEHOME_DIR = './data/Office-home/OfficeHomeDataset_10072016'
JITTERS = ['None', ]
for model_name in ['alexnet', 'resnet50', 'cresnet50', 'resnet50dsbn', 'cpuanet50', 'cpuanet50dsbn']:
    for split in SPLITS:
        for jitter in JITTERS:
            if model_name == 'alexnet':
                if jitter == 'None':
                    if split == 'train':
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                    else:
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                else:
                    continue
            else:  # for resnet
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])

            for domain in ['Art', 'Clipart', 'Product', 'RealWorld']:
                name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                       split=split, jitter=jitter)
                __sets[name] = (
                    lambda domain=domain, transform=transform: OFFICEHOME(OFFICEHOME_DIR, domain=domain,
                                                                          transform=transform))

# for OfficeCaltech DA
OFFICECALTECH_DIR = './data/OfficeCaltech'
JITTERS = ['None', ]
for model_name in ['alexnet', 'resnet50', 'cresnet50', 'resnet50dsbn', 'cpuanet50', 'cpuanet50dsbn']:
    for split in SPLITS:
        for jitter in JITTERS:
            if model_name == 'alexnet':
                if jitter == 'None':
                    if split == 'train':
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                    else:
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                else:
                    continue
            else:  # for resnet
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])

            for domain in ['amazonOC', 'webcamOC', 'dslrOC', 'caltechOC']:
                name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                       split=split, jitter=jitter)
                __sets[name] = (
                    lambda domain=domain, transform=transform: OFFICECALTECH(OFFICECALTECH_DIR, domain=domain[:-2],
                                                                             transform=transform))


def get_dataset(name):
    if name not in __sets:
        raise KeyError('Unknown Dataset: {}'.format(name))
    return __sets[name]()
