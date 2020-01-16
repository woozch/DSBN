import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from utils.io_utils import download_url, check_integrity
import zipfile
import h5py
import functools


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.classes = [str(i) for i in range(10)]

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], int(self.train_labels[index])
        else:
            img, target = self.test_data[index], int(self.test_labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class USPS(data.Dataset):
    url = "https://www.kaggle.com/bistaumanga/usps-dataset/downloads/usps.h5"

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.classes = [str(i) for i in range(10)]

        self.preprocess()

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], int(self.train_labels[index])
        else:
            img, target = self.test_data[index], int(self.test_labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def preprocess(self):
        """Download the USPS data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        filename = self.url.rpartition('/')[2]

        print('Processing...')
        # unzip downloaded file
        zip_file_path = os.path.join(self.root, self.raw_folder, filename + '.zip')
        h5_file_path = os.path.join(self.root, self.raw_folder, filename)
        if os.path.isfile(zip_file_path) and not os.path.isfile(h5_file_path):
            zip_ref = zipfile.ZipFile(zip_file_path)
            zip_ref.extractall(os.path.join(self.root, self.raw_folder))
            zip_ref.close()

        if not os.path.isfile(h5_file_path):
            raise RuntimeError('USPS Dataset not found.' +
                               ' Download dataset from {} and place to {}'.format(self.url,
                                                                                  os.path.join(self.root,
                                                                                               self.raw_folder)))

        with h5py.File(os.path.join(self.root, self.raw_folder, filename)) as hf:
            train = hf.get('train')
            X_tr = np.array(train.get('data')[:] * 255, dtype=np.uint8)
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = np.array(test.get('data')[:] * 255, dtype=np.uint8)
            y_te = test.get('target')[:]

        # convert h5 file to torch tensor file
        training_set = (
            torch.from_numpy(X_tr.reshape(X_tr.shape[0], 16, 16)),
            torch.from_numpy(y_tr).long()
        )
        test_set = (
            torch.from_numpy(X_te.reshape(X_te.shape[0], 16, 16)),
            torch.from_numpy(y_te).long()
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set
        self.classes = [str(i) for i in range(10)]

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class OFFICE(data.Dataset):
    """A OFFICE dataset loader: ::
    Args:
        root (string): Root directory path.
        domain (string): amazon, dslr, webcam
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.

        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, domain, list_file=None, transform=None, target_transform=None):

        self.extensions = ['jpg', 'jpeg', 'png']
        domain_root_dir = os.path.join(root, domain, 'images')
        classes = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
                   'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
                   'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer',
                   'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
                   'tape_dispenser', 'trash_can']
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        samples = make_dataset(domain_root_dir, class_to_idx, self.extensions, list_file=list_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(self.extensions)))

        self.root = domain_root_dir
        self.loader = Image.open
        self.domain = domain

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'OFFICE Dataset \n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class IMAGECLEF(data.Dataset):
    """A imageCLEF-DA dataset(https://github.com/thuml/Xlearn/tree/master/caffe) loader: ::
    Args:
        root (string): Root directory path.
        domain (string): bing, caltech, imagenet, pascal
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.

        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, domain, list_file=None, transform=None, target_transform=None):

        self.extensions = ['jpg', 'jpeg', 'png']
        classes = ['aeroplane', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'dog', 'horse', 'monitor', 'motorbike',
                   'people']
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        if (domain == 'imagenet'):
            list_file = os.path.join(root, 'list/iList.txt')
        elif (domain == 'pascal'):
            list_file = os.path.join(root, 'list/pList.txt')
        elif (domain == 'bing'):
            list_file = os.path.join(root, 'list/bList.txt')
        elif (domain == 'caltech'):
            list_file = os.path.join(root, 'list/cList.txt')
        else:
            raise AttributeError('Unexpected domain: {}'.format(domain))

        samples = make_dataset(root, class_to_idx, self.extensions, list_file=list_file, include_dir=True)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(self.extensions)))

        self.root = root
        self.loader = Image.open
        self.domain = domain

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # # L (gray, 1-chn) >> RGB (3-chns)
        # if (sample.size(0) == 1):
        #     sample = torch.cat((sample, sample, sample), 0)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'ImageCLEF-DA Dataset \n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class VISDA(data.Dataset):
    """A OFFICE dataset loader: ::
    Args:
        root (string): Root directory path.
        domain (string): train (synthetic), test (real)
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, domain, list_file=None, transform=None, target_transform=None):

        self.extensions = ['jpg', 'jpeg', 'png']
        domain_root_dir = os.path.join(root, domain)
        classes = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant',
                   'skateboard', 'train', 'truck']
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        samples = self.make_dataset(domain_root_dir, class_to_idx, self.extensions, list_file=list_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(self.extensions)))

        self.root = domain_root_dir
        self.loader = Image.open
        self.domain = domain

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def make_dataset(self, dir, class_to_idx, extensions, list_file=None):
        if list_file is None:
            images = []
            dir = os.path.expanduser(dir)
            for target in sorted(os.listdir(dir)):
                d = os.path.join(dir, target)
                if not os.path.isdir(d):
                    continue

                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, extensions):
                            path = os.path.join(root, fname)
                            item = (path, class_to_idx[target])
                            images.append(item)
        else:
            print('load dataset from {}'.format(list_file))
            with open(list_file) as f:
                images = [l.strip().split(' ') for l in f.readlines()]
                images = [(path, int(idx)) for (path, idx) in images]

        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path).convert('RGB')

        # L (gray, 1-chn) >> RGB (3-chns)
        # if (sample.size(0) == 1):
        #     sample = torch.cat((sample, sample, sample), 0)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'VisDA Dataset \n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class OFFICEHOME(data.Dataset):
    """A OFFICE Home dataset loader: ::
    Args:
        root (string): Root directory path.
        domain (string): Art, Clipart, Product, Real World
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.

        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, domain, list_file=None, transform=None, target_transform=None):

        self.extensions = ['jpg', 'jpeg', 'png']
        domain_root_dir = os.path.join(root, domain)
        classes = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar',
                   'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser',
                   'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer',
                   'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop',
                   'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer',
                   'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers',
                   'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']

        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        samples = make_dataset(domain_root_dir, class_to_idx, self.extensions, list_file=list_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(self.extensions)))

        self.root = domain_root_dir
        self.loader = Image.open
        self.domain = domain

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'OFFICE-Home Dataset \n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class OFFICECALTECH(data.Dataset):
    """A OFFICE dataset loader: ::
    Args:
        root (string): Root directory path.
        domain (string): amazonOC, webcamOC, dslrOC, caltechOC
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.

        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, domain, list_file=None, transform=None, target_transform=None):

        self.extensions = ['jpg', 'jpeg', 'png']
        domain_root_dir = os.path.join(root, domain, 'images')
        classes = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
                   'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        samples = make_dataset(domain_root_dir, class_to_idx, self.extensions, list_file=list_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(self.extensions)))

        self.root = domain_root_dir
        self.loader = Image.open
        self.domain = domain

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'OFFICE+CALTECH Dataset \n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions, list_file=None, include_dir=False):
    if list_file is None:
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
    else:
        print('load dataset from {}'.format(list_file))
        with open(list_file) as f:
            images = [l.strip().split(' ') for l in f.readlines()]
            if include_dir:
                images = [(os.path.join(dir, path), int(idx)) for (path, idx) in images]
            else:
                images = [(path, int(idx)) for (path, idx) in images]

    return images
