import os
import torch
import numpy as np


def download_lsun_data(data_name='', data_path='./Data') -> [torch.tensor, ] * 4:
    os.makedirs(data_path, exist_ok=True)

    from torchvision import datasets
    data_train = datasets.LSUN('/mnt/sdb1/weit/code/Demo_DL_RL/Data',
                               classes='train')
    # data_train = datasets.LSUN(root=data_path, classes='test')
    print(data_train)

    def to_images(ary, ):
        # ary = ten.numpy()
        ary = ary / 255.0
        ary = np.transpose(ary, (0, 3, 1, 2))
        # ary = ary.reshape((-1, 3, 28, 28))
        # ary = np.pad(ary, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')
        # ary = ary.reshape((-1, 1, 28, 28, 3))
        ary = torch.tensor(ary, dtype=torch.float32)
        return ary

    def to_labels(ary, ):
        # ary = ten.numpy()
        ary = ary.reshape((-1,))
        # classes_num = 10
        # data_sets = np.eye(classes_num)[data_sets] # one_hot
        ary = torch.tensor(ary, dtype=torch.long)
        return ary

    train_image = to_images(data_train.data)
    # train_label = to_labels(data_train.targets)
    # test_image = to_images(data_test.data)
    # test_label = to_labels(data_test.targets)
    return train_image, None,  # train_label, test_image, test_label


def download_cifar10_data(image_size, data_name='', data_path='./Data') -> [torch.tensor, ] * 4:
    os.makedirs(data_path, exist_ok=True)

    from torchvision import datasets
    data_train = datasets.CIFAR10("./Data", train=True, download=True, )
    # data_train = datasets.FashionMNIST("./Data", train=True, download=True, )
    # data_test = datasets.FashionMNIST("./Data", train=False, download=True, )
    print(data_train)

    def to_images(ary, ):
        # ary = ten.numpy()
        ary = ary / 255.0
        ary = np.transpose(ary, (0, 3, 1, 2))
        # ary = ary.reshape((-1, 3, 28, 28))
        # ary = np.pad(ary, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')
        # ary = ary.reshape((-1, 1, 28, 28, 3))
        ary = torch.tensor(ary, dtype=torch.float32)
        return ary

    def to_labels(ary, ):
        # ary = ten.numpy()
        ary = ary.reshape((-1,))
        # classes_num = 10
        # data_sets = np.eye(classes_num)[data_sets] # one_hot
        ary = torch.tensor(ary, dtype=torch.long)
        return ary

    train_image = to_images(data_train.data)
    # train_label = to_labels(data_train.targets)
    # test_image = to_images(data_test.data)
    # test_label = to_labels(data_test.targets)
    return train_image, None,  # train_label, test_image, test_label


def download_mnist_data(data_path='./Data') -> [torch.tensor, ] * 4:
    os.makedirs(data_path, exist_ok=True)

    from torchvision import datasets
    data_train = datasets.FashionMNIST("./Data", train=True, download=True, )
    data_test = datasets.FashionMNIST("./Data", train=False, download=True, )
    print(data_train)

    def to_images(ary, ):
        # ary = ten.numpy()
        ary = ary / 255.0
        ary = ary.reshape((-1, 3, 28, 28))
        ary = np.pad(ary, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')
        ary = ary.reshape((-1, 1, 28, 28, 3))
        ary = torch.tensor(ary, dtype=torch.float32)
        return ary

    def to_labels(ary, ):
        ary = ary.reshape((-1,))
        # classes_num = 10
        # data_sets = np.eye(classes_num)[data_sets] # one_hot
        ary = torch.tensor(ary, dtype=torch.long)
        return ary

    train_image = to_images(data_train.data)
    train_label = to_labels(data_train.targets)
    test_image = to_images(data_test.data)
    test_label = to_labels(data_test.targets)
    return train_image, train_label, test_image, test_label


def load_data(data_path, img_shape=(32, 32, 3)):  # 2021-01-28
    # data_path = './Data/MNIST/MNIST.npz'
    # data_path = './Data/CIFAR10/CIFAR10.npz'
    data_ary_list = np.load(data_path, allow_pickle=True)['arr_0']

    def to_images(ary):
        h, w, c = img_shape

        ary = ary / 128.0 - 1
        ary = ary.reshape((-1, h, w, c))
        ary = np.transpose(ary, (0, 3, 1, 2))
        ary = torch.as_tensor(ary, dtype=torch.float32)
        return ary

    def to_labels(ary, ):
        ary = ary.reshape((-1,))
        # classes_num = 10
        # ary = np.eye(classes_num)[ary]  # one_hot
        ary = torch.as_tensor(ary, dtype=torch.long)
        return ary

    train_images = to_images(data_ary_list[0])
    train_labels = to_labels(data_ary_list[1])

    eval_images = to_images(data_ary_list[2])
    eval_labels = to_labels(data_ary_list[3])
    return train_images, train_labels, eval_images, eval_labels


def load_data_ary(data_path, img_shape=(32, 32, 3), if_one_hot=True):  # 2021-01-30
    # data_path = './Data/MNIST/MNIST.npz'
    # data_path = './Data/CIFAR10/CIFAR10.npz'
    data_ary_list = np.load(data_path, allow_pickle=True)['arr_0']

    def to_images(ary):
        h, w, c = img_shape

        ary = ary / 128.0 - 1
        ary = ary.reshape((-1, h, w, c))
        ary = np.transpose(ary, (0, 3, 1, 2))
        return ary

    def to_labels(ary):
        ary = ary.astype(np.int)
        if if_one_hot:
            classes_num = int(ary.max()) + 1
            ary = np.eye(classes_num)[ary]
        else:
            ary = ary.reshape((-1,))
        return ary

    train_images = to_images(data_ary_list[0])
    train_labels = to_labels(data_ary_list[1])

    eval__images = to_images(data_ary_list[2])
    eval__labels = to_labels(data_ary_list[3])
    return train_images, train_labels, eval__images, eval__labels


def load_torch_model(mod_dir, model):
    model_save_path = os.path.join(mod_dir, 'model.pth')
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=lambda storage, loc: storage))
        model.eval()
    elif not os.path.exists(mod_dir):
        os.mkdir(mod_dir)
    else:
        pass
    return model_save_path


def whether_remove_history(mod_dir, remove=None):
    print('  Model: %s' % mod_dir)
    if remove is None:
        remove = bool(input("  'y' to REMOVE: %s? " % mod_dir) == 'y')
    if remove and os.path.exists(mod_dir):
        import shutil
        shutil.rmtree(mod_dir)
        print("| Remove")
        del shutil

    os.makedirs(mod_dir, exist_ok=True)

# if __name__ == '__main__':
#     from torchvision import datasets
#
#     datasets.svhn("./Data", train=True, download=True, )
