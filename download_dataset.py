import urllib.request
import os
import tarfile

# 创建数据目录
os.makedirs('./data/ptb', exist_ok=True)

# 下载 Penn Treebank 数据集
ptb_data = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
for f in ['train.txt', 'test.txt', 'valid.txt']:
    file_path = os.path.join('./data/ptb', f)
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(ptb_data + f, file_path)

# 下载 CIFAR-10 数据集
cifar_dir = "./data/cifar-10-batches-py"
if not os.path.isdir(cifar_dir):
    urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "./data/cifar-10-python.tar.gz")
    
    # 解压缩 CIFAR-10 数据集
    with tarfile.open('./data/cifar-10-python.tar.gz', 'r:gz') as tar:
        tar.extractall(path='./data')
