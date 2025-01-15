# Needle
Needle is torch like educational Mechine Learning framework, The skeleton of this repository is from [CMU 10-414/714](https://dlsyscourse.org/) Deep Learning System course.

# Features
- Automatic differentiation support
- Three backends ( CPU, CUDA, numpy )
- common oprators (e.g. [Im2Col Conv](https://github.com/Nicooo-Wang/Needle/blob/343deaa73e357fcb9e1b23b4be3ba898f3c705ba/python/needle/ops/ops_mathematic.py#L513), [2D Tiling Matmal](https://github.com/Nicooo-Wang/Needle/blob/343deaa73e357fcb9e1b23b4be3ba898f3c705ba/src/ndarray_backend_cuda.cu#L548))
- basic optimzers (e.g. adam, SGD)
- common models (e.g. [ Resnet9 ](https://github.com/Nicooo-Wang/Needle/blob/343deaa73e357fcb9e1b23b4be3ba898f3c705ba/sample.py#L14))

# Sample Use
1. create conda environment
```bash
conda create --name Needle
conda activate Needle
conda install pybind11
```
2. download dataset
```
python download_dataset.py
```
3. run `Resnet9` sample
```
python sample.py
```

# References
1. [10-414/714: Deep Learning Systems ](https://dlsyscourse.org/)
2. [The forum of 10-414/714 ](https://forum.dlsyscourse.org/)
4. [知乎：PyTorch 源码解读之 torch.utils.data ](https://zhuanlan.zhihu.com/p/337850513)
5. [https://github.com/wzbitl/needle](https://github.com/wzbitl/needle)
