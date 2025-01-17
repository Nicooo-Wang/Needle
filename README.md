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
4. waiting for results
```
(infra) nico@Poor3070:~/Needle$ python sample.py 
Using needle backend
running on 0th batch
running on 1th batch
running on 2th batch
running on 3th batch
running on 4th batch
running on 5th batch
running on 6th batch
running on 7th batch
running on 8th batch
running on 9th batch
running on 10th batch
running on 11th batch
running on 12th batch
running on 13th batch
running on 14th batch
running on 15th batch
running on 16th batch
running on 17th batch
running on 18th batch
running on 19th batch
running on 20th batch
running on 21th batch
running on 22th batch
...
Epoch 0 | Train Accuracy: 0.3351582480818414 | Train Loss: 1.8470205068588257
...
Epoch 1 | Train Accuracy: 0.44824568414322247 | Train Loss: 1.5284370183944702
...
Epoch 2 | Train Accuracy: 0.4963235294117647 | Train Loss: 1.4007797241210938
...
Epoch 3 | Train Accuracy: 0.5313419117647058 | Train Loss: 1.305657148361206
...
Epoch 4 | Train Accuracy: 0.5585158248081841 | Train Loss: 1.228232502937317
...
```
# References
1. [10-414/714: Deep Learning Systems ](https://dlsyscourse.org/)
2. [The forum of 10-414/714 ](https://forum.dlsyscourse.org/)
4. [知乎：PyTorch 源码解读之 torch.utils.data ](https://zhuanlan.zhihu.com/p/337850513)
5. [https://github.com/wzbitl/needle](https://github.com/wzbitl/needle)
