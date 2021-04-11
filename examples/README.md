# Scalable Learned Image Compression With A Recurrent Neural Networks-Based Hyperprior

This repository includes the code with trained models in the following paper:

"Scalable Learned Image Compression With A Recurrent Neural Networks-Based Hyperprior," 2020 IEEE International Conference on Image Processing([ICIP](https://ieeexplore.ieee.org/abstract/document/9190704))

Rige Su, Zhengxue Cheng, Heming Sun, Jiro Katto

# Summary of Paper

Recently learned image compression has achieved many great progresses, such as representative hyperprior and its variants based on convolutional neural networks (CNNs). However, CNNs are not fit for scalable coding and multiple models need to be trained separately to achieve variable rates. In this paper, we incorporate differentiable quantization and accurate entropy models into recurrent neural networks (RNNs) architectures to achieve a scalable learned image compression. First, we present an RNN architecture with quantization and entropy coding. To realize the scalable coding, we allocate the bits to multiple layers, by adjusting the layer-wise lambda values in Lagrangian multiplier-based rate-distortion optimization function. Second, we add an RNN-based hyperprior to improve the accuracy of entropy models for multiplelayer residual representations. Experimental results demonstrate that our performance can be comparable with recent CNN-based hyperprior methods on Kodak dataset. Besides, our method is a scalable and flexible coding approach, to achieve multiple rates using one single model, which is very appealing in practice.

# Environment

- Python==3.6.4
- Tensorflow==1.15.0
- Tensorflow-Compression==1.2


```
pip3 install tensorflow-compression
```

# Usage 

This model is optimized by MS-SSIM using lambda = 0.01, 0.02, 0.04, 0.08 for each layer.

For training

```
python rnn.py train
```

For testing,  put your images to the directory of valid/ and run the py files

```
python rnn.py --batchsize 1 compress xxx.png compressed.bin
```

```
python rnn.py --batchsize 1 decompress compressed.bin xxx.png
```

# Reconstructed Samples

![image-20210411220145289](C:\Users\kangs\AppData\Roaming\Typora\typora-user-images\image-20210411220145289.png)



# Evaluation Results

![image-20210411220215502](C:\Users\kangs\AppData\Roaming\Typora\typora-user-images\image-20210411220215502.png)

# Notes

If you think it is useful for your reseach, please cite our ICIP2020 paper. Our original RD data in the paper is contained in the folder RD/.

```
@INPROCEEDINGS{9190704,
  author={R. {Su} and Z. {Cheng} and H. {Sun} and J. {Katto}},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)}, 
  title={Scalable Learned Image Compression With A Recurrent Neural Networks-Based Hyperprior}, 
  year={2020},
  volume={},
  number={},
  pages={3369-3373},
  doi={10.1109/ICIP40778.2020.9190704}}
```
