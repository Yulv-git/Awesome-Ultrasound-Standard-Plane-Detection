<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-03-18 10:33:37
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-03-27 21:43:43
 * @FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/SonoNet/README.md
 * @Description: PyTorch implementation of SonoNet.
 * Init from https://github.com/rdroste/SonoNet_PyTorch master 58e9f636be6fbfd87fbd1a27520b872828afa218
-->

# PyTorch SonoNet

**Disclaimer**  
These files come without any warranty!  
In particular, there might be unforeseen differences to the original implementation.

## About this repository

This is a PyTorch implementation of SonoNet:

Baumgartner et al., "Real-Time Detection and Localisation of Fetal Standard Scan Planes in 2D Freehand Ultrasound", arXiv preprint:1612.05601 (2016)

This repository is based on https://github.com/baumgach/SonoNet-weights which provides a theano+lasagne implementation.

## Files

sononet/sononet.py:  
PyTorch implementation of the original models.py file.

sononet/SonoNet16.pth, sononet/SonoNet32.pth, sononet/SonoNet64.pth:  
The original pretrained weights converted into PyTorch format.

infer.py:  
Modified version of the original example.py file. This file runs classification on the examples images.

## Dependencies

NumPy, Pillow, Matplotlib, PyTorch.  
Tested with PyTorch 0.4.0 and 1.3.1.

## Usage

After installing the dependencies, classify the example example images with:

``` bash
cd code/SonoNet
python infer.py
```
