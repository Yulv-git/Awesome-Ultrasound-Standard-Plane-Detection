<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-04-03 18:16:07
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-04-05 23:24:21
 * @FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/DCNN-MF-SP/Readme.md
 * @Description: Evaluation of Deep Convolutional Neural Networks for Automatic Classification of Common Maternal Fetal Ultrasound Planes
 * Init from https://github.com/Oussamayousre/automatic-classification-of-common-maternal-fetal-ultrasound-planes b784f0107fd8cd0368622c5da09a0b41d0a3eb04
-->

<h1><center> Evaluation of Deep Convolutional Neural Networks for Automatic Classification of Common Maternal Fetal Ultrasound Planes </center></h1>

# Dataset

[Fetal_Planes_DB](../../data/FETAL_PLANES_DB/)

# Dependent libraries

``` txt
tensorflow 2.4
tensorflow-gpu 2.4
keras 2.4
...
```

# Running

``` bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_type base_model --model_name base_model

CUDA_VISIBLE_DEVICES=1 python main.py --model_type DCNN --model_name VGG19
```
