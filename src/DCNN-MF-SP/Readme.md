<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-04-03 18:16:07
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-04-05 17:52:23
 * @FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/DCNN-MF-SP/Readme.md
 * @Description: Evaluation of Deep Convolutional Neural Networks for Automatic Classification of Common Maternal Fetal Ultrasound Planes
 * Init from https://github.com/Oussamayousre/automatic-classification-of-common-maternal-fetal-ultrasound-planes b784f0107fd8cd0368622c5da09a0b41d0a3eb04
-->

<h1><center> Evaluation of Deep Convolutional Neural Networks for Automatic Classification of Common Maternal Fetal Ultrasound Planes </center></h1>

# Dataset

[Fetal_Planes_DB](../../data/FETAL_PLANES_DB/)

# Running

``` bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_type base_model --model_name base_model

CUDA_VISIBLE_DEVICES=1 python main.py --model_type VGG --model_name VGG19
# CUDA_VISIBLE_DEVICES=1 python main.py --model_type VGG --model_name VGG19 --imagenet_pretrained True

CUDA_VISIBLE_DEVICES=2 python main.py --model_type ResNet --model_name ResNet50
# CUDA_VISIBLE_DEVICES=2 python main.py --model_type ResNet --model_name ResNet50 --imagenet_pretrained True

CUDA_VISIBLE_DEVICES=3 python main.py --model_type DenseNet --model_name DenseNet121
# CUDA_VISIBLE_DEVICES=3 python main.py --model_type DenseNet --model_name DenseNet121 --imagenet_pretrained

CUDA_VISIBLE_DEVICES=4 python main.py --model_type EfficientNet --model_name EfficientNetB6
# CUDA_VISIBLE_DEVICES=4 python main.py --model_type EfficientNet --model_name EfficientNetB6 --imagenet_pretrained

CUDA_VISIBLE_DEVICES=4 python main.py --model_type ViT --model_name ViT
```
