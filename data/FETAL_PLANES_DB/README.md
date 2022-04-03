<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-03-29 22:07:29
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-04-03 22:54:49
 * @FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/data/FETAL_PLANES_DB/README.md
 * @Description: Fetal_Planes_DB
 * Init from https://zenodo.org/record/3904280# FETAL_PLANES_ZENODO.zip/README.md
-->

# Fetal_Planes_DB

**Burgos-Artizzu, X.P., Coronado-Gutiérrez, D., Valenzuela-Alcaraz, B. et al. Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes. Sci Rep 10, 10200 (2020). https://doi.org/10.1038/s41598-020-67076-5**

## Data Description

A large dataset of routinely acquired maternal-fetal screening ultrasound images collected from two different hospitals by several operators and ultrasound machines. All images were manually labeled by an expert maternal fetal clinician (B.V-A.). Images were divided into 6 classes: four of the most widely used fetal anatomical planes (Abdomen, Brain, Femur and Thorax), the mother’s cervix (widely used for prematurity screening) and a general category to include any other less common image plane. Fetal brain images were further categorized into the 3 most common fetal brain planes (Trans-thalamic, Trans-cerebellum, Trans-ventricular) to judge fine grain categorization performance. The final dataset is comprised of over 12,400 images from 1,792 patients.

Images are in `./Images/*.png`

All information related with the images is in `FETAL_PLANES_DB_data` (provided both in csv and xlsx formats)

The dataset details are described in our open-acces paper: [Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes](https://rdcu.be/b47NX)

If you find this dataset useful, please cite:

    @article{Burgos-ArtizzuFetalPlanesDataset,
      title={Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes},
      author={Burgos-Artizzu, X.P. and Coronado-Gutiérrez, D. and Valenzuela-Alcaraz, B. and Bonet-Carne, E. and Eixarch, E. and Crispi, F. and Gratacós, E.},
      journal={Nature Scientific Reports}, 
      volume={10},
      pages={10200},
      doi="10.1038/s41598-020-67076-5",
      year={2020}
    } 

## Download

``` bash
bash ./download.sh
```
