<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-03-18 00:27:15
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-03-26 21:33:07
 * @FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/README.md
 * @Description: A curated list of awesome ultrasound standard plane detection.
 * Repository: https://github.com/Yulv-git/Awesome-Ultrasound-Standard-Plane-Detection
-->

<h1><center> Awesome-Ultrasound-Standard-Plane-Detection </center></h1>

A curated list of awesome ultrasound standard plane detection.

---

- [1. Papers](#1-papers)
- [2. Others](#2-others)
- [3. Practice](#3-practice)
- [4. Acknowledgements](#4-acknowledgements)

---

# 1. Papers

| Papers & Code | Short for Schemes | Notes |
| :------------ | :---------------: | :---- |
| Automated Selection of Standardized Planes from Ultrasound Volume [[MICCAI-MLMI 2011]](https://link.springer.com/content/pdf/10.1007/978-3-642-24319-6_5.pdf) | 2STC | Sliding Window, Haar, AdaBoost, AS Detection, SP Classification |
| Learning-based scan plane identification from fetal head ultrasound images [[Medical Imaging 2012]](https://sci-hub.se/10.1117/12.911516) | - | Template Matching, Active Appearance Models, AS Detection, LDA, SP Classification |
| Intelligent Scanning: Automated Standard Plane Selection and Biometric Measurement of Early Gestational Sac in Routine Ultrasound Examination [[Medical Physics 2012]](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1118/1.4736415?saml_referrer) | IS | Sliding Window, Haar, Cascade AdaBoost, AS Localization, Relative Position, Local Context Information, SP Classification |
| Selective Search and Sequential Detection for Standard Plane Localization in Ultrasound [[MICCAI-CCCAI 2013]](https://link.springer.com/content/pdf/10.1007/978-3-642-41083-3_23.pdf) | SSSD | Haar, AdaBoost, Segmentation, Accumulative Vessel Probability Map, Selective Search, Geometric Relationship, Sequence AS Detection, SP Localization |
| Standard Plane Localization in Ultrasound by Radial Component [[ISBI 2014]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6868086) | RCD | Random Forest, Geometric Constrain, Radial Component, AS Detection, SVM, SP Localization |
| Automatic Recognition of Fetal Standard Plane in Ultrasound Image [[ISBI 2014]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6867815) | FV-Aug | AdaBoost, Dense Sampling Feature Transform Descriptor, Fish Vector, Spatial Pyramid Coding, Gaussian Mixture Model, SVM, SP Classification |
| Standard Plane Localization in Ultrasound by Radial Component Model and Selective Search [[Ultrasound in Medicine and Biology 2014]](https://www.sciencedirect.com/science/article/pii/S0301562914004098/pdfft?md5=2f202092b37f5f31009c48b8845b10d3&pid=1-s2.0-S0301562914004098-main.pdf) | RVD | Random Forest, Geometric Constrain, Radial Component, Vessel Probability Map, Selective Search, AS Detection, SVM, SP Localization |
| Diagnostic Plane Extraction from 3D Parametric Surface of the Fetal Cranium [[MIUA 2014]](http://www.bmva.org/miua/2014/miua-14-04.pdf) | - | Topological Manifold Representation, Landmark Alignment, 3D Parametric Surface Model, SP Localization |
| A Constrained Regression Forests Solution to 3D Fetal Ultrasound Plane Localization for Longitudinal Analysis of Brain Growth and Maturation [[MICCAI-MLMI 2014]](https://link.springer.com/content/pdf/10.1007/978-3-319-10581-9_14.pdf) | CRF-FA-Dist | Informative Voxels, Reference Plane, Constrained Regression Forest, SP Localization |
| Automatic Recognition of Fetal Facial Standard Plane in Ultrasound Image via Fisher Vector [[PLOS ONE 2015]](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0121838&type=printable) | FV-Chi2-SDCA | Spatial Stacking, Densely Sampled Root Scale Invariant Feature Transform, Gaussian Mixture Model, Fisher Vector, Multilayer Fisher Network, SVM, SP Classification |
| Plane Localization in 3-D Fetal Neurosonography for Longitudinal Analysis of the Developing Brain [[JBHI 2015]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7110508) | CRF-FA-Dist-M | Informative Voxels, Manual Reference Plane, Constrained Regression Forest, SP Localization |
| Standard Plane Localization in Fetal Ultrasound via Domain Transferred Deep Neural Networks [[JBHI 2015]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7090943) | T-CNN | Knowledge Transfer, CNN, SP Localization |
| Automatic Fetal Ultrasound Standard Plane Detection Using Knowledge Transferred Recurrent Neural Networks [[MICCAI 2015]](https://link.springer.com/content/pdf/10.1007/978-3-319-24553-9_62.pdf) | T-RNN | CNN, Knowledge Transfer, Joint Learning, Spatio-temporal Feature, RNN, SP Classification |
| Fetal Facial Standard Plane Recognition via Very Deep Convolutional Networks [[EMBC 2016]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7590780) | - | DCNN, SP Classification |
| Real-Time Standard Scan Plane Detection and Localisation in Fetal Ultrasound Using Fully Convolutional Neural Networks [[MICCAI 2016]](https://link.springer.com/content/pdf/10.1007/978-3-319-46723-8_24.pdf) | - | CNN, Unsupervision,  Saliency Maps, AS Localization, SP Classification |
| Ultrasound Standard Plane Detection Using a Composite Neural Network Framework [[Transactions on Cybernetics 2017]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7890445) | T-RNN | CNN, RNN, Composite Framework, SP Classification |
| SonoNet: Real-Time Detection and Localisation of Fetal Standard Scan Planes in Freehand Ultrasound [[TMI 2017]](https://arxiv.org/pdf/1612.05601v2.pdf) [[Official Code]](https://github.com/baumgach/SonoNet-weights) [[Third-Party Code]](https://github.com/rdroste/SonoNet_PyTorch) | SonoNet | CNN, SP Classification, Weakly Supervision, AS Localization |
| Automatic Detection of Standard Sagittal Plane in the First Trimester of Pregnancy Using 3-D Ultrasound Data [[Ultrasound in Medicine and Biology 2017]](https://www.sciencedirect.com/science/article/pii/S0301562916302708/pdfft?md5=cac6c0141b1bc88c489d2178a575c530&pid=1-s2.0-S0301562916302708-main.pdf) | - | Deep Belief Network, Circle Detection, SP Classification |
| Attention-Gated Networks for Improving Ultrasound Scan Plane Detection [[MIDL 2018]](https://arxiv.org/pdf/1804.05338v1.pdf) [[Official Code]](https://github.com/ozan-oktay/Attention-Gated-Networks) | AG-SonoNet | CNN, Attention, SP Classification, Weakly Supervision, AS Localization |
| Standard Plane Localisation in 3D Fetal Ultrasound Using Network with Geometric and Image Loss [[MIDL 2018]](https://openreview.net/pdf?id=BykcN8siz) | - | CNN, Rigid Transformation, Geometric Loss, Image Loss, SP Localization |
| Standard Plane Detection in 3D Fetal Ultrasound Using an Iterative Transformation Network [[MICCAI 2018]](https://arxiv.org/pdf/1806.07486v2.pdf) [[Official Code]](https://github.com/yuanwei1989/plane-detection) | ITN | CNN, Rigid Transformation, SP Localization |
| Automatic and Efficient Standard Plane Recognition in Fetal Ultrasound Images via Multi-scale Dense Networks [[MICCAI-DATRA/PIPPI 2018]](https://link.springer.com/content/pdf/10.1007/978-3-030-00807-9_16.pdf) | MSDNet | Multi-scale, Cascade, Dense Connection, CNN, SP Classification |
| SPRNet: Automatic Fetal Standard Plane Recognition Network for Ultrasound Images [[ MICCAI-PIPPI/SUSI 2019]](https://link.springer.com/content/pdf/10.1007/978-3-030-32875-7_5.pdf) | SPRNet | CNN, Weight-share, Transfer Learning, SP Classification |
| Deep Learning-Based Methodology for Recognition of Fetal Brain Standard Scan Planes in 2D Ultrasound Images [[IEEE Access 2019]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8887441) | - | Data Augmentation, DCNN, Domain Transfer, SP Classification |
| Standard Plane Identification in Fetal Brain Ultrasound Scans Using A Differential Convolutional Neural Network [[IEEE Access 2020]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9084099) | Different-CNN | Differential Operator, Differential CNN, SP Classification |
| Evaluation of Deep Convolutional Neural Networks for Automatic Classification of Common Maternal Fetal Ultrasound Planes [[Scientific Reports 2020]](https://www.nature.com/articles/s41598-020-67076-5.pdf) | - | Data Augmentation, PCA, Hog, Boosting, VGG, MobileNet, Inception-v3, ResNet, SENet, SE-ResNet, DenseNet, SP Classification |
| Recognition of Fetal Facial Ultrasound Standard Plane Based on Texture Feature Fusion [[CMMM 2021]](https://downloads.hindawi.com/journals/cmmm/2021/6656942.pdf) | LH-SVM | Local Binary Pattern, Histogram of Oriented Gradient, Feature Fusion, SVM, SP Classification |
| Agent with Warm Start and Adaptive Dynamic Termination for Plane Localization in 3D Ultrasound [[TMI 2021]](https://arxiv.org/pdf/2103.14502v1.pdf) [[Official Code]](https://github.com/wulalago/AgentSPL) | AgentSPL | Landmark Alignment, Reinforcement Learning, CNN, RNN, SP Localization |
| Automatic Fetal Ultrasound Standard Plane Recognition Based on Deep Learning and IIoT [[Transactions on Industrial Informatics 2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9389489) | FUSPR | CNN, RNN, Spatial-temporal Feature, SP Classification |

Tags:
Standard Plane --> SP | Anatomical Structure --> AS

# 2. Others

pass

# 3. Practice

- [SonoNet](./src/SonoNet/README.md): [infer.py](./src/SonoNet/infer.py)
- [ITN](./src/ITN/README.md): [train.py](./src/ITN/train.py), [infer.py](./src/ITN/infer.py)
- [AG-SonoNet](./src/AG-SonoNet/README.md)
- [AgentSPL](./src/AgentSPL/README.md)

# 4. Acknowledgements

Thanks to the contributors of all the above papers, code, and other resources.
