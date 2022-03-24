<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-03-18 00:27:15
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-03-24 12:20:18
 * @FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/README.md
 * @Description: A curated list of awesome ultrasound standard plane detection.
 * Repository: https://github.com/Yulv-git/Awesome-Ultrasound-Standard-Plane-Detection
-->

<h1><center> Awesome-Ultrasound-Standard-Plane-Detection </center></h1>

A curated list of awesome ultrasound standard plane detection.

---

- [1. Papers](#1-papers)
- [2. Reviews](#2-reviews)
- [3. Others](#3-others)
- [4. Practice](#4-practice)
- [5. Acknowledgements](#5-acknowledgements)

---

# 1. Papers

| Papers & Code | Schemes | Notes |
| :------------ | :-----: | :---- |
| Automated Selection of Standardized Planes from Ultrasound Volume [[MICCAI-MLMI 2011]](https://link.springer.com/content/pdf/10.1007/978-3-642-24319-6_5.pdf) | | Haar, AdaBoost, AS Detection, SP Classification |
| SonoNet: Real-Time Detection and Localisation of Fetal Standard Scan Planes in Freehand Ultrasound [[TMI 2017]](https://arxiv.org/pdf/1612.05601v2.pdf) [[Official Code]](https://github.com/baumgach/SonoNet-weights) [[Third-Party Code]](https://github.com/rdroste/SonoNet_PyTorch) | SonoNet | CNN, SP Classification, Weakly Supervision, AS Localization |
| Standard Plane Detection in 3D Fetal Ultrasound Using an Iterative Transformation Network [[MICCAI 2018]](https://arxiv.org/pdf/1806.07486v2.pdf) [[Official Code]](https://github.com/yuanwei1989/plane-detection) | ITN | CNN, SP Localization |
| Attention-Gated Networks for Improving Ultrasound Scan Plane Detection [[MIDL 2018]](https://arxiv.org/pdf/1804.05338v1.pdf) [[Official Code]](https://github.com/ozan-oktay/Attention-Gated-Networks) | AG_SonoNet | CNN, Attention, SP Classification, Weakly Supervision, AS Localization |
| Agent with Warm Start and Adaptive Dynamic Termination for Plane Localization in 3D Ultrasound [[TMI 2021]](https://arxiv.org/pdf/2103.14502v1.pdf) [[Official Code]](https://github.com/wulalago/AgentSPL) | AgentSPL | Reinforcement Learning, CNN, SP Localization, Landmark Alignment, RNN |

Tags:
Standard Plane --> SP
Anatomical Structure --> AS

# 2. Reviews

| Papers & Code | Schemes | Notes |
| :------------ | :-----: | :---- |
|               |         |       |

# 3. Others

pass

# 4. Practice

- [SonoNet](./src/SonoNet/README.md): [infer.py](./src/SonoNet/infer.py)
- [ITN](./src/ITN/README.md): [train.py](./src/ITN/train.py), [infer.py](./src/ITN/infer.py)
- [AG_SonoNet](./src/AG_SonoNet/README.md)
- [AgentSPL](./src/AgentSPL/README.md)

# 5. Acknowledgements

Thanks to the contributors of all the above papers, code, reviews, and other resources.
