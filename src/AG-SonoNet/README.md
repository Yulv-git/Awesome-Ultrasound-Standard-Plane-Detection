<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-03-20 18:17:37
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-04-03 12:31:26
 * @FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/AG-SonoNet/README.md
 * @Description: Modify here please
 * Init from https://github.com/ozan-oktay/Attention-Gated-Networks eee4881fdc31920efd873773e0b744df8dacbfb6
-->

# Attention Gated Networks <br /> (Image Classification & Segmentation)

Pytorch implementation of attention gates used in U-Net and VGG-16 models. The framework can be utilised in both medical image classification and segmentation tasks.

<p align="center">
    <img src="figures/figure1.png" width="640"> <br />
    <em> The schematics of the proposed Attention-Gated Sononet</em>
</p>

<p align="center">
    <img src="figures/figure2.jpg" width="640"> <br />
    <em> The schematics of the proposed additive attention gate</em>
</p>

## References

1) "Attention-Gated Networks for Improving Ultrasound Scan Plane Detection", MIDL'18, Amsterdam <br />
[Conference Paper](https://openreview.net/pdf?id=BJtn7-3sM) <br />
[Conference Poster](https://www.doc.ic.ac.uk/~oo2113/posters/MIDL2018_poster_Jo.pdf)

2) "Attention U-Net: Learning Where to Look for the Pancreas", MIDL'18, Amsterdam <br />
[Conference Paper](https://openreview.net/pdf?id=Skft7cijM) <br />
[Conference Poster](https://www.doc.ic.ac.uk/~oo2113/posters/MIDL2018_poster.pdf)

## Dependent library installation

```bash
cd src/AG_SonoNet
pip install -e git+https://github.com/ozan-oktay/torchsample.git#egg=torchsample
```

**Please note that the original repository does not specify the environment in which its code is run, such as the versions of the various Python libraries, which may cause errors in the run, such as when using VisDOM for visualization.**
