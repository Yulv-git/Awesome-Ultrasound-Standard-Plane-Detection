<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-03-20 18:17:37
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-03-23 22:15:46
 * @FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/AG_SonoNet/README.md
 * @Description: Modify here please
 * Init from https://github.com/ozan-oktay/Attention-Gated-Networks
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
wget https://github.com/ncullen93/torchsample/archive/refs/tags/v0.1.3.zip
unzip v0.1.3.zip
rm v0.1.3.zip
cd torchsample-0.1.3
python setup.py install
cd ..
rm -rf torchsample-0.1.3
```
