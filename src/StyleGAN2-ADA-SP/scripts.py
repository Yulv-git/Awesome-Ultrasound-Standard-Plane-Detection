#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-31 21:43:42
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-31 22:37:21
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/StyleGAN2-ADA/scripts.py
Description: 
Init from https://github.com/albertoMontero/stylegan2-ada
'''
from pathlib import Path

from calc_metrics import calc_metrics


def calc_metrics_dir(path, metric_names, metricdata, mirror, gpus=1, n=None, start=None, end=None, reverse=True):
    """
    calc_metric for networks contained in directory. Optionally, select n last/first snapshots (reverse True/False),
    or by start and end with network-snapshot numbers (sorted based on reverse).
    """

    if not isinstance(metric_names, list):
        metric_names = [metric_names]

    path = Path(path)
    print(path)
    _networks = [p for p in path.glob("*.pkl")]
    _networks = sorted(_networks, reverse=reverse)

    if len(_networks) == 0:
        print(f"No networks found in {path}")
        return

    print("total networks in folder: ", len(_networks))
    info = None

    if n:
        networks = _networks[:n]
        if reverse:
            info = f"Calculating metrics {metric_names} for last {n} networks:\n{[n.name for n in networks]}"
        else:
            info = f"Calculating metrics {metric_names} for first {n} networks:\n{[n.name for n in networks]}"
    elif start and end:
        networks = []
        for ns in _networks:
            tick = int(ns.name.split(".")[0].split("-")[-1])
            if start <= tick <= end:
                networks.append(ns)
    else:
        networks = _networks

    if not info:
        info = f"Calculating metrics {metric_names} for networks:\n{[n.name for n in networks]}"
    print(info)

    for ns in networks:
        calc_metrics(str(ns), metric_names, metricdata, mirror, gpus)


if __name__ == "__main__":
    # path = "/home/alberto/Data/github/stylegan2-ada/models/trv"
    # data = "/home/alberto/Data/github/stylegan2-ada/data/trv_s128"
    # calc_metrics_dir(path, ["fid50k_full", "pr50k3_full"], data, True, n=3, start=2, end=4, reverse=False)

    # path = "/home/alberto/Data/master/TFM/tmp"
    # data = "/home/alberto/Data/master/TFM/usbrains/data/processed/colab_2/tf_records/dbp/"
    # calc_metrics_dir(path, ["fid50k_full"], data, False, n=1, reverse=True)

    path = "/home/alberto/Data/github/stylegan2-ada/models/bl2/dbp/s2"
    data = "/home/alberto/Data/github/stylegan2-ada/data/bl2/dbp"
    # calc_metrics_dir(path, ["fid50k_full"], data, False, n=1, reverse=True)
    # calc_metrics_dir(path, ["pr50k3_full"], data, False, n=1, reverse=True)
    calc_metrics_dir(path, ["fid50k_full"], data, False, start=589, end=589, reverse=False)
