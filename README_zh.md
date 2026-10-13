# ALM-HNM

ALM-HNM 是用于复现论文实验的代码和结果归档项目，研究主题为在严重类别不平衡条件下，将 semi-hard top-k 困难负样本挖掘融合到增广拉格朗日训练中。

## 环境要求与安装

训练环境为 Ubuntu 22.04、Python 3.8.5、PyTorch 2.4.1+cu121、torchvision 0.19.1、NumPy 1.19.1、pandas 1.1.3、SciPy 1.5.2、scikit-learn 0.23.2、matplotlib 3.3.2、Pillow 7.2.0、PyYAML 5.4.1 和 tqdm 4.50.2。推荐使用 conda：

```bash
conda env create -f environments/environment_train.yml
conda activate alm-hnm-train
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

只核验结果时可使用 `environment_analysis.yml`，无需安装 CUDA 或重新训练。

## 数据准备

代码通过 torchvision 下载公开 CIFAR10、CIFAR100 和 SVHN。首次运行时设置 `DOWNLOAD=1`，数据保存到仓库下的 `datasets/cifar10/`、`datasets/cifar100/` 和 `datasets/svhn/`。原始图像不包含在本仓库中。二分类标签、失衡比例、预处理和数据来源见 `docs/REPRODUCIBILITY.md` 与 `docs/DATA_PROVENANCE.md`。

## 目录结构

`main.py` 为训练入口；`models/` 为模型；`utils/` 为数据集、损失函数、训练和集成代码；`configs/` 为固定配置；`run_*.sh` 为正式实验脚本；`results/` 为表格、统计、日志和六个无权重结果归档；`analysis/` 为离线核验与作图脚本；`environments/` 为环境文件。

## 引用与联系方式

请引用论文：[论文标题]，International Journal of Machine Learning and Cybernetics，[年份]，[DOI]。发表后同时引用 Zenodo 版本 DOI；GitHub 可根据根目录 `CITATION.cff` 生成软件引用。通讯作者：Wang Yu，wangyu@haou.edu.cn。

## 数据可用性

代码、配置、源数据表、预测输出、逐种子指标、日志和六个结果归档通过同一 GitHub 仓库发布，并由 GitHub 联动的 Zenodo Release 归档。CIFAR10、CIFAR100 和 SVHN 为 torchvision 提供的公开第三方数据集，不在本项目中重复存储；模型权重不归档，但可通过固定环境、脚本和随机种子重新训练。

## 许可证

Copyright (c) 2026 Wang Yu。本项目使用 MIT License，见 `LICENSE`。公开发布前必须确认仓库中包含的上游 ALM-DNN 代码具有合法再分发权限；原始本地副本没有附带许可证。
