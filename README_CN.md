## 描述
The code includes training and inference procedures for CDNet: Centripetal Direction Network for Nuclear Instance Segmentation.

## 数据集：[MoNuSeg](<https://monuseg.grand-challenge.org/>)
- MoNuSeg
    - 训练集：data/MoNuSeg_oridata/images/train_300
    - 测试集：data/MoNuSeg_oridata/images/test1
- 数据格式：
    - png，为提升训练时的数据读取效率，train的数据在 data_process.py做预处理。

## 环境要求
- 硬件（Ascend或GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
    - Mindspore 1.5
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)
- 依赖安装
    - pip install -r requirement.txt

## 脚本说明
- 数据预处理
    - 为提升训练中对数据的读取，在训练前执行 
    - python data_process.py

- 训练
    - python train.py 
    - 模型保存在 checkpoint 文件夹中

- 评测
    - python eval.py

## 评估指标
- sklearn AJI: 0.45
- hover AJI: 0.58
- hover Dice: 0.80