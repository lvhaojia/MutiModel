# Lab5-多模态情感分析


## 实验任务

- 给定配对的文本和图像，预测对应的情感标签。
- 三分类任务：positive, neutral, negative。


## 环境

具体在requirements.txt中

```shell
pip install -r requirements.txt
```


## 文件结构

```
|-- data	# 文本+图像的数据集，太大，不上传
|-- bert-base-chinese  # 预加载bert模型，太大，不上传
|-- main.py	
|-- resnet18-f37072fd.pth # pytorch自带ResNet18模型
|-- model.py # main.py中用到的类
|-- train.txt	# 训练集: 数据的guid和对应的情感标签
|-- test_without_label.txt	# 测试集: 数据的guid和空的情感标签
|-- test.txt	# 预测结果
|-- README.md   # 本文件
|-- requirements.txt   # 需要下载的环境
```


## 执行流程

```shell
python main.py
```
可能会出现很多warnings


## 参考

主要用到了pytorch

