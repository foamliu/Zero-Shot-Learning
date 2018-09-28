# 零样本学习

零样本学习（zero-shot learning）是在已知类别上训练物体识别模型，要求模型能够识别来自未知类别的样本。对图像理解、（从已知类别到未知类别的）知识迁移具有重要意义。

## 依赖
- Python 3.5
- PyTorch 0.4

## 数据集
使用 AI Challenger 2018 的图像属性数据集，本数据集共78,017张图片，可划分为5个超类（super-class），分别是动物（Animals）、水果（Fruits）、交通工具（Vehicles）、电子产品（Electronics）、发型（Hairstyles）。其中，动物和水果属于自然产物，交通工具和电子产品属于人造物，发型属于抽象概念。每个超类分别包含A: 50, F: 50, V: 50, E: 50, H: 30 个类别，总计230个类别。对于每个超类（super-class），分别设计了A: 123, F: 58, V: 81, E: 75, H: 22 个属性，共359个属性。每张图片只包含一个前景物体，标注了标签和物体包围框。对于每个类别，随机挑选了20张图片进行属性标注。

- 训练集（seen classes）：80%类别 
- 测试集（unseen classes）：20%类别

标注示例图：

![image](https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/dataset.png)


你可以从[这里](https://challenger.ai/dataset/lad2018)下载该数据集。

## 用法

### 数据预处理
提取78,017张图片及相应的标注文件：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

如果想可视化训练过程，请执行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [pre-trained model](https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.11-0.6262.hdf5) 放在 models 目录然后执行:

```bash
$ python demo.py
```
