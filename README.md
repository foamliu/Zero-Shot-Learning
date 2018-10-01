# 零样本学习

零样本学习（zero-shot learning）是在已知类别上训练物体识别模型，要求模型能够识别来自未知类别的样本。对图像理解、（从已知类别到未知类别的）知识迁移具有重要意义。

## 依赖
- Python 3.5
- PyTorch 0.4

## 数据集
使用 AI Challenger 2018 的图像属性数据集，本数据集共78,017张图片、230个类别、359种属性。

图片可划分为5个超类（super-class），分别是动物（Animals）、水果（Fruits）、交通工具（Vehicles）、电子产品（Electronics）、发型（Hairstyles）。其中，动物和水果属于自然产物，交通工具和电子产品属于人造物，发型属于抽象概念。每个超类分别包含A: 50, F: 50, V: 50, E: 50, H: 30 个类别，总计230个类别。对于每个超类（super-class），分别设计了A: 123, F: 58, V: 81, E: 75, H: 22 个属性，共359个属性。每张图片只包含一个前景物体，标注了标签和物体包围框。对于每个类别，随机挑选了20张图片进行属性标注。

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
下载 [pre-trained model](https://github.com/foamliu/Zero-Shot-Learning/releases/download/v1.0/model.11-0.6262.hdf5) 放在 models 目录然后执行:

```bash
$ python demo.py
```

|原图|属性标签|类别|
|---|---|---|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_0.jpg" width="224" />|是白色的, 有柔软的皮肤, 有羽毛, 是小的（比猪小）, 有短的腿, 是两条腿走路的, 只有两条腿, 有两个胳膊, 有长的翅膀, 有爪子, 有长脖子, 有短尾巴, 有短的喙, 有舌头, 有眼睛, 有耳朵, 有鼻子, 有脊椎, 能游泳, 能行走, 能下蛋, 能捕鱼, 吃植物, 吃昆虫, 吃鱼, 吃树叶, 吃种子, 行动快速, 是弱小的, 有肌肉, 是友好的, 是胆小的, 是活跃的, 是群居动物, 是吵闹的, 是恒温动物, 生活在水里|Label_A_28|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_1.jpg" width="224" />|是毛茸茸的, 有柔软的皮肤, 有四条腿, 有爪子, 有肉垫, 有短脖子, 有牙齿, 有獠牙, 有舌头, 有眼睛, 有耳朵, 有鼻子, 有脊椎, 能游泳, 能行走, 吃植物, 吃肉, 吃花蜜, 是强壮的, 有肌肉, 是活跃的, 是安静的, 是恒温动物, 能产奶, 生活在地面上|Label_A_13|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_2.jpg" width="224" />|是毛茸茸的, 有柔软的皮肤, 是光滑的, 有胡须, 是小的（比猪小）, 有四条腿, 有肉垫, 有短脖子, 有牙齿, 有舌头, 有眼睛, 有耳朵, 有鼻子, 能行走, 能跳跃, 行动快速, 是活跃的, 是安静的, 是恒温动物, 能产奶, 生活在地面上|Label_A_22|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_3.jpg" width="224" />|是毛茸茸的, 有柔软的皮肤, 有胡须, 是小的（比猪小）, 有长的腿, 有四条腿, 有爪子, 有肉垫, 有短脖子, 有牙齿, 有长尾巴, 有舌头, 有眼睛, 有耳朵, 有鼻子, 有脊椎, 能游泳, 能行走, 能跳跃, 吃肉, 行动快速, 有肌肉, 是活跃的, 是安静的, 是恒温动物, 能产奶, 生活在地面上|Label_A_01|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_4.jpg" width="224" />|是毛茸茸的, 有柔软的皮肤, 有胡须, 有四条腿, 有爪子, 有肉垫, 有短脖子, 有牙齿, 有獠牙, 有舌头, 有眼睛, 有耳朵, 有鼻子, 有脊椎, 能游泳, 能行走, 能跳跃, 吃肉, 行动快速, 是强壮的, 有肌肉, 是聪明的, 是活跃的, 是恒温动物, 能产奶, 生活在地面上, 是家养动物|Label_A_01|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_5.jpg" width="224" />|是毛茸茸的, 有柔软的皮肤, 是小的（比猪小）, 有四条腿, 有爪子, 有肉垫, 有短脖子, 有舌头, 有眼睛, 有耳朵, 有鼻子, 有脊椎, 能行走, 吃植物, 吃昆虫, 吃肉, 行动快速, 有肌肉, 是活跃的, 是恒温动物, 生活在地面上|Label_A_22|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_6.jpg" width="224" />|是无毛的, 有坚韧的皮肤, 有壳, 是小的（比猪小）, 有短的腿, 有短尾巴, 有舌头, 有眼睛, 有鼻子, 有壳, 能下蛋, 吃植物, 吃树叶, 吃浮游生物, 是弱小的, 是友好的, 是胆小的, 会冬眠, 是群居动物, 是安静的, 是冷血动物, 生活在地面上|Label_A_38|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_7.jpg" width="224" />|有柔软的皮肤, 有羽毛, 是小的（比猪小）, 有短的腿, 是两条腿走路的, 只有两条腿, 有两个胳膊, 有长的翅膀, 有爪子, 有长脖子, 有短尾巴, 有短的喙, 有舌头, 有眼睛, 有耳朵, 有鼻子, 有脊椎, 能游泳, 能行走, 能下蛋, 能捕鱼, 吃植物, 吃昆虫, 吃鱼, 吃树叶, 吃种子, 行动快速, 有肌肉, 是胆小的, 是活跃的, 是群居动物, 是吵闹的, 是恒温动物|Label_A_28|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_8.jpg" width="224" />|是毛茸茸的, 有柔软的皮肤, 有胡须, 有长的腿, 有四条腿, 有爪子, 有肉垫, 有牙齿, 有舌头, 有眼睛, 有耳朵, 有鼻子, 有脊椎, 能行走, 能跳跃, 行动快速, 有肌肉, 是活跃的, 是安静的, 是恒温动物, 能产奶, 生活在地面上|Label_A_16|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_9.jpg" width="224" />|是无毛的, 有柔软的皮肤, 是光滑的, 是小的（比猪小）, 有触手, 吃浮游生物, 行动缓慢, 是弱小的, 是胆小的, 是群居动物, 是冷血动物, 是有毒的, 生活在海洋里, 生活在水里|Label_A_37|

