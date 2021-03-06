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


各超类训练结束最佳的验证集准确率和损失为：

|度量|动物|水果|交通工具|电子产品|发型|
|---|---|---|---|---|---|
|ACCURACY|97.222|82.755|92.500|90.790|53.854|
|LOSS|0.025|0.044|0.023|0.051|0.077|

### Demo
下载下列预训练模型放在 models 目录然后执行:
- [动物](https://github.com/foamliu/Zero-Shot-Learning/releases/download/v1.0/BEST_Animals_checkpoint.tar)
- [水果](https://github.com/foamliu/Zero-Shot-Learning/releases/download/v1.0/BEST_Fruits_checkpoint.tar)
- [交通工具](https://github.com/foamliu/Zero-Shot-Learning/releases/download/v1.0/BEST_Vehicles_checkpoint.tar)
- [电子产品](https://github.com/foamliu/Zero-Shot-Learning/releases/download/v1.0/BEST_Electronics_checkpoint.tar)
- [发型](https://github.com/foamliu/Zero-Shot-Learning/releases/download/v1.0/BEST_Hairstyles_checkpoint.tar)

```bash
$ python demo.py -s "Animals"
```

此处超类可以是5个超类中任意一个。

#### 动物

|原图|属性标签|类别|
|---|---|---|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Animals_0.jpg" width="224" />|有柔软的皮肤, 是小的（比猪小）, 是弱小的, 是胆小的, 是群居动物, 是冷血动物, 生活在水里|Label_A_37|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Animals_1.jpg" width="224" />|是无毛的, 有柔软的皮肤, 是小的（比猪小）, 吃浮游生物, 是弱小的, 是胆小的, 是群居动物, 是安静的, 是冷血动物, 是有毒的, 生活在海洋里, 生活在水里|Label_A_36|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Animals_2.jpg" width="224" />|有柔软的皮肤, 有胡须, 是小的（比猪小）, 有牙齿, 有耳朵, 有鼻子, 能游泳, 能跳跃, 吃肉, 是活跃的, 是恒温动物|Label_A_01|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Animals_3.jpg" width="224" />|有柔软的皮肤, 是大的（比猪大）, 有眼睛, 有鼻子, 有脊椎, 能游泳, 吃鱼, 吃肉, 有肌肉, 是友好的, 是活跃的, 是捕食者, 是群居动物, 是安静的|Label_A_42|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Animals_4.jpg" width="224" />|是毛茸茸的, 有胡须, 有长的腿, 有四条腿, 有短脖子, 有牙齿, 有舌头, 有眼睛, 有耳朵, 有鼻子, 有脊椎, 能行走, 行动快速, 是强壮的, 有肌肉, 是活跃的, 是恒温动物, 能产奶, 生活在地面上|Label_A_01|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Animals_5.jpg" width="224" />|是灰色的, 是光滑的, 有舌头, 有眼睛, 有耳朵, 有鼻子, 有脊椎, 能行走, 吃种子, 行动快速, 有肌肉, 是活跃的|Label_A_24|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Animals_6.jpg" width="224" />|有柔软的皮肤, 是群居动物, 是安静的, 是有毒的|Label_A_36|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Animals_7.jpg" width="224" />|有柔软的皮肤, 有羽毛, 是小的（比猪小）, 有短的腿, 是两条腿走路的, 只有两条腿, 有两个胳膊, 有长的翅膀, 有爪子, 有长脖子, 有短尾巴, 有短的喙, 有舌头, 有眼睛, 有耳朵, 有鼻子, 有脊椎, 能游泳, 能行走, 能下蛋, 能捕鱼, 吃植物, 吃昆虫, 吃树叶, 吃种子, 行动缓慢, 是弱小的, 有肌肉, 是友好的, 是胆小的, 是活跃的, 是群居动物, 是吵闹的, 是恒温动物, 生活在平原, 是家养动物|Label_A_27|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Animals_8.jpg" width="224" />|有柔软的皮肤, 是光滑的, 是小的（比猪小）, 只有两条腿, 有爪子, 有短脖子, 有尖的喙, 有舌头, 有耳朵, 有鼻子, 有脊椎, 能飞行, 吃昆虫, 吃树叶, 吃肉, 是弱小的, 有肌肉, 是友好的, 是吵闹的, 是恒温动物|Label_A_30|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Animals_9.jpg" width="224" />|是毛茸茸的, 有柔软的皮肤, 有长的腿, 有短脖子, 有牙齿, 有舌头, 有眼睛, 有耳朵, 吃肉, 能产奶, 生活在地面上|Label_A_22|

#### 水果

|原图|属性标签|类别|
|---|---|---|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Fruits_0.jpg" width="224" />|味道是甜的|Label_A_14|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Fruits_1.jpg" width="224" />|是红色的, 味道是甜的, 是原生的（未被加工过的）, 是常见的, 可直接食用的（不需要加工）, 含水量高, 有果核|Label_A_47|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Fruits_2.jpg" width="224" />|是黄色的, 味道是甜的, 味道是酸的, 是完整的, 是原生的（未被加工过的）, 是常见的, 含水量高, 有果核, 产于热带, 长在树上|Label_A_27|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Fruits_3.jpg" width="224" />|是红色的, 是小的（比苹果小）, 味道是甜的, 是完整的, 是软的, 是常见的, 可直接食用的（不需要加工）, 含水量高, 长在树上|Label_A_10|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Fruits_4.jpg" width="224" />|是红色的, 味道是甜的, 有皮, 可直接食用的（不需要加工）, 有果核|Label_A_11|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Fruits_5.jpg" width="224" />|是红色的, 味道是甜的, 是原生的（未被加工过的）, 是软的, 是常见的, 含水量高, 有果核|Label_A_10|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Fruits_6.jpg" width="224" />|味道是甜的, 是光滑的, 是完整的|Label_A_44|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Fruits_7.jpg" width="224" />|是红色的, 味道是甜的, 味道是酸的, 是完整的, 是原生的（未被加工过的）, 是软的, 是常见的, 可直接食用的（不需要加工）, 含水量高, 长在树上|Label_A_33|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Fruits_8.jpg" width="224" />|味道是甜的|Label_A_46|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Fruits_9.jpg" width="224" />|是红色的, 味道是甜的, 是完整的, 是原生的（未被加工过的）, 是常见的, 含水量高|Label_A_47|

#### 交通工具

|原图|属性标签|类别|
|---|---|---|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Vehicles_0.jpg" width="224" />|是长的, 有座位, 有发动机, 有喇叭, 有方向盘, 有刹车, 有照明灯, 能以移动, 能被驾驶, 是安全的, 是金属造的, 是塑料造的, 是贵的|Label_A_13|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Vehicles_1.jpg" width="224" />|是大的（比汽车大）, 有数吨重, 是长的, 有门, 有座位, 有窗户, 有发动机, 有喇叭, 有照明灯, 能以移动, 能被驾驶, 能载少量(≤10)乘客, 是金属造的, 是塑料造的, 是贵的|Label_A_13|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Vehicles_2.jpg" width="224" />|是大的（比汽车大）, 是长的, 有门, 有窗户, 有发动机, 有方向盘, 有照明灯, 能以移动, 能被驾驶, 可用于民用, 是安全的, 是金属造的, 是塑料造的, 是贵的|Label_A_12|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Vehicles_3.jpg" width="224" />|有方向盘, 能以移动, 能被驾驶, 可用于民用, 可在城市道路上使用, 是危险的, 是金属造的, 是贵的|Label_A_42|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Vehicles_4.jpg" width="224" />|是长的, 有门, 有座位, 有窗户, 有发动机, 有照明灯, 能以移动, 能被驾驶, 能载少量(≤10)乘客, 是金属造的, 是塑料造的, 是贵的|Label_A_13|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Vehicles_5.jpg" width="224" />|是白色的, 有数吨重, 是长的, 速度快, 有门, 有座位, 有窗户, 有发动机, 有喇叭, 有方向盘, 有刹车, 有牌照, 有四个轮子, 有照明灯, 能以移动, 能被驾驶, 能载少量(≤10)乘客, 可用于救援, 可在城市道路上使用, 可在乡村道路上使用, 是安全的, 是金属造的, 是塑料造的, 是贵的|Label_A_13|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Vehicles_6.jpg" width="224" />|是长的, 有门, 有座位, 有窗户, 有发动机, 能以移动, 能被驾驶, 可用于民用, 是金属造的, 是塑料造的, 是贵的|Label_A_35|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Vehicles_7.jpg" width="224" />|有数吨重, 是长的, 有门, 有座位, 有窗户, 有发动机, 有刹车, 有照明灯, 能以移动, 能被驾驶, 是金属造的, 是塑料造的, 是贵的|Label_A_48|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Vehicles_8.jpg" width="224" />|是白色的, 有数吨重, 是长的, 速度快, 有座位, 有窗户, 有喇叭, 有照明灯, 能以移动, 能被驾驶, 能载少量(≤10)乘客, 可用于救援, 可在乡村道路上使用, 是安全的, 是金属造的, 是塑料造的, 是贵的|Label_A_16|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Vehicles_9.jpg" width="224" />|是危险的, 是塑料造的|Label_A_31|

#### 电子产品

|原图|属性标签|类别|
|---|---|---|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Electronics_0.jpg" width="224" />|可用于商业, 可用于家庭使用, 可用于个人使用, 是安全的, 是金属造的, 是塑料造的, 是常见的|Label_A_05|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Electronics_1.jpg" width="224" />|是黑色的, 是大的（比手机大）, 可用于商业, 可用于办公, 可用于展示, 是安全的, 是金属造的, 是塑料造的, 是常见的|Label_A_02|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Electronics_2.jpg" width="224" />|可用于展示, 是安全的, 是金属造的, 是塑料造的, 是常见的|Label_A_43|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Electronics_3.jpg" width="224" />|可用于商业, 可用于家庭使用, 可用于个人使用, 是安全的, 是金属造的, 是塑料造的, 是常见的, 是可被随时移动的, 是小功率设备|Label_A_16|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Electronics_4.jpg" width="224" />|有数千克重, 有（电源）插头, 是安全的, 是金属造的, 是塑料造的, 是常见的|Label_A_02|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Electronics_5.jpg" width="224" />|是黑色的, 有（电源）插头, 有指示灯, 可用于商业, 可用于办公, 可用于安防, 是安全的, 是金属造的, 是塑料造的, 是常见的, 是安静的, 是小功率设备|Label_A_02|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Electronics_6.jpg" width="224" />|是小的（比手机小）, 有腕带, 可用于商业, 可用于家庭使用, 可用于办公, 可用于展示, 可用于个人使用, 是金属造的, 是塑料造的, 是常见的, 是可被随时移动的, 是安静的, 是可以手持的|Label_A_47|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Electronics_7.jpg" width="224" />|是大的（比手机大）, 可用于家庭使用, 是安全的, 是金属造的, 是塑料造的, 是常见的, 是安静的|Label_A_40|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Electronics_8.jpg" width="224" />|有指示灯, 可用于商业, 可用于展示, 是安全的, 是金属造的, 是塑料造的, 是常见的|Label_A_02|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Electronics_9.jpg" width="224" />|可用于商业, 可用于家庭使用, 可用于办公, 可用于个人使用, 是安全的, 是金属造的, 是塑料造的, 是常见的, 是小功率设备|Label_A_18|

#### 发型

|原图|属性标签|类别|
|---|---|---|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Hairstyles_0.jpg" width="224" />|适合皮肤白皙的人, 适合尖尖鼻子的人, 适合化浓妆的人, 容易变乱, 是简约的, 是否优雅|Label_A_22|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Hairstyles_1.jpg" width="224" />|适合小眼睛的人, 适合尖尖鼻子的人|Label_A_29|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Hairstyles_2.jpg" width="224" />|适合小眼睛的人, 适合皮肤白皙的人, 适合尖尖鼻子的人, 适合戴帽子的人, 适合戴耳环的人, 适合化浓妆的人, 容易变乱, 是简约的, 是否优雅, 是否性感, 是否纯净, 是否活泼|Label_A_12|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Hairstyles_3.jpg" width="224" />|适合小眼睛的人, 适合皮肤白皙的人, 适合尖尖鼻子的人, 适合化浓妆的人, 容易变乱|Label_A_09|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Hairstyles_4.jpg" width="224" />|适合小眼睛的人, 适合皮肤白皙的人, 适合尖尖鼻子的人, 适合化浓妆的人, 容易变乱, 是否优雅, 是否活泼|Label_A_09|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Hairstyles_5.jpg" width="224" />|适合皮肤白皙的人, 适合尖尖鼻子的人, 适合化浓妆的人|Label_A_22|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Hairstyles_6.jpg" width="224" />|适合小眼睛的人, 适合皮肤白皙的人, 适合尖尖鼻子的人|Label_A_15|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Hairstyles_7.jpg" width="224" />|适合皮肤白皙的人, 适合尖尖鼻子的人, 适合化浓妆的人, 是否性感|Label_A_24|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Hairstyles_8.jpg" width="224" />|适合小眼睛的人, 适合皮肤白皙的人, 适合尖尖鼻子的人, 适合化浓妆的人, 容易变乱, 是否优雅, 是否活泼|Label_A_12|
|<img src="https://github.com/foamliu/Zero-Shot-Learning/raw/master/images/image_Hairstyles_9.jpg" width="224" />|适合小眼睛的人, 适合皮肤白皙的人, 适合尖尖鼻子的人, 是否优雅|Label_A_30|

