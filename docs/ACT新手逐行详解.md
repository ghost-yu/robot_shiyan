# ACT 新手逐行学习指南

这份文档面向 Python 基础学习者。官方源码保存在
`references/act_official_source/`，本文件不替换官方代码，方便你一边看原代码一边对照说明。

## 1. 先理解整体流程

```text
示范数据 HDF5
    |
    v
utils.py 读取图像、关节位置、动作，并做归一化
    |
    v
imitate_episodes.py 创建训练集、验证集和 ACTPolicy
    |
    v
policy.py 把观测送入 ACT 模型，预测未来一段动作
    |
    v
detr/models/detr_vae.py
视觉特征 + 机器人状态 + CVAE 潜变量 -> Transformer -> action chunk
    |
    v
训练时比较预测动作与示范动作，反向传播更新参数
```

ACT 的关键不是只预测下一步动作，而是一次预测连续的 `chunk_size` 个动作。
例如 `chunk_size=100` 时，模型收到当前观察后，会输出未来 100 个动作。

## 2. 文件职责表

| 文件 | 新手需要知道的内容 |
|---|---|
| `imitate_episodes.py` | 程序入口，解析命令行参数，训练和评估循环 |
| `policy.py` | 把通用训练循环和 ACT 模型连接起来 |
| `detr/models/detr_vae.py` | ACT 的 CVAE、视觉编码器和 Transformer |
| `utils.py` | 读取 HDF5 示范数据，切分训练集/验证集 |
| `record_sim_episodes.py` | 用仿真脚本生成示范数据 |
| `sim_env.py` | MuJoCo/DM Control 仿真环境 |
| `constants.py` | 任务名称、相机名称、数据路径等配置 |

## 3. `imitate_episodes.py`：训练入口

打开 `references/act_official_source/imitate_episodes.py`，重点按下面顺序阅读。

### 3.1 导入模块

文件开头的 `import` 语句把功能分成四类：

1. `torch`、`numpy`：张量计算和数值处理。
2. `argparse`、`os`：读取命令行参数和操作路径。
3. `utils`：读取 HDF5 数据并生成 DataLoader。
4. `policy`：创建 `ACTPolicy`，其中包含真正的神经网络。

Python 的 `import` 不会执行训练，它只是把另一个文件的函数和类引入当前文件。

### 3.2 `train_bc()` 函数

`train_bc(train_dataloader, val_dataloader, config)` 是训练主函数。

- `train_dataloader`：训练数据，每次返回一批观测和动作。
- `val_dataloader`：验证数据，只计算损失，不更新参数。
- `config`：字典，保存 batch size、学习率、训练轮数等配置。

函数首先调用 `make_policy(config)` 创建策略。这里的“策略”不是一条规则，
而是一个可以学习的神经网络。

随后创建 Adam 优化器：

```python
optimizer = optim.AdamW(policy.parameters(), lr=lr)
```

- `policy.parameters()` 返回模型所有可训练参数。
- `lr` 是学习率，决定每次更新幅度。
- `AdamW` 根据梯度调整参数。

每个 epoch 做两件事：

1. 训练阶段：`policy.train()`，计算 loss，执行 `loss.backward()` 和 `optimizer.step()`。
2. 验证阶段：`policy.eval()`，使用 `torch.no_grad()`，只计算 loss，不改变参数。

### 3.3 一批数据的训练过程

训练循环可以按这 6 行理解：

```python
optimizer.zero_grad()       # 清除上一批留下的梯度
loss = policy(qpos, image, action)  # 前向计算损失
loss.backward()             # 根据损失计算每个参数的梯度
optimizer.step()            # 用梯度更新参数
```

ACT 训练时传入 `action`，因为 CVAE 的训练编码器需要看到示范动作。
评估时不传 `action`，模型只能根据当前观测生成动作。

### 3.4 命令行参数

`argparse.ArgumentParser()` 定义命令行参数。例如：

- `--task_name`：仿真任务名称。
- `--ckpt_dir`：模型检查点保存目录。
- `--policy_class ACT`：选择 ACT，而不是普通 CNN-MLP。
- `--chunk_size 100`：一次预测 100 步动作。
- `--kl_weight 10`：CVAE KL 散度损失的权重。
- `--num_epochs 2000`：训练轮数。

命令行参数最后会进入 `config` 字典，再传给数据加载器和策略模型。

## 4. `policy.py`：ACT 策略适配器

### 4.1 `ACTPolicy.__init__`

构造函数保存任务配置，并创建 `ACTModel`：

```python
self.model = ACT(...)
```

这里的 `ACT` 来自 `detr.models.detr_vae`。`ACTPolicy` 的作用是适配训练框架，
让外层代码不必知道 Transformer 的内部细节。

### 4.2 `ACTPolicy.__call__`

调用 `policy(qpos, image, actions)` 时，代码通常分两条路径：

- `actions is not None`：训练路径，返回动作重建损失和 KL 损失。
- `actions is None`：推理路径，返回预测的动作 chunk。

这是 Python 中常见的“同一个函数同时支持训练和推理”写法。

### 4.3 归一化与反归一化

机器人关节角度和动作的数值范围可能很大。`policy.py` 使用数据集统计量把它们
归一化到接近 `[-1, 1]`。模型输出后再反归一化，恢复成机器人可以执行的单位。

如果忘记反归一化，模型输出的数字不能直接发给真实机器人。

## 5. `detr/models/detr_vae.py`：ACT 核心

### 5.1 输入张量

典型输入形状如下：

```text
qpos   [batch, state_dim]
image  [batch, camera_num, 3, height, width]
action [batch, chunk_size, action_dim]
```

例如 batch size 为 8、3 个相机、图像 480x640、chunk size 为 100：

```text
image  [8, 3, 3, 480, 640]
action [8, 100, 14]
```

第一维永远是 batch。第二维是时间序列长度。最后一维是单步动作的维度。

### 5.2 CNN 视觉编码器

ResNet 或 Backbone 把每张图片转换成视觉特征。图片的高和宽会逐层缩小，
通道数增加。最终得到的不是像素，而是包含物体位置、颜色和形状信息的特征向量。

代码中的 `torchvision.models.resnet18` 等调用只负责提取特征，
不负责生成动作。

### 5.3 CVAE 编码器

训练时，CVAE 编码器同时看到当前观测和示范动作：

```text
(observation, target action chunk) -> mu, logvar
```

`mu` 是潜变量均值，`logvar` 是方差的对数。然后使用重参数化技巧：

```python
std = torch.exp(0.5 * logvar)
z = mu + std * torch.randn_like(std)
```

这样随机采样仍然可以反向传播。

### 5.4 Transformer 解码器

解码器输入当前观测、潜变量 `z` 和一组位置编码，输出整个动作序列：

```text
(image feature, qpos, z) -> [action_1, action_2, ..., action_T]
```

Transformer 的 self-attention 可以让不同时间步相互参考，因此输出的动作 chunk
通常比逐步独立预测更加平滑。

### 5.5 损失函数

ACT 的训练损失可以概括为：

```text
总损失 = 动作重建损失 + kl_weight * KL 散度
```

- 动作重建损失：预测动作与示范动作的差异，通常是 L1 或 L2。
- KL 散度：让潜变量分布接近标准正态分布，避免潜空间失控。

`kl_weight` 太大时，模型可能忽略动作细节；太小时，潜变量分布可能不稳定。

## 6. `utils.py`：数据读取

### 6.1 HDF5 数据结构

官方数据通常包含：

```text
episode_0.hdf5
  /action                 [time, action_dim]
  /observations/qpos      [time, state_dim]
  /observations/qvel      [time, state_dim]
  /observations/images/<camera_name> [time, height, width, 3]
```

`EpisodicDataset.__getitem__` 根据一个随机索引读取一整个 episode 的某个时间片，
再取出当前观察和后续 `chunk_size` 个动作。

如果 episode 剩余动作不足一个 chunk，代码会用最后一个动作补齐，
同时使用 padding mask 告诉损失函数哪些位置是真实数据。

### 6.2 训练/验证切分

`load_data()` 先打乱 episode 编号，再按 80%/20% 切分训练集和验证集。
注意切分单位是 episode，不是单张图片，避免同一段动作同时出现在训练和验证中。

## 7. 建议的新手阅读顺序

1. 先读本文件第 1、2 节，记住数据流。
2. 阅读 `utils.py` 的 `EpisodicDataset.__getitem__`。
3. 阅读 `policy.py` 的 `ACTPolicy.__call__`。
4. 阅读 `detr/models/detr_vae.py` 的模型构造和 `forward`。
5. 最后阅读 `imitate_episodes.py` 的训练循环。

不要一开始就读 `sim_env.py` 和所有 DETR 工具文件；它们属于仿真和通用辅助代码。

## 8. 官方最小实验流程

```bash
# 生成 50 个仿真示范 episode
python record_sim_episodes.py --task_name sim_transfer_cube_scripted \
  --dataset_dir <数据目录> --num_episodes 50

# 训练 ACT
python imitate_episodes.py --task_name sim_transfer_cube_scripted \
  --ckpt_dir <模型目录> --policy_class ACT --kl_weight 10 \
  --chunk_size 100 --hidden_dim 512 --batch_size 8 \
  --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0

# 评估模型
python imitate_episodes.py --eval --task_name sim_transfer_cube_scripted \
  --ckpt_dir <模型目录> --policy_class ACT --chunk_size 100
```

官方代码依赖 Python 3.8、MuJoCo、DM Control、PyTorch、OpenCV、h5py 等，
不适合直接放进当前 VAE 的 Python 3.11 虚拟环境。建议为 ACT 单独创建 conda 环境。
