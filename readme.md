## CelebA VAE 人脸生成

本项目使用 PyTorch 训练卷积 VAE，对 CelebA 人脸进行重建和随机生成。

### 数据集

下载 CelebA 数据集后，将对齐人脸图片放到：

`F:\img_align_celeba\img_align_celeba`

原始下载链接：<https://pan.baidu.com/s/1eSNpdRG>

### CPU 快速运行

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python vae.py
```

当前脚本配置为 CPU 演示模式：使用前 2,000 张图片，训练 20 轮，自动保存第 5、10、15、20 轮的效果图。

### 输出

- `recon_epochN.png`：原图与重建图对比
- `sample_epochN.png`：随机潜变量生成的人脸
- `vae_epochN.pth`：模型检查点（本地生成，未提交到仓库）

`epoch0` 图片表示模型随机初始化、尚未进行反向传播时的效果。

## ACT 论文与参考代码

- 论文：`docs/act_paper.pdf`
- 官方代码说明：`docs/act_official_readme.md`
- 官方参考代码：`references/act_official/`

官方工程包含仿真数据采集、训练和评估入口。核心训练命令示例：

```bash
python imitate_episodes.py --task_name sim_transfer_cube_scripted \\
  --ckpt_dir <ckpt_dir> --policy_class ACT --kl_weight 10 \\
  --chunk_size 100 --hidden_dim 512 --batch_size 8 \\
  --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0
```

完整依赖、仿真数据生成和评估步骤见 `references/act_official/README.md`。
