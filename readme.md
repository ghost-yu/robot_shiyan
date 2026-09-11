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
