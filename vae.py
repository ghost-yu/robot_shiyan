"""
============================================================
 vae.py —— 带详细中文注释版（VAE 人脸生成训练）
============================================================
【这个文件干什么】
  用 CelebA 人脸数据集训练一个 VAE（变分自编码器）：
  1. 学会把一张 64x64 人脸压缩成 128 维"密码"（编码器）
  2. 学会从"密码"还原出人脸（解码器）
  3. 训练完成后，随便编一组随机密码，就能凭空生成新的人脸

【本版已按你的机器适配】
  - 数据路径：F:\\img_align_celeba\\img_align_celeba（55,134 张人脸）
  - 显卡：RTX 4060（8GB 显存），batch_size 降到 512 更稳
  - 训练轮数：40 轮（约 10~15 分钟出效果）
  - 想更快/更慢：改下方 num_epochs；想换更小数据集：设 MAX_IMAGES

【怎么运行】
  在项目目录打开命令行，执行：  python vae.py
  结果会存到 ./vae_checkpoints 文件夹：
    - recon_epoch5.png   原图(上排) vs 重建图(下排)，看"画得像不像"
    - sample_epoch5.png  纯随机密码生成的新人脸，看"会不会无中生有"
    - vae_epoch5.pth     模型存档，可断点续训
============================================================
"""

# ---------- 第 1 部分：导入需要用到的库 ----------
import os                      # 文件和文件夹路径操作（建目录、拼路径）
from datetime import datetime  # 获取当前时间，打印日志时带时间戳
import glob                    # 用通配符查找文件（找出所有 *.jpg）
from PIL import Image          # 打开 JPG 图片
import torch                   # PyTorch 主库：张量计算、GPU 加速
import torch.nn as nn          # 神经网络层（卷积、全连接等都在这）
import torch.nn.functional as F  # 函数式接口（算损失用 F.mse_loss）
from torch.utils.data import DataLoader, Dataset  # 数据加载两大件
from torchvision import transforms, utils         # 图片预处理 + 存图

# ========== 路径设置 ==========
# 你的数据：F 盘 img_align_celeba 文件夹（里面是 55,134 张对齐裁剪后的人脸）
# r"..." 表示原始字符串：\ 不会被当成转义符（比如 \i 不会出错）
data_root = r"F:\img_align_celeba\img_align_celeba"

# 训练结果（检查点 + 效果图）保存到当前目录下的 vae_checkpoints 文件夹
save_dir = "./vae_checkpoints"
os.makedirs(save_dir, exist_ok=True)  # 文件夹不存在就创建；已存在也不报错

# ========== 超参数（可随意调整，注释说明用途） ==========
batch_size = 32        # CPU 快速演示使用较小批次
lr = 2e-4              # 学习率 = 0.0002，参数每次更新的步长
num_epochs = 20        # 从已有第 1 轮检查点继续训练到第 20 轮
latent_dim = 128       # 潜变量 z 的维度（"密码"的长度）
sample_every = 5       # 每 5 轮保存一次效果图 + 模型存档
num_sample_images = 8  # 每次保存 8 张示例图
image_size = 64        # 图片统一缩放/裁剪到 64x64
# 可选：只想用前 N 张图训练（进一步提速）。None = 用全部 55,134 张
MAX_IMAGES = 2000      # CPU 快速演示只取前 2000 张

# 自动选择设备：有 NVIDIA 显卡用 cuda（GPU），没有就用 cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据预处理流程（每张图进来都按顺序走这三步） ==========
transform = transforms.Compose([
    transforms.Resize(image_size),      # 1. 缩放到 64x64
    transforms.CenterCrop(image_size),  # 2. 从中心裁 64x64（去掉多余背景）
    transforms.ToTensor(),              # 3. 转成 PyTorch 张量：0~1 小数，形状 (3,64,64)
])


# ========== 第 2 部分：数据集类（负责"单张图怎么取"） ==========
class CelebADataset(Dataset):
    """
    继承 PyTorch 的 Dataset，实现三个魔法方法后，
    DataLoader 就知道怎么批量取数据了。
    """

    def __init__(self, root, transform=None):
        # 创建对象时自动执行一次：扫描文件夹里所有 jpg，记住路径列表
        self.root = root
        # glob.glob 找出所有 .jpg 路径；sorted 排序让顺序稳定
        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        if MAX_IMAGES is not None:          # 如果设了采样上限
            self.paths = self.paths[:MAX_IMAGES]  # 只保留前 N 张
        self.transform = transform          # 记住预处理流程

    def __len__(self):
        # 返回图片总数。DataLoader 靠它知道一共要取多少张
        return len(self.paths)

    def __getitem__(self, idx):
        # 核心方法：按编号取一张图，处理成标准张量
        img_path = self.paths[idx]          # 拿到第 idx 张图的路径
        img = Image.open(img_path).convert("RGB")  # 打开并强制转 RGB 三通道
        if self.transform:                  # 如果有预处理流程就执行
            img = self.transform(img)       # 得到形状 (3,64,64) 的张量
        return img                          # 交出去


# 创建数据集对象（这里只是记住路径，不把图片读进内存）
dataset = CelebADataset(root=data_root, transform=transform)

# 创建 DataLoader：负责批量、打乱、并行读图
loader = DataLoader(
    dataset,
    batch_size=batch_size,  # 每批打包 512 张
    shuffle=True,           # 每轮打乱顺序，防止模型"背顺序"
    num_workers=0,          # CPU/Windows 演示使用主进程读图
    pin_memory=True         # GPU 训练时加速数据搬运（配合 cuda 使用）
)


# ========== 第 3 部分：模型（编码器 + 解码器） ==========
class ConvVAE(nn.Module):
    """
    卷积 VAE 模型。nn.Module 是 PyTorch 的模型基类，
    自动帮你管理参数、算梯度、搬到 GPU 等。
    """

    def __init__(self, latent_dim=128, ch=64, image_size=64):
        super().__init__()          # 必须调用父类初始化
        self.latent_dim = latent_dim  # 密码长度（128）
        self.ch = ch                  # 基础通道数（64），每层卷积从这里翻倍
        self.image_size = image_size  # 输入图片边长（64）

        # ---------- 编码器：图像 → 特征（尺寸减半、通道翻倍） ----------
        # nn.Sequential：把层按顺序串起来，数据一层层往下传
        # nn.Conv2d(输入通道, 输出通道, 卷积核4x4, 步长2, 填充1)
        #   步长=2 → 每层图缩小一半；通道翻倍 → 提取更高级的特征
        #   尺寸变化：64→32→16→8→4，通道：3→64→128→256→512
        # nn.ReLU：激活函数，把负数变 0，给网络非线性能力
        # nn.BatchNorm2d：批归一化，让训练更稳更快
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1),       # 层1：3→64 通道，64→32
            nn.ReLU(True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1),  # 层2：64→128，32→16
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),  # 层3：128→256，16→8
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),  # 层4：256→512，8→4
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True)
        )

        # 动态计算"展平后的特征维度"：让网络自己跑一遍报出结果，
        # 这样以后改 image_size 或层数都不用手算
        with torch.no_grad():               # 这段只量尺寸，不记录梯度
            dummy = torch.zeros(1, 3, image_size, image_size)  # 造一张假图
            feat_dim = self.enc(dummy).view(1, -1).size(1)     # 4x4x512=8192

        # 三个全连接层（nn.Linear：把一维向量做加权求和）
        self.fc_mu = nn.Linear(feat_dim, latent_dim)     # 8192 → 128：潜变量均值 μ
        self.fc_logvar = nn.Linear(feat_dim, latent_dim) # 8192 → 128：方差的对数 logσ²
        self.fc_dec = nn.Linear(latent_dim, feat_dim)    # 128 → 8192：解码器入口

        # 记录特征图形状 (512, 4, 4)，解码时把向量"捏回"这个形状
        self._feat_shape = self.enc(dummy).shape[1:]

        # ---------- 解码器：特征 → 图像（尺寸翻倍、通道减半） ----------
        # nn.ConvTranspose2d：转置卷积（反卷积），把图放大回去
        #   尺寸变化：4→8→16→32→64，通道：512→256→128→64→3
        # 最后一层用 nn.Sigmoid：把输出压到 0~1（图片像素值范围）
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1),  # 512→256
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1),  # 256→128
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1),      # 128→64
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),           # 64→3
            nn.Sigmoid()   # 输出像素值保证在 0~1
        )

    def encode(self, x):
        """编码：图像 → μ（中心）和 logσ²（模糊度）"""
        h = self.enc(x)                     # 4 层卷积 → (N, 512, 4, 4)
        h = h.view(h.size(0), -1)           # 展平 → (N, 8192)
        mu = self.fc_mu(h)                  # 全连接 → μ (N, 128)
        logvar = self.fc_logvar(h)          # 全连接 → logσ² (N, 128)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化（VAE 灵魂）：z = μ + ε·σ
        把随机性交给外部的 ε，让 μ、σ 保持可导，梯度才能回流训练"""
        std = torch.exp(0.5 * logvar)       # exp(0.5*logσ²) = σ（标准差）
        eps = torch.randn_like(std)         # 生成标准正态随机数 ε
        return mu + eps * std               # 采样结果 z (N, 128)

    def decode(self, z):
        """解码：密码 z → 重建图像"""
        h = self.fc_dec(z)                          # 128 → 8192
        h = h.view(h.size(0), *self._feat_shape)    # 捏回 (N, 512, 4, 4)
        x_recon = self.dec(h)                       # 4 层转置卷积 → (N, 3, 64, 64)
        return x_recon

    def forward(self, x):
        """前向入口：写 model(imgs) 时自动调用。编码→采样→解码"""
        mu, logvar = self.encode(x)          # ① 压缩
        z = self.reparameterize(mu, logvar)  # ② 采样
        x_recon = self.decode(z)             # ③ 还原
        return x_recon, mu, logvar           # 返回重建图 + 潜分布参数


# ========== 第 4 部分：损失函数 ==========
def vae_loss(recon_x, x, mu, logvar):
    """
    总损失 = 重建损失 + KL 散度
      - 重建损失：管"画得像不像"（x̂ 和原图 x 逐像素比）
      - KL 散度：管"密码规不规律"（逼潜分布贴近标准正态 N(0,1)）
    """
    # MSE 重建损失：像素差平方的总和（sum 表示全部加总）
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL 散度公式（已化简）：
    #   -0.5 * Σ(1 + logσ² - μ² - σ²)
    # 效果：把 μ 推向 0、把 σ² 推向 1
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 返回：总损失、重建损失、KL（后两个用于打印观察）
    return recon_loss + kld, recon_loss, kld


# ========== 第 5 部分：工具函数 ==========
def save_checkpoint(model, optim, epoch, path):
    """存档：模型参数 + 优化器状态 + 轮数，用于断点续训"""
    state = {
        "epoch": epoch,                    # 训练到第几轮
        "model_state": model.state_dict(),  # 模型全部参数
        "optim_state": optim.state_dict()   # 优化器状态（动量等）
    }
    torch.save(state, path)                 # 存成 .pth 文件


def save_image_grid(tensor, filename, nrow=8):
    """把一批小图拼成网格保存为 PNG"""
    tensor = torch.clamp(tensor, 0, 1)      # 数值硬裁到 0~1，防止杂点
    utils.save_image(tensor, filename, nrow=nrow, padding=2)  # 拼网格存图


# ========== 第 6 部分：训练主循环 ==========
def train():
    # 创建模型并搬到 GPU/CPU
    model = ConvVAE(latent_dim=latent_dim, image_size=image_size).to(device)
    # Adam 优化器：自动为每个参数调节步长
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    resume_path = os.path.join(save_dir, "vae_epoch1.pth")
    start_epoch = 1
    if os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optim.load_state_dict(state["optim_state"])
        start_epoch = int(state["epoch"]) + 1
        print(f"Resuming from epoch {state['epoch']}")
    global_step = 0  # 全局步数计数器（本代码仅累加，备用）

    # 外层循环：一个 epoch = 把整个数据集过一遍
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()          # 切到训练模式（BatchNorm 用当前批统计）
        epoch_loss = 0.0       # 本轮总损失累加器
        epoch_recon = 0.0      # 本轮重建损失累加器
        epoch_kld = 0.0        # 本轮 KL 累加器

        # 内层循环：一次拿一批图
        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)  # 搬到 GPU
            optim.zero_grad()                          # 清空上次梯度（必须！）
            recon_imgs, mu, logvar = model(imgs)       # 前向传播
            loss, recon_l, kld = vae_loss(recon_imgs, imgs, mu, logvar)  # 算损失
            loss.backward()                            # 反向传播：算梯度
            optim.step()                               # 更新参数

            # 累加本轮的三个损失
            epoch_loss += loss.item()
            epoch_recon += recon_l.item()
            epoch_kld += kld.item()
            global_step += 1

            # 每 100 批打印一次实时日志，方便观察训练进度
            if batch_idx % 100 == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Epoch {epoch}/{num_epochs} Batch {batch_idx}/{len(loader)} "
                      f"Loss {loss.item():.4f} "
                      f"(recon {recon_l.item():.4f}, kld {kld.item():.4f})")

        # 本轮结束：打印平均损失（除以图片总数，即"每张图的平均损失"）
        n_samples = len(loader.dataset)
        print(f"=== Epoch {epoch} finished. Avg loss: {epoch_loss / n_samples:.4f} "
              f"(recon {epoch_recon / n_samples:.4f}, kld {epoch_kld / n_samples:.4f}) ===")

        # 每 sample_every 轮（以及第 1 轮）保存效果
        if epoch % sample_every == 0 or epoch == 1:
            # ① 保存检查点
            ckpt_path = os.path.join(save_dir, f"vae_epoch{epoch}.pth")
            save_checkpoint(model, optim, epoch, ckpt_path)

            # ② 评估模式 + 关梯度（只出图，不训练）
            model.eval()
            with torch.no_grad():
                # 重建演示：拿 8 张原图和它们的重建图拼一起
                # 上排 = 原图，下排 = 重建图，一眼看出"画得像不像"
                imgs = next(iter(loader))               # 重新取一批
                imgs = imgs.to(device)[:num_sample_images]  # 只要前 8 张
                recon_imgs, _, _ = model(imgs)          # 重建
                combined = torch.cat([imgs, recon_imgs], dim=0)  # 拼成 16 张
                save_image_grid(combined,
                                os.path.join(save_dir, f"recon_epoch{epoch}.png"),
                                nrow=8)                 # 每行 8 张 → 两行

                # 生成演示：纯随机密码 → 解码器 → 凭空画 8 张新脸！
                z = torch.randn(num_sample_images, latent_dim).to(device)  # 随机 z
                samples = model.decode(z)               # 解码
                save_image_grid(samples,
                                os.path.join(save_dir, f"sample_epoch{epoch}.png"),
                                nrow=8)

            model.train()   # ③ 别忘了切回训练模式！

    print("Training complete.")


# ========== 主入口 ==========
# 只有"直接运行本文件"时才执行；被 import 时不执行（Windows 多进程也必须靠它保护）
if __name__ == "__main__":
    print("Starting training on device:", device)
    print("Dataset size:", len(dataset))
    train()
