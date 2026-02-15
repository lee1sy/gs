import torch
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # 服务器专用，不弹窗
import matplotlib.pyplot as plt
from torchvision import transforms

# ================= 🔧 必须确认的配置 =================
DATA_ROOT = "/mnt/nuscenes/"
INFO_PATH = "/mnt/nuscenes/nuscenes_infos_bs.pkl"
DATABASE_PATH = "/mnt/nuscenes/bs_db.npy"
QUERY_PATH = "/mnt/nuscenes/bs_train_query.npy"

# 🔥 这里指向你【新生成】的数据路径
GAUSSIAN_PATH = "/home/james/LSY/11/nuscenes/"

RESIZE = (448, 800)
OUTPUT_DIR = "verify_new_data_results"
NUM_SAMPLES = 10 # 检查 10 张图

# 引入 Dataset
try:
    from dataset.NuScenesDataset import TripletDataset
except ImportError:
    sys.path.append(os.getcwd())
    from dataset.NuScenesDataset import TripletDataset

def check_new_data():
    print(f"🚀 开始验证新生成的 .npy 数据 (去地面版)...")
    print(f"📂 数据源: {GAUSSIAN_PATH}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. 检查目录下是否有文件
    if not os.path.exists(GAUSSIAN_PATH):
        print(f"❌ 错误: 路径不存在 {GAUSSIAN_PATH}")
        return
    
    files = os.listdir(GAUSSIAN_PATH)
    npy_files = [f for f in files if f.endswith('.npy')]
    print(f"📊 目录下发现 {len(npy_files)} 个 .npy 文件")
    if len(npy_files) == 0:
        print("❌ 错误: 目录下没有 .npy 文件！请检查生成脚本是否运行成功。")
        return

    # 2. 初始化 Dataset
    img_transforms = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("⏳ 初始化 Dataset...")
    dataset = TripletDataset(
        data_root_dir=DATA_ROOT,
        database_path=DATABASE_PATH,
        query_path=QUERY_PATH,
        info_path=INFO_PATH,
        cache_dir="./",         
        img_transforms=img_transforms,
        nNeg=1, nNegSample=1, nonTrivPosDistThres=10, posDistThr=25, margin=0.5,
        gaussian_path=GAUSSIAN_PATH, # 指向新路径
        resize=RESIZE
    )

    # 3. 均匀采样检查
    total_len = len(dataset)
    indices = np.linspace(0, total_len - 1, NUM_SAMPLES, dtype=int)
    
    for i, index in enumerate(indices):
        print(f"[{i+1}/{NUM_SAMPLES}] 检查 Index {index} ...")
        
        try:
            real_index = int(dataset.queries[index][0])
            
            # 使用 Dataset 内部逻辑加载
            data_dict = dataset.load_data_with_matrices(real_index)
            
            img_tensor = data_dict['images'][0]
            points = data_dict['gaussians'] # [4096, 14]
            extrinsic = data_dict['extrinsics'][0]
            intrinsic = data_dict['intrinsics'][0]
            
            # 检查是否全是0
            if points[:, :3].abs().sum() == 0:
                print("   ❌ 警告: 读到了全 0 数据！文件名可能依然不匹配。")
                continue
                
            visualize(img_tensor, points, extrinsic, intrinsic, index, real_index)
            
        except Exception as e:
            print(f"   ❌ 出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ 验证结束！请查看 '{OUTPUT_DIR}' 文件夹。")

def visualize(img_tensor, points, T, K, idx, db_idx):
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img_np = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
    
    # 投影
    xyz = points[:, :3]
    ones = torch.ones(xyz.shape[0], 1)
    xyz_homo = torch.cat([xyz, ones], dim=1)
    
    xyz_cam = (T @ xyz_homo.T).T
    mask_z = xyz_cam[:, 2] > 0.1
    xyz_cam = xyz_cam[mask_z]
    
    uv_homo = (K @ xyz_cam[:, :3].T).T
    u = uv_homo[:, 0] / uv_homo[:, 2]
    v = uv_homo[:, 1] / uv_homo[:, 2]
    
    # 绘图
    plt.figure(figsize=(16, 9))
    plt.imshow(img_np)
    
    H, W = img_np.shape[:2]
    mask_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    u_valid = u[mask_img]
    v_valid = v[mask_img]
    d_valid = xyz_cam[mask_img, 2]
    
    # 🔥 关键：点画大一点 (s=5)，用显眼的颜色
    plt.scatter(u_valid, v_valid, s=5, c=d_valid, cmap='spring', edgecolors='black', linewidth=0.1, alpha=0.9)
    
    # 统计信息
    num_visible = len(u_valid)
    plt.title(f"NEW DATA Check | Idx: {idx} | Visible Points: {num_visible}\nGround Removed? Check if road is empty.", 
              fontsize=14, color='blue', fontweight='bold')
    plt.axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, f"check_{idx:05d}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"   💾 图片已保存: {save_path} (可见点数: {num_visible})")

if __name__ == "__main__":
    check_new_data()