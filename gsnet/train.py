import os  # 1. 记得导入 os 模块
import torch
from tools.runner import Trainer
from tools.utils import load_config, check_path, check_dir
from modules.GS import GaussianFusionNet
from torch.utils.data import DataLoader
from dataset.NuScenesDataset import TripletDataset, DatabaseQueryDataset, collate_fn
from torchvision.transforms import transforms

def main():
    # 🔥🔥🔥 [修改] 强制只使用前三张显卡 (0, 1, 2) 🔥🔥🔥
    # 这一行必须写在 load_config 和任何 torch 调用之前
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    cfg = load_config('config/config.yaml')

    # ====================parse config====================
    data_root_dir = cfg['data']['data_root_dir']
    database_path = cfg['data']['database_path']
    train_query_path = cfg['data']['train_query_path']
    test_query_path = cfg['data']['test_query_path']
    val_query_path = cfg['data']['val_query_path']
    info_path = cfg['data']['info_path']
    gaussian_path = cfg['data'].get('gaussian_path', None)

    nonTrivPosDistThres = cfg['runner']['nonTrivPosDistThres']
    posDistThr = cfg['runner']['posDistThr']
    
    # 🔥🔥🔥 [优化] 针对 3x4090 显卡 🔥🔥🔥
    # 原来是 10 (针对4卡)，现在只有 3 张卡，建议稍微减小一点 nNeg 或者保持不变
    # 如果显存够大（4090），保持 10 也没问题；如果爆显存，改成 6 或 8
    nNeg = 10  
    
    nNegSample = cfg['runner']['nNegSample']
    margin = cfg['runner']['margin']
    resize = cfg['runner']['resize']
    lr = cfg['runner']['lr']
    step_size = cfg['runner']['step_size']
    gamma = cfg['runner']['gamma']
    num_epochs = cfg['runner']['num_epochs']
    
    # 🔥🔥🔥 [优化] CPU 线程数 🔥🔥🔥
    # 3 张卡，16 个线程依然是可以的，或者稍微降到 12 也可以
    num_workers_train = 16  
    num_workers_test = 16
    
    resume_path = cfg['runner']['resume_path']
    log = cfg['runner']['log']
    resume_scheduler = cfg['runner']['resume_scheduler']

    ckpt_dir = cfg['runner']['ckpt_dir']
    result_dir = cfg['runner']['result_dir']
    cache_dir = cfg['runner']['cache_dir']
    log_dir = cfg['runner']['log_dir']

    # ====================check dirs and paths====================
    check_path(data_root_dir, database_path, train_query_path, test_query_path, val_query_path, info_path)
    check_dir(ckpt_dir, result_dir, cache_dir, log_dir)

    # ==========================dataset===========================

    img_transforms = transforms.Compose([transforms.Resize(resize),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])

    train_set = TripletDataset(data_root_dir, database_path, train_query_path, info_path, cache_dir,
                               img_transforms, nNeg, nNegSample, nonTrivPosDistThres, posDistThr, margin,
                               gaussian_path=gaussian_path, resize=resize)

    # 🔥 注意：batch_size 必须保持为 1 🔥
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers_train)

    whole_train_set = DatabaseQueryDataset(data_root_dir, database_path, train_query_path, info_path,
                                           img_transforms, nonTrivPosDistThres,
                                           gaussian_path=gaussian_path, resize=resize)

    whole_train_loader = DataLoader(dataset=whole_train_set, batch_size=8, shuffle=False,
                                    num_workers=num_workers_test)
    whole_val_set = DatabaseQueryDataset(data_root_dir, database_path, val_query_path, info_path,
                                         img_transforms, nonTrivPosDistThres,
                                         gaussian_path=gaussian_path, resize=resize)
    whole_val_loader = DataLoader(dataset=whole_val_set, batch_size=8, shuffle=False,
                                  num_workers=num_workers_test)

    # Initialize GaussianFusionNet with parameters from config
    model = GaussianFusionNet(
        visual_dim=64,
        gaussian_dim=14,
        hidden_dim=256, 
        netvlad_clusters=64,
        netvlad_dim=128,
        output_dim=256
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 打印一下显卡信息，确认现在 PyTorch 只看得到 3 张卡
    if torch.cuda.device_count() > 1:
        print(f"🚀 准备使用 {torch.cuda.device_count()} 张显卡 (GPU 0, 1, 2) 进行加速训练！")
    
    trainer = Trainer(model, train_loader, whole_train_loader, whole_val_set, whole_val_loader, device,
                      num_epochs, resume_path, log, log_dir, ckpt_dir, cache_dir,
                      resume_scheduler, lr, step_size, gamma, margin)
    trainer.train()


if __name__ == '__main__':
    main()