import torch
# 🔥 修改 1: 导入新模型，而不是 LCPR
from modules.GS import GaussianFusionNet 
from dataset.NuScenesDataset import DatabaseQueryDataset
from tools.utils import load_config, check_dir, check_path
from tools.runner import Evaluator
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os

cfg = load_config('config/config.yaml')

# ====================parse config====================
data_root_dir = cfg['data']['data_root_dir']
database_path = cfg['data']['database_path']
train_query_path = cfg['data']['train_query_path']
test_query_path = cfg['data']['test_query_path']
val_query_path = cfg['data']['val_query_path']
info_path = cfg['data']['info_path']
# 🔥 修改 2: 获取 gaussian_path
gaussian_path = cfg['data'].get('gaussian_path', None) 

nonTrivPosDistThres = cfg['runner']['nonTrivPosDistThres']
resize = cfg['runner']['resize']
num_workers_test = cfg['runner']['num_workers_test']

result_dir = cfg['runner']['result_dir']
cache_dir = cfg['runner']['cache_dir']

# ====================check dirs and paths====================
check_path(data_root_dir, database_path, train_query_path, test_query_path, val_query_path, info_path)
check_dir(result_dir, cache_dir)

# ===========================model============================
# 🔥 修改 3: 初始化新模型 (参数必须和 train.py 里一模一样)
model = GaussianFusionNet(
    visual_dim=64,
    gaussian_dim=14,
    hidden_dim=256,
    netvlad_clusters=64,
    netvlad_dim=128,
    output_dim=256
)

# ==========================dataset===========================
img_transforms = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🔥 修改 4: 传入 gaussian_path 和 resize
whole_test_set = DatabaseQueryDataset(
    data_root_dir, 
    database_path, 
    test_query_path, 
    info_path,
    img_transforms, 
    nonTrivPosDistThres,
    gaussian_path=gaussian_path, # 必须加
    resize=resize                # 必须加
)

whole_test_loader = DataLoader(dataset=whole_test_set, batch_size=1, shuffle=False,
                               num_workers=num_workers_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evaluator = Evaluator(model, whole_test_set, whole_test_loader, result_dir, device)

feature_name = 'test.pickle'
# 请确认这个路径下的权重确实是 GaussianFusionNet 训练出来的
weights = '/home/james/LSY/GSNET/weights/LCPR_epoch_31.pth.tar'  
feature_path = os.path.join(evaluator.result_dir, feature_name)

# 开始测试
evaluator.get_feature(weights, feature_path)
evaluator.get_recall_at_n(feature_path)