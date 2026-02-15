import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm
import numpy as np
import faiss
import gc

class GaussianUDGLoss(nn.Module):
    def __init__(self, margin=0.5, lambda_gcl=0.0, lambda_pml=0.0, lambda_ssl=0.0, device='cuda'):
        super(GaussianUDGLoss, self).__init__()
        self.margin = margin
        self.lambda_gcl = lambda_gcl
        self.lambda_pml = lambda_pml
        self.lambda_ssl = lambda_ssl
        self.device = device
        
        # 128 匹配模型输出
        self.geo_proj = nn.Linear(256, 6).to(device)
        self.vis_to_geo = nn.Linear(384, 6).to(device)

    def forward(self, global_des, batch_dict, output_dict, nNeg, stage=1):
        gaussians = batch_dict['gaussians']
        fused_feat = output_dict.get('fused_feat')
        if fused_feat is None:
            fused_feat = output_dict.get('embedding')

        sampled_feats = output_dict.get('sampled_visual_feats')
        reliability_pred = output_dict.get('reliability')
        offsets = output_dict.get('offsets')
        
        # 🔥 Stage 2: Competitive Loss (MoE)
        if stage == 2:
            router_weights = output_dict.get('router_weights')  # [B, 3]
            expert_descriptors = output_dict.get('expert_descriptors')  # (geo_desc, vis_desc, fused_desc)
            
            if router_weights is not None and expert_descriptors is not None:
                geo_desc, vis_desc, fused_desc = expert_descriptors
                
                # 计算三个专家的独立损失
                L_g = self._compute_gaussian_wtl(geo_desc, reliability_pred, nNeg)
                L_v = self._compute_gaussian_wtl(vis_desc, reliability_pred, nNeg)
                L_f = self._compute_gaussian_wtl(fused_desc, reliability_pred, nNeg)
                
                # 提取权重 w_g, w_v, w_f (平均batch维度)
                if router_weights.dim() == 3:
                    router_weights = router_weights.mean(dim=1)
                
                w_g = router_weights[:, 0].mean()
                w_v = router_weights[:, 1].mean()
                w_f = router_weights[:, 2].mean()
                
                # 🔥 FIX: 选择性detach
                # L_g和L_v被detach（Geo和Vis专家被冻结，不需要梯度）
                # L_f保持连接（Fusion专家被解冻，必须接收梯度以更新权重）
                total_loss = w_g * L_g.detach() + w_v * L_v.detach() + w_f * L_f
                return total_loss
        
        # Stage 1: 原始损失
        loss_wtl = self._compute_gaussian_wtl(global_des, reliability_pred, nNeg)
        
        loss_gcl = self._compute_gaussian_gcl(fused_feat, gaussians) if fused_feat is not None else torch.tensor(0.0).to(self.device)
        loss_pml = self._compute_gaussian_pml(sampled_feats, gaussians) if sampled_feats is not None else torch.tensor(0.0).to(self.device)
        loss_ssl = self._compute_gaussian_ssl(gaussians, reliability_pred) if reliability_pred is not None else torch.tensor(0.0).to(self.device)
        loss_off = self._compute_offset_reg(offsets) if offsets is not None else torch.tensor(0.0).to(self.device)

        total_loss = loss_wtl + \
                     self.lambda_gcl * loss_gcl + \
                     self.lambda_pml * loss_pml + \
                     self.lambda_ssl * loss_ssl + \
                     0.0 * loss_off
        
        return total_loss

    def _compute_gaussian_wtl(self, global_des, reliability_pred, nNeg):
        num_neg = nNeg
        neg_des, pos_des, query_des = torch.split(global_des, [num_neg, 1, 1], dim=0)
        
        query_des = query_des.expand(num_neg, -1)
        pos_des = pos_des.expand(num_neg, -1)
        
        d_pos = F.pairwise_distance(query_des, pos_des, p=2)
        d_neg = F.pairwise_distance(query_des, neg_des, p=2)
        
        base_loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        
        aggressive_term = 0.5 * d_pos
        
        if reliability_pred is not None:
            sample_rel = reliability_pred.mean(dim=1).squeeze(-1)
            
            w_anc = sample_rel[num_neg + 1]
            w_pos = sample_rel[num_neg]
            w_joint = w_anc * w_pos
            w_final = w_joint.expand(num_neg)
            
            kendall_loss = base_loss * w_final - torch.log(w_final + 1e-6)
            
            final_loss = kendall_loss + aggressive_term
            
            valid_mask = (base_loss > 1e-16).float()
            num_valid = valid_mask.sum()
            
            if num_valid > 0:
                loss_valid = (final_loss * valid_mask).sum() / num_valid
                loss_easy = (final_loss * (1 - valid_mask)).sum() / (1 - valid_mask).sum().clamp(min=1.0)
                return loss_valid + 0.1 * loss_easy
            else:
                return final_loss.mean()
        else:
            return (base_loss + aggressive_term).mean()

    def _compute_gaussian_gcl(self, fused_feat, gaussians):
        # 几何一致性损失 (Geometric Consistency Loss)
        geo_raw = torch.cat([gaussians[..., :3], gaussians[..., 4:7]], dim=-1)
        gaussian_geo = F.normalize(geo_raw, p=2, dim=-1) # 形状: [B, 4096, 6]
        
        feat_proj = self.geo_proj(fused_feat) # 形状: [B, 6]
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)
        
        # 🔥🔥🔥 修复点在这里 🔥🔥🔥
        # 我们把 [B, 6] 变成 [B, 1, 6]
        # 这样 PyTorch 就能把它自动广播到 [B, 4096, 6] 进行计算了
        feat_proj = feat_proj.unsqueeze(1) 

        # 现在: [B, 1, 6] vs [B, 4096, 6] -> 成功!
        geo_similarity = F.cosine_similarity(feat_proj, gaussian_geo, dim=-1)
        return 1 - geo_similarity.mean()

    def _compute_gaussian_pml(self, sampled_feats, gaussians):
        geo_raw = torch.cat([gaussians[..., :3], gaussians[..., 4:7]], dim=-1)
        gaussian_geo_feat = F.normalize(geo_raw, p=2, dim=-1)
        
        visual_proj = self.vis_to_geo(sampled_feats)
        visual_proj = F.normalize(visual_proj, p=2, dim=-1)
        
        proj_match = F.cosine_similarity(visual_proj, gaussian_geo_feat, dim=-1)
        
        scale_magnitude = torch.norm(gaussians[..., 4:7], p=2, dim=-1)
        opacity = gaussians[..., 11:12].squeeze(-1)
        weight = scale_magnitude * opacity
        
        loss = (1 - proj_match) * weight
        return loss.mean()

    def _compute_gaussian_ssl(self, gaussians, reliability):
        xyz = gaussians[..., :3]
        dist_matrix = torch.cdist(xyz, xyz, p=2)
        adj_matrix = (dist_matrix < 0.5).float()
        
        rel_diff = torch.abs(reliability - reliability.transpose(1, 2))
        weighted_diff = rel_diff * adj_matrix
        
        loss = weighted_diff.sum(dim=-1) / (adj_matrix.sum(dim=-1) + 1e-6)
        return loss.mean()

    def _compute_offset_reg(self, offsets):
        return torch.mean(torch.norm(offsets, p=2, dim=-1))


class Trainer:
    # 🔥 阶段切换常量
    STAGE2_START_EPOCH = 20  # 从第10个epoch开始Stage 2
    
    def __init__(self, model, train_loader, whole_train_loader, whole_val_set, whole_val_loader,
                 device, num_epochs, resume_path, log, log_dir, ckpt_dir, cache_dir,
                 resume_scheduler, lr, step_size, gamma, margin):
        self.train_loader = train_loader
        self.whole_train_loader = whole_train_loader
        self.whole_val_set = whole_val_set
        self.whole_val_loader = whole_val_loader
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.device = device

        self.num_epochs = num_epochs
        self.resume_path = resume_path
        self.log = log
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.cache_dir = cache_dir
        self.output_dim = 256
        self.resume_scheduler = resume_scheduler
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.margin = margin
        self.current_stage = 1  # 当前训练阶段
        
        self.criterion = GaussianUDGLoss(margin=margin, device=device).to(device)

    def train(self):
        print("Start training ...")
        checkpoint = None
        if self.resume_path is not None:
            print("Resuming from ", self.resume_path)
            checkpoint = torch.load(self.resume_path)
            starting_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['net'])
        else:
            print("Training from scratch ...")
            starting_epoch = 0

        self.model = self.model.to(self.device)

        self.optimizer = Adam([
            {'params': self.model.visual_encoder.parameters(), 'lr': self.lr * 0.1},
            {'params': self.model.uncertainty_net.parameters(), 'lr': self.lr},
            {'params': self.model.gaussian_encoder.parameters(), 'lr': self.lr},
            {'params': self.model.projection_layer.parameters(), 'lr': self.lr}
        ], lr=self.lr)

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size,
                                             gamma=self.gamma)

        if self.resume_path and checkpoint is not None:
            if self.resume_scheduler and 'scheduler' in checkpoint:
                print("Resuming scheduler")
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            if 'optimizer' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                except:
                    pass
            
            del checkpoint

        writer = None
        if self.log:
            time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
            writer = SummaryWriter(log_dir=os.path.join(self.log_dir, time_stamp))
        
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

        for epoch in range(starting_epoch + 1, self.num_epochs):
            # 🔥 CRITICAL: 阶段切换逻辑
            if epoch == self.STAGE2_START_EPOCH:
                print("=" * 50)
                print(f"🔥 SWITCHING TO STAGE 2 (Frozen Competitive MoE) at Epoch {epoch}")
                print("=" * 50)
                
                # 1. 切换数据集阶段
                if hasattr(self.train_loader.dataset, 'set_stage'):
                    self.train_loader.dataset.set_stage(2)
                    print("✓ Dataset switched to Stage 2 (Robust)")
                
                # 2. 冻结Visual和Gaussian编码器的基础层（backbones）
                model_to_freeze = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                for param in model_to_freeze.visual_encoder.parameters():
                    param.requires_grad = False
                for param in model_to_freeze.uncertainty_net.parameters():
                    param.requires_grad = False
                for param in model_to_freeze.projection_layer.parameters():
                    param.requires_grad = False
                for param in model_to_freeze.gaussian_encoder.proj_gaussian.parameters():
                    param.requires_grad = False
                for param in model_to_freeze.gaussian_encoder.proj_visual.parameters():
                    param.requires_grad = False
                print("✓ Visual encoder, UncertaintyNet, ProjectionLayer, and projection layers FROZEN")
                
                # 3. 🔥 FIX: 解冻Fusion模块和Expert Head层（允许适应Noisy输入）
                # 解冻Fusion模块 (DualCrossAttention)
                for param in model_to_freeze.gaussian_encoder.dual_attn.parameters():
                    param.requires_grad = True
                
                # 解冻NetVLAD专家层
                for param in model_to_freeze.gaussian_encoder.netvlad_geo.parameters():
                    param.requires_grad = True
                for param in model_to_freeze.gaussian_encoder.netvlad_vis.parameters():
                    param.requires_grad = True
                for param in model_to_freeze.gaussian_encoder.netvlad_fused.parameters():
                    param.requires_grad = True
                
                # 解冻Bottleneck专家层
                for param in model_to_freeze.gaussian_encoder.bottleneck_geo.parameters():
                    param.requires_grad = True
                for param in model_to_freeze.gaussian_encoder.bottleneck_vis.parameters():
                    param.requires_grad = True
                for param in model_to_freeze.gaussian_encoder.bottleneck_fused.parameters():
                    param.requires_grad = True
                
                # 解冻Router
                for param in model_to_freeze.gaussian_encoder.router.parameters():
                    param.requires_grad = True
                print("✓ Fusion module (DualCrossAttention), Expert Heads (NetVLAD+Bottleneck), and Router UNFROZEN")
                
                # 4. 重新初始化优化器（跟踪Fusion、Expert Heads和Router）
                self.optimizer = Adam([
                    {'params': model_to_freeze.gaussian_encoder.dual_attn.parameters(), 'lr': self.lr * 0.1},
                    {'params': model_to_freeze.gaussian_encoder.netvlad_geo.parameters(), 'lr': self.lr * 0.1},
                    {'params': model_to_freeze.gaussian_encoder.netvlad_vis.parameters(), 'lr': self.lr * 0.1},
                    {'params': model_to_freeze.gaussian_encoder.netvlad_fused.parameters(), 'lr': self.lr * 0.1},
                    {'params': model_to_freeze.gaussian_encoder.bottleneck_geo.parameters(), 'lr': self.lr * 0.1},
                    {'params': model_to_freeze.gaussian_encoder.bottleneck_vis.parameters(), 'lr': self.lr * 0.1},
                    {'params': model_to_freeze.gaussian_encoder.bottleneck_fused.parameters(), 'lr': self.lr * 0.1},
                    {'params': model_to_freeze.gaussian_encoder.router.parameters(), 'lr': self.lr}
                ], lr=self.lr)
                self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
                print("✓ Optimizer re-initialized (Fusion + Expert Heads + Router)")
                
                self.current_stage = 2
            
            print("============================================\n")
            print('epoch: ', epoch)
            print('stage: ', self.current_stage)
            print('number of queries: ', len(self.train_loader))
            print('learning rate: ', self.optimizer.state_dict()['param_groups'][0]['lr'])
            if self.log:
                writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
            print("\n============================================\n")

            print('building cache..')
            self.build_cache()

            self.model.train()
            loss_each_epoch = 0
            used_num = 0
            
            for i, (input_dict, nNeg) in enumerate(self.train_loader):
                if input_dict is None:
                    continue
                used_num += 1
                nNeg = nNeg[0]

                batch_dict = {}
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        batch_dict[key] = value.squeeze(0).to(self.device)
                    else:
                        batch_dict[key] = value

                self.optimizer.zero_grad()

                output_dict = self.model(batch_dict)
                global_des = output_dict['embedding']
                global_des = nn.functional.normalize(global_des, p=2, dim=1)

                # 🔥 传入stage参数用于Competitive Loss
                loss = self.criterion(global_des, batch_dict, output_dict, nNeg, stage=self.current_stage)

                if torch.isnan(loss):
                    print('something wrong!!!')
                    continue

                loss.backward()
                self.optimizer.step()
                
                if i % 10 == 0:
                    print(f"Epoch {epoch} Iter {used_num}: Loss={loss.item():.4f}")
                    if self.current_stage == 2:
                        router_weights = output_dict.get('router_weights')
                        if router_weights is not None:
                            w_g, w_v, w_f = router_weights[0, 0].item(), router_weights[0, 1].item(), router_weights[0, 2].item()
                            print(f"  Router weights: w_g={w_g:.3f}, w_v={w_v:.3f}, w_f={w_f:.3f}")

                loss_each_epoch = loss_each_epoch + loss.item()

            self.scheduler.step()
            avg_loss = loss_each_epoch / used_num
            print("epoch {} avg loss {}".format(epoch, avg_loss))
            print("saving weights ...")
            ckpt_path = os.path.join(self.ckpt_dir, 'LCPR_epoch_' + str(epoch) + '.pth.tar')

            if isinstance(self.model, nn.DataParallel):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()

            checkpoint = {'epoch': epoch,
                          'net': model_state,
                          'scheduler': self.scheduler.state_dict(),
                          'optimizer': self.optimizer.state_dict(),
                          'stage': self.current_stage
                          }
            torch.save(checkpoint, ckpt_path)
            print("Model Saved As " + 'LCPR_epoch_' + str(epoch) + '.pth.tar')
            if self.log:
                writer.add_scalar("train/loss", avg_loss, global_step=epoch)

            recalls = self.val()

            if self.log:
                for n, recall in recalls.items():
                    writer.add_scalar('val/recall@{}'.format(n), recall, epoch)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def build_cache(self):
        self.model.eval()
        cache_path = os.path.join(self.cache_dir, "feat_cache.hdf5")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        with h5py.File(cache_path, mode='w') as h5:
            # 🔥🔥🔥 核心修复 1: 使用 .dataset 的长度，而不是 loader 的长度 🔥🔥🔥
            dataset_len = len(self.whole_train_loader.dataset)
            h5feat = h5.create_dataset('features', [dataset_len, self.output_dim], dtype=np.float32)
            
            ptr = 0 # 写入指针
            with torch.no_grad():
                for i, input_dict in enumerate(tqdm(self.whole_train_loader, desc="Building Cache")):
                    batch_dict = {}
                    for key, value in input_dict.items():
                        if isinstance(value, torch.Tensor):
                            batch_dict[key] = value.to(self.device)
                        else:
                            batch_dict[key] = value
                    
                    output_dict = self.model(batch_dict)
                    descriptor = output_dict['embedding']
                    descriptor = nn.functional.normalize(descriptor, p=2, dim=1)
                    
                    # 🔥🔥🔥 核心修复 2: 批量写入 h5 🔥🔥🔥
                    current_batch_size = descriptor.shape[0]
                    h5feat[ptr : ptr + current_batch_size, :] = descriptor.cpu().numpy()
                    ptr += current_batch_size
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def val(self):
        self.model.eval()
        with torch.no_grad():
            print('getting recall..')
            cache = os.path.join(self.cache_dir, 'feat_cache_val.hdf5')
            with h5py.File(cache, mode='w') as h5:
                # 🔥🔥🔥 核心修复 1: 使用 .dataset 的长度 🔥🔥🔥
                dataset_len = len(self.whole_val_loader.dataset)
                dbFeat = h5.create_dataset('features', [dataset_len, self.output_dim], dtype=np.float32)
                
                ptr = 0 # 写入指针
                for i, input_dict in enumerate(tqdm(self.whole_val_loader, desc="Validating")):
                    batch_dict = {}
                    for key, value in input_dict.items():
                        if isinstance(value, torch.Tensor):
                            batch_dict[key] = value.to(self.device)
                        else:
                            batch_dict[key] = value
                    
                    output_dict = self.model(batch_dict)
                    descriptor = output_dict['embedding']
                    descriptor = nn.functional.normalize(descriptor, p=2, dim=1)
                    
                    # 🔥🔥🔥 核心修复 2: 批量写入 🔥🔥🔥
                    current_batch_size = descriptor.shape[0]
                    dbFeat[ptr : ptr + current_batch_size, :] = descriptor.cpu().numpy()
                    ptr += current_batch_size

        with h5py.File(cache, mode='r') as h5:
            dbFeat = h5.get('features')

            n_values = [1, 5, 10, 20]
            qFeat = dbFeat[self.whole_val_set.num_db:].astype('float32')
            dbFeat = dbFeat[:self.whole_val_set.num_db].astype('float32')
            faiss_index = faiss.IndexFlatL2(self.output_dim)
            faiss_index.add(dbFeat)
            dists, predictions = faiss_index.search(qFeat, len(dbFeat)) 

            gt = self.whole_val_set.getPositives()
            correct_at_n = np.zeros(len(n_values))
            for qIx, pred in enumerate(predictions):
                for i, n in enumerate(n_values):
                    if np.any(np.in1d(pred[:n], gt[qIx])):
                        correct_at_n[i:] += 1
                        break
            recall_at_n = correct_at_n / self.whole_val_set.num_query * 100.0

            recalls = {}
            for i, n in enumerate(n_values):
                recalls[n] = recall_at_n[i]

            print('[validate]')
            print('recall@1: {:.2f}\t'.format(recalls[1]), end='')
            print('recall@5: {:.2f}\t'.format(recalls[5]), end='')
            print('recall@10: {:.2f}\t'.format(recalls[10]), end='')
            print('recall@20: {:.2f}\t'.format(recalls[20]))

            return recalls


class Evaluator:
    def __init__(self, model, whole_test_set, whole_test_loader, result_dir, device):
        self.whole_test_set = whole_test_set
        self.whole_test_loader = whole_test_loader
        self.model = model
        self.output_dim = 256
        self.result_dir = result_dir
        self.device = device

    @torch.no_grad()
    def get_feature(self, ckpt_path, feature_path):
        print('=================evaluating=================')
        assert ckpt_path is not None
        print('load weights from: ', ckpt_path)
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['net'])
        self.model = self.model.to(self.device)
        print('predicting.....')
        print('database: ', self.whole_test_set.num_db, 'test query: ', self.whole_test_set.num_query)
        self.model.eval()

        with torch.no_grad():
            gt = self.whole_test_set.getPositives()
            Feat = np.empty((len(self.whole_test_set), self.output_dim))
            
            ptr = 0
            for i, input_dict in enumerate(tqdm(self.whole_test_loader)):
                batch_dict = {}
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        batch_dict[key] = value.to(self.device)
                    else:
                        batch_dict[key] = value
                
                output_dict = self.model(batch_dict)
                descriptor = output_dict['embedding']
                descriptor = nn.functional.normalize(descriptor, p=2, dim=1)
                
                # 批量写入
                current_batch_size = descriptor.shape[0]
                Feat[ptr : ptr + current_batch_size, :] = descriptor.detach().cpu().numpy()
                ptr += current_batch_size
                
                del input_dict, descriptor, batch_dict

            qFeat = Feat[self.whole_test_set.num_db:].astype('float32')
            dbFeat = Feat[:self.whole_test_set.num_db].astype('float32')

            with open(feature_path, 'wb') as f:
                feature = {'qFeat': qFeat, 'dbFeat': dbFeat, 'gt': gt}
                pickle.dump(feature, f, protocol=pickle.HIGHEST_PROTOCOL)
                print('saved at: ', feature_path)

    def get_recall_at_n(self, feature_path):
        assert feature_path is not None
        with open(feature_path, 'rb') as f:
            feature = pickle.load(f)
            qFeat = feature['qFeat']
            dbFeat = feature['dbFeat']
            gt = feature['gt']

        n_values = [1, 5, 10]
        faiss_index = faiss.IndexFlatL2(self.output_dim)
        faiss_index.add(dbFeat)
        dists, preds = faiss_index.search(qFeat, len(dbFeat))

        correct_at_n = np.zeros(len(n_values))
        print('getting recall...')
        for qIx, pred in enumerate(preds):
            for i, n in enumerate(n_values):
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    break
        recall_at_n = correct_at_n / self.whole_test_set.num_query * 100.0

        recalls = {}
        for i, n in enumerate(n_values):
            recalls[n] = recall_at_n[i]

        print('[test]')
        for i, n in enumerate(n_values):
            print('recall@', n, ': {:.2f}\t'.format(recalls[n]), end='')
        print()
        return recalls

    def get_pr(self, feature_path, vis=True):
        with open(feature_path, 'rb') as f:
            feature = pickle.load(f)
            qFeat = feature['qFeat']
            dbFeat = feature['dbFeat']
            gt = feature['gt']

        faiss_index = faiss.IndexFlatL2(self.output_dim)
        faiss_index.add(dbFeat)
        dists, preds = faiss_index.search(qFeat, len(dbFeat))
        dists_max = dists[:, 0].max()
        dists_min = dists[:, 0].min()
        if dists_min - 0.1 > 0:
            dists_min -= 0.1
        dists_u = np.linspace(dists_min, dists_max + 0.1, 1000)

        recalls = []
        precisions = []
        print('getting pr...')
        for th in tqdm(dists_u, ncols=40):
            TPCount = 0
            FPCount = 0
            FNCount = 0
            TNCount = 0
            for index_q in range(dists.shape[0]):
                if dists[index_q, 0] < th:
                    if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                        TPCount += 1
                    else:
                        FPCount += 1
                else:
                    if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                        FNCount += 1
                    else:
                        TNCount += 1
            if TPCount + FNCount == 0 or TPCount + FPCount == 0:
                continue
            recall = TPCount / (TPCount + FNCount)
            precision = TPCount / (TPCount + FPCount)
            recalls.append(recall)
            precisions.append(precision)
        return recalls, precisions

    def get_f1score(self, feature_path):
        recalls, precisions = self.get_pr(feature_path)
        recalls = np.array(recalls)
        precisions = np.array(precisions)
        ind = np.argsort(recalls)
        recalls = recalls[ind]
        precisions = precisions[ind]
        f1s = []
        for index_j in range(len(recalls)):
            f1 = 2 * precisions[index_j] * recalls[index_j] / (precisions[index_j] + recalls[index_j])
            f1s.append(f1)

        print('f1 score: ', max(f1s))
        return f1s