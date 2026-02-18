import os
import h5py
import torch
import pickle
import numpy as np
from PIL import Image
from PIL import ImageFile
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, default_collate
from pyquaternion import Quaternion

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset(Dataset):
    def __init__(self, data_root_dir, info_path, gaussian_path=None, resize=None):
        super().__init__()
        self.infos = self.read_infos(info_path)
        self.dataroot = data_root_dir
        self.img_transforms = None
        self.gaussian_path = gaussian_path
        self.resize = resize
        self.num_points = 4096

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def read_infos(self, info_path):
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        return infos

    def load_data(self, index):
        channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        camera_images = self.load_camera_data(index, channels)
        gaussians = self.load_lidar_data(index)
        return camera_images, gaussians

    def get_pose_matrix(self, translation, rotation):
        if isinstance(rotation, (list, np.ndarray)):
            q = Quaternion(rotation)
        else:
            q = rotation
        R = q.rotation_matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = np.array(translation)
        return T

    def load_lidar_data(self, index):
        lidar_data = self.infos[index]['lidar_infos']['LIDAR_TOP']
        
        if self.gaussian_path is None:
            return torch.zeros(self.num_points, 14, dtype=torch.float32)
        
        raw_filename = lidar_data['filename']
        base_filename = os.path.basename(raw_filename)
        fname = base_filename.replace('.bin', '.npy')
        
        gaussian_data_path = os.path.join(self.gaussian_path, fname)
        
        if not os.path.exists(gaussian_data_path):
            return torch.zeros(self.num_points, 14, dtype=torch.float32)
        
        try:
            gaussian_data = np.load(gaussian_data_path, allow_pickle=True)
        except Exception:
            return torch.zeros(self.num_points, 14, dtype=torch.float32)
        
        if gaussian_data.shape[0] == 0:
            return torch.zeros(self.num_points, 14, dtype=torch.float32)
        
        gaussian_tensor = torch.from_numpy(gaussian_data).float()
        
        N = gaussian_tensor.shape[0]
        if N > self.num_points:
            indices = torch.randperm(N)[:self.num_points]
            gaussian_tensor = gaussian_tensor[indices]
        elif N < self.num_points:
            padding = torch.zeros(self.num_points - N, 14, dtype=torch.float32)
            gaussian_tensor = torch.cat([gaussian_tensor, padding], dim=0)
        
        return gaussian_tensor

    def load_camera_data(self, index, channels):
        imgs = []
        for channel in channels:
            cam_data = self.infos[index]['camera_infos'][channel]
            filename = cam_data['filename']
            img_path = os.path.join(self.dataroot, filename)
            if not os.path.exists(img_path):
                raise Exception(f'FileNotFound! {img_path}')
            
            img = Image.open(img_path)

            if self.resize is not None:
                target_h, target_w = self.resize[0], self.resize[1]
                if img.size != (target_w, target_h):
                    img = img.resize((target_w, target_h), Image.BILINEAR)

            img_tensor = self.img_transforms(img)
            imgs.append(img_tensor)

        imgs = torch.stack(imgs)
        return imgs

    def load_data_with_matrices(self, index):
        channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        
        images = self.load_camera_data(index, channels)
        gaussians = self.load_lidar_data(index)
        
        info = self.infos[index]
        lidar_info = info['lidar_infos']['LIDAR_TOP']
        
        lidar_calib = lidar_info['calibrated_sensor']
        lidar_translation = np.array(lidar_calib['translation'])
        lidar_rotation = Quaternion(lidar_calib['rotation'])
        T_lidar_to_ego = self.get_pose_matrix(lidar_translation, lidar_rotation)
        
        extrinsics_list = []
        intrinsics_list = []
        
        for cam_name in channels:
            cam_info = info['camera_infos'][cam_name]
            cam_calib = cam_info['calibrated_sensor']
            
            cam_translation = np.array(cam_calib['translation'])
            cam_rotation = Quaternion(cam_calib['rotation'])
            T_cam_to_ego = self.get_pose_matrix(cam_translation, cam_rotation)
            
            T_ego_to_cam = np.linalg.inv(T_cam_to_ego)
            T_lidar_to_cam = T_ego_to_cam @ T_lidar_to_ego
            extrinsics_list.append(T_lidar_to_cam)
            
            K = np.array(cam_calib['camera_intrinsic'])
            
            if self.resize is not None:
                try:
                    img_path = os.path.join(self.dataroot, cam_info['filename'])
                    if os.path.exists(img_path):
                        with Image.open(img_path) as img:
                            orig_w, orig_h = img.size
                    else:
                        orig_w, orig_h = 1600, 900
                except:
                    orig_w, orig_h = 1600, 900
                
                target_w, target_h = self.resize[1], self.resize[0]
                
                scale_x = target_w / orig_w
                scale_y = target_h / orig_h
                
                K_scaled = K.copy()
                K_scaled[0, 0] *= scale_x
                K_scaled[0, 2] *= scale_x
                K_scaled[1, 1] *= scale_y
                K_scaled[1, 2] *= scale_y
                K = K_scaled
            
            intrinsics_list.append(K)
        
        extrinsics = torch.from_numpy(np.stack(extrinsics_list)).float()
        intrinsics = torch.from_numpy(np.stack(intrinsics_list)).float()
        
        return {
            'images': images,
            'gaussians': gaussians,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics
        }


class RobustnessAugmentation:
    def __init__(self, stage=1, clean_prob=0.5):
        self.stage = stage
        self.clean_prob = clean_prob
        
    def apply_lidar_aug(self, gaussians):
        if self.stage == 1 or torch.rand(1).item() < self.clean_prob:
            return gaussians
        
        gaussians = gaussians.clone()
        
        if torch.rand(1).item() < 0.5:
            noise_scale = 0.05
            if gaussians.dim() == 2:
                gaussians[:, :3] += torch.randn_like(gaussians[:, :3]) * noise_scale
            elif gaussians.dim() == 3:
                gaussians[:, :, :3] += torch.randn_like(gaussians[:, :, :3]) * noise_scale
            
            dropout_rate = 0.1
            mask = torch.rand(gaussians.shape[:-1] + (1,), device=gaussians.device) > dropout_rate
            gaussians = gaussians * mask.float()
        
        return gaussians
    
    def apply_camera_aug(self, images):
        if self.stage == 1 or torch.rand(1).item() < self.clean_prob:
            return images
        
        images = images.clone()
        
        dropout_prob = torch.rand(1).item()
        valid_views = torch.ones(6, dtype=torch.bool)
        
        if dropout_prob < 0.4:
            images[1:] = 0.0
            valid_views[1:] = False
        elif dropout_prob < 0.7:
            num_drop = torch.randint(1, 4, (1,)).item()
            drop_indices = torch.randperm(6)[:num_drop]
            images[drop_indices] = 0.0
            valid_views[drop_indices] = False
            
        deg_prob = torch.rand(1).item()
        if deg_prob < 0.3:
            dark_factor = 0.1 + 0.2 * torch.rand(1).item()
            images[valid_views] = images[valid_views] * dark_factor
            noise_std = 0.05
            noise = torch.randn_like(images[valid_views]) * noise_std
            images[valid_views] = torch.clamp(images[valid_views] + noise, 0.0, 1.0)
        elif deg_prob < 0.5:
            glare_factor = 1.5 + 1.0 * torch.rand(1).item()
            images[valid_views] = torch.clamp(images[valid_views] * glare_factor, 0.0, 1.0)
            
        return images


class TripletDataset(BaseDataset):
    def __init__(self, data_root_dir, database_path, query_path, info_path, cache_dir, img_transforms, nNeg, nNegSample,
                 nonTrivPosDistThres, posDistThr, margin, gaussian_path=None, resize=None):
        super().__init__(data_root_dir, info_path, gaussian_path=gaussian_path, resize=resize)
        self.data_base = np.load(database_path)
        self.nNeg = nNeg
        self.nNegSample = nNegSample
        self.nonTrivPosDistThres = nonTrivPosDistThres
        self.posDistThr = posDistThr
        self.margin = margin
        self.queries = np.load(query_path)
        self.img_transforms = img_transforms
        self.stage = 1
        self.robust_aug = RobustnessAugmentation(stage=self.stage, clean_prob=1.0)

        knn = NearestNeighbors()
        knn.fit(self.data_base[:, 1:])
        self.nontrivial_positives = list(knn.radius_neighbors(self.queries[:, 1:],
                                                              radius=self.nonTrivPosDistThres,
                                                              return_distance=False))
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)

        potential_positives = list(knn.radius_neighbors(self.queries[:, 1:], radius=self.posDistThr,
                                                        return_distance=False))
        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.data_base.shape[0]), pos,
                                                         assume_unique=True))

        self.cache = os.path.join(cache_dir, 'feat_cache.hdf5')
        self.negCache = [np.empty((0,)) for _ in range(len(self.queries))]
    
    def set_stage(self, stage, clean_prob=0.5):
        self.stage = stage
        self.robust_aug.stage = stage
        self.robust_aug.clean_prob = clean_prob

    def __getitem__(self, index):
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get('features')
            qOffset = len(self.data_base)
            qFeat = h5feat[index + qOffset]
            qFeat = torch.tensor(qFeat)
            posFeat = h5feat[self.nontrivial_positives[index]]
            posFeat = torch.tensor(posFeat)
            dist = torch.norm(qFeat - posFeat, dim=1)
            result = dist.topk(1, largest=False)
            posdist, posidx = result.values, result.indices
            posIndex = self.nontrivial_positives[index][posidx].item()
            if self.negCache[index].ndim == 0:
                self.negCache[index] = self.negCache[index].reshape(1)

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]).astype(int))
            negFeat = h5feat[negSample]
            negFeat = torch.tensor(negFeat)
            dist = torch.norm(qFeat - negFeat, dim=1)
            result = dist.topk(self.nNeg * 10, largest=False)
            negdist, negidx = result.values, result.indices

            vilatingNeg = negdist.numpy() < posdist.numpy() + self.margin
            if np.sum(vilatingNeg) < 1:
                if self.stage == 2:
                    negidx = negidx[:self.nNeg]
                else:
                    return None
            else:
                negidx = negidx[vilatingNeg][:self.nNeg]
            nNeg = len(negidx)
            negIndex = negSample[negidx].astype(int)
            self.negCache[index] = negIndex

        query_idx = int(self.queries[index][0])
        pos_idx = int(self.data_base[posIndex][0])

        q_dict = self.load_data_with_matrices(query_idx)
        pos_dict = self.load_data_with_matrices(pos_idx)

        if self.data_base[negIndex].ndim == 1:
            negIndex = np.array([negIndex])
        neg_idx = self.data_base[negIndex][:, 0].astype(int)
        
        images_list, gaussians_list, extrinsics_list, intrinsics_list = [], [], [], []
        for i in range(len(neg_idx)):
            neg_dict = self.load_data_with_matrices(neg_idx[i])
            neg_dict['images'] = self.robust_aug.apply_camera_aug(neg_dict['images'])
            neg_dict['gaussians'] = self.robust_aug.apply_lidar_aug(neg_dict['gaussians'])
            images_list.append(neg_dict['images'])
            gaussians_list.append(neg_dict['gaussians'])
            extrinsics_list.append(neg_dict['extrinsics'])
            intrinsics_list.append(neg_dict['intrinsics'])

        if self.stage == 2:
            pos_dict['images'] = self.robust_aug.apply_camera_aug(pos_dict['images'])
            pos_dict['gaussians'] = self.robust_aug.apply_lidar_aug(pos_dict['gaussians'])
            
            q_dict['images'] = self.robust_aug.apply_camera_aug(q_dict['images'])
            q_dict['gaussians'] = self.robust_aug.apply_lidar_aug(q_dict['gaussians'])
        
        images_list.extend([pos_dict['images'], q_dict['images']])
        gaussians_list.extend([pos_dict['gaussians'], q_dict['gaussians']])
        extrinsics_list.extend([pos_dict['extrinsics'], q_dict['extrinsics']])
        intrinsics_list.extend([pos_dict['intrinsics'], q_dict['intrinsics']])

        res_dict = {
            'images': torch.stack(images_list),
            'gaussians': torch.stack(gaussians_list),
            'extrinsics': torch.stack(extrinsics_list),
            'intrinsics': torch.stack(intrinsics_list)
        }
        return res_dict, nNeg

    def __len__(self):
        return len(self.queries)

    def read_infos(self, infos_path):
        with open(infos_path, 'rb') as f:
            infos = pickle.load(f)
        return infos


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None
    input_dict, nNeg = zip(*batch)
    return default_collate(input_dict), list(nNeg)


class DatabaseQueryDataset(BaseDataset):
    def __init__(self, data_root_dir, database_path, query_path, info_path, transforms, nonTrivPosDistThres, gaussian_path=None, resize=None):
        super().__init__(data_root_dir, info_path, gaussian_path=gaussian_path, resize=resize)
        data_base = np.load(database_path)
        query = np.load(query_path)

        self.dataset = np.concatenate((data_base, query), axis=0)
        self.num_db = len(data_base)
        self.num_query = len(query)
        self.positives = None
        self.distances = None
        self.cache = None
        self.nonTrivPosDistThres = nonTrivPosDistThres
        self.img_transforms = transforms

    def __getitem__(self, item):
        index = int(self.dataset[item][0])
        res_dict = self.load_data_with_matrices(index)
        return res_dict

    def __len__(self):
        return len(self.dataset)

    def getPositives(self):
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            dataset = np.ascontiguousarray(self.dataset[:self.num_db, 1:])
            knn.fit(dataset)
            self.positives = list(knn.radius_neighbors(self.dataset[self.num_db:, 1:], radius=self.nonTrivPosDistThres,
                                                       return_distance=False))
        return self.positives
