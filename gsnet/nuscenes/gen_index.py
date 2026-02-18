import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
import pickle

def check_path(*paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

def process_region(infos, separate_th, dis_th_db=1.0, pos_th=9.0, val_ratio=0.0):
    pos_whole = []
    timestamps = []
    
    for info in infos:
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
        pos_whole.append(pos[:2])
        timestamps.append(info['timestamp'])

    pos_whole = np.array(pos_whole, dtype=np.float32)
    timestamps = np.array(timestamps, dtype=np.float32)

    pos_whole = np.concatenate(
        (np.arange(len(pos_whole), dtype=np.int32).reshape(-1, 1), pos_whole),
        axis=1).astype(np.float32)

    timestamps = (timestamps - timestamps.min()) / (3600 * 24 * 1e6)

    past_idx = np.where(timestamps < separate_th)[0]
    future_idx = np.where(timestamps >= separate_th)[0]

    pos_past = pos_whole[past_idx]
    pos_future = pos_whole[future_idx]

    pos_db = pos_past[0, :].reshape(1, -1)
    db_indices = {int(pos_past[0, 0])}
    
    for i in range(1, pos_past.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_db[:, 1:3])
        dis, _ = knn.kneighbors(pos_past[i, 1:3].reshape(1, -1))
        
        if dis[0][0] > dis_th_db:
            pos_db = np.concatenate((pos_db, pos_past[i, :].reshape(1, -1)), axis=0)
            db_indices.add(int(pos_past[i, 0]))

    train_query_idx = [i for i in past_idx if i not in db_indices]
    pos_train_query_pool = pos_whole[train_query_idx] if train_query_idx else np.empty((0, 3))
    pos_test_query_pool = pos_future

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_db[:, 1:3])

    def filter_queries(queries_pool):
        if len(queries_pool) == 0:
            return queries_pool
        dis, _ = knn.kneighbors(queries_pool[:, 1:3])
        valid_mask = dis.flatten() < pos_th
        return queries_pool[valid_mask]

    pos_train_query = filter_queries(pos_train_query_pool)
    pos_test_query = filter_queries(pos_test_query_pool)

    pos_val_query = np.empty((0, 3))
    if val_ratio > 0 and len(pos_test_query) > 0:
        num_val = int(len(pos_test_query) * val_ratio)
        val_indices = random.sample(range(len(pos_test_query)), num_val)
        test_indices = list(set(range(len(pos_test_query))) - set(val_indices))
        
        pos_val_query = pos_test_query[val_indices]
        pos_test_query = pos_test_query[test_indices]

    return pos_whole, pos_db, pos_train_query, pos_val_query, pos_test_query

def main():
    random.seed(1)
    np.random.seed(1)
    
    nuscenes_root = '/mnt/nuscenes/'
    dataroot = '/home/james/LSY/GSNET/nuscenes/'
    
    infos_bs_path = os.path.join(dataroot, 'nuscenes_infos-bs.pkl')
    infos_son_path = os.path.join(dataroot, 'nuscenes_infos-son.pkl')
    infos_shv_path = os.path.join(dataroot, 'nuscenes_infos-shv.pkl')
    infos_sq_path = os.path.join(dataroot, 'nuscenes_infos-sq.pkl')
    
    check_path(infos_bs_path, infos_son_path, infos_sq_path)

    with open(infos_bs_path, 'rb') as f:
        infos_bs = pickle.load(f)
    with open(infos_son_path, 'rb') as f:
        infos_son = pickle.load(f)
    with open(infos_sq_path, 'rb') as f:
        infos_sq = pickle.load(f)

    print('==> processing boston seaport (bs)...')
    pos_whole_bs, pos_bs_db, pos_bs_train_query, pos_bs_val_query, pos_bs_test_query = process_region(
        infos_bs, separate_th=105, val_ratio=0.25)

    print('==> processing singapore one north (son)...')
    pos_whole_son, pos_son_db, pos_son_train_query, _, pos_son_test_query = process_region(
        infos_son, separate_th=30, val_ratio=0.0)

    print('==> processing singapore queenstown (sq)...')
    pos_whole_sq, pos_sq_db, pos_sq_train_query, _, pos_sq_test_query = process_region(
        infos_sq, separate_th=15, val_ratio=0.0)

    print('-' * 40)
    print(f'Boston Seaport -> Total: {len(pos_whole_bs)}, DB: {len(pos_bs_db)}, TrainQ: {len(pos_bs_train_query)}, ValQ: {len(pos_bs_val_query)}, TestQ: {len(pos_bs_test_query)}')
    print(f'Singapore ON   -> Total: {len(pos_whole_son)}, DB: {len(pos_son_db)}, TrainQ: {len(pos_son_train_query)}, TestQ: {len(pos_son_test_query)}')
    print(f'Singapore QT   -> Total: {len(pos_whole_sq)}, DB: {len(pos_sq_db)}, TrainQ: {len(pos_sq_train_query)}, TestQ: {len(pos_sq_test_query)}')

    print('===> saving database and queries..')
    np.save(os.path.join(dataroot, 'bs_db.npy'), pos_bs_db)
    np.save(os.path.join(dataroot, 'bs_train_query.npy'), pos_bs_train_query)
    np.save(os.path.join(dataroot, 'bs_val_query.npy'), pos_bs_val_query)
    np.save(os.path.join(dataroot, 'bs_test_query.npy'), pos_bs_test_query)
    
    np.save(os.path.join(dataroot, 'son_db.npy'), pos_son_db)
    np.save(os.path.join(dataroot, 'son_query.npy'), pos_son_test_query)
    
    np.save(os.path.join(dataroot, 'sq_db.npy'), pos_sq_db)
    np.save(os.path.join(dataroot, 'sq_train_query.npy'), pos_sq_train_query)
    np.save(os.path.join(dataroot, 'sq_test_query.npy'), pos_sq_test_query)

if __name__ == '__main__':
    main()