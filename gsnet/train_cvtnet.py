import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import torch
from tensorboardX import SummaryWriter
import yaml
import numpy as np
import modules.loss as PNV_loss
from modules.mvmt_net import MVMT_Net
from tools.read_samples import read_one_ri_bev_from_seq
from tools.read_samples import read_one_batch_ri_bev_from_seq

np.set_printoptions(threshold=sys.maxsize)

class trainHandler():
    def __init__(self, lr=0.001, step_size=5, gamma=0.9, overlap_th=0.3, use_shuffle=False,
                 num_pos=6, num_neg=6, resume=False,
                 pretrained_weights=None, save_path=None, train_set=None, ri_bev_root=None):
        super(trainHandler, self).__init__()

        self.learning_rate = lr
        self.ri_bev_root = ri_bev_root
        self.resume = resume
        self.pretrained_weights = pretrained_weights
        self.save_path = save_path
        self.overlap_thresh = overlap_th
        self.use_shuffle = use_shuffle
        self.max_num_pos = num_pos
        self.max_num_neg = num_neg

        self.train_set = train_set
        self.train_set_imgf1_imgf2_overlap = np.load(self.train_set)

        self.amodel = MVMT_Net.create()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.parameters = self.amodel.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def train(self):
        epochs = 50

        if self.resume:
            print("resuming from ", self.pretrained_weights)
            checkpoint = torch.load(self.pretrained_weights)
            starting_epoch = checkpoint['epoch']
            self.amodel.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("training from scratch ...")
            starting_epoch = 0

        writer1 = SummaryWriter(comment="MVMT_Net_Training")

        for i in range(starting_epoch + 1, epochs):

            if self.use_shuffle:
                np.random.shuffle(self.train_set_imgf1_imgf2_overlap)

            self.train_imgf1 = self.train_set_imgf1_imgf2_overlap[:, 0]
            self.train_imgf2 = self.train_set_imgf1_imgf2_overlap[:, 1]
            self.train_dir1 = np.zeros((len(self.train_imgf1),))
            self.train_dir2 = np.zeros((len(self.train_imgf2),))
            self.train_overlap = self.train_set_imgf1_imgf2_overlap[:, 2].astype(float)

            print("=======================================================================\n")
            print(f"Epoch {i} start. Total pairs: {len(self.train_imgf1)}")
            print("\n=======================================================================")

            loss_each_epoch = 0
            cnt = 0
            used_set = set()

            for j in range(len(self.train_imgf1)):
                f1_index = self.train_imgf1[j]
                dir1_index = self.train_dir1[j]
                
                pair_key = (f1_index, dir1_index)
                if pair_key in used_set:
                    continue
                used_set.add(pair_key)

                try:
                    current_bev, current_riv = read_one_ri_bev_from_seq(f1_index, self.ri_bev_root)
                    
                    sample_bev, sample_riv, sample_truth, pos_num, neg_num = read_one_batch_ri_bev_from_seq(
                        f1_index, dir1_index, self.train_imgf1, self.train_imgf2, 
                        self.train_dir1, self.train_dir2, self.train_overlap,
                        self.overlap_thresh, self.ri_bev_root
                    )
                except ValueError:
                    print(f"Error loading data for index {f1_index}. Skipping.")
                    continue

                if pos_num >= self.max_num_pos and neg_num >= self.max_num_neg:
                    sample_bev = torch.cat((sample_bev[0:self.max_num_pos], sample_bev[pos_num:pos_num+self.max_num_neg]), dim=0)
                    sample_riv = torch.cat((sample_riv[0:self.max_num_pos], sample_riv[pos_num:pos_num+self.max_num_neg]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:self.max_num_pos], sample_truth[pos_num:pos_num+self.max_num_neg]), dim=0)
                    pos_num = self.max_num_pos
                    neg_num = self.max_num_neg
                    
                elif pos_num >= self.max_num_pos:
                    sample_bev = torch.cat((sample_bev[0:self.max_num_pos], sample_bev[pos_num:]), dim=0)
                    sample_riv = torch.cat((sample_riv[0:self.max_num_pos], sample_riv[pos_num:]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:self.max_num_pos], sample_truth[pos_num:]), dim=0)
                    pos_num = self.max_num_pos
                    
                elif neg_num >= self.max_num_neg:
                    sample_bev = sample_bev[0:pos_num+self.max_num_neg]
                    sample_riv = sample_riv[0:pos_num+self.max_num_neg]
                    sample_truth = sample_truth[0:pos_num+self.max_num_neg]
                    neg_num = self.max_num_neg

                if neg_num == 0 or pos_num == 0:
                    continue

                input_bev = torch.cat((current_bev, sample_bev), dim=0).to(self.device).requires_grad_(True)
                input_riv = torch.cat((current_riv, sample_riv), dim=0).to(self.device).requires_grad_(True)

                self.amodel.train()
                self.optimizer.zero_grad()

                global_des = self.amodel(input_bev, input_riv)

                o1, o2, o3 = torch.split(global_des, [1, pos_num, neg_num], dim=0)
                
                MARGIN = 0.5
                loss = PNV_loss.triplet_loss(o1, o2, o3, MARGIN, lazy=False)
                
                loss.backward()
                self.optimizer.step()
                
                if cnt % 10 == 0:
                    print(f"Step {cnt}, Loss: {loss.item()}")

                loss_each_epoch += loss.item()
                cnt += 1

            if cnt > 0:
                avg_loss = loss_each_epoch / cnt
                print(f"Epoch {i} finished. Avg Loss: {avg_loss}")
                writer1.add_scalar("loss", avg_loss, global_step=i)
            
            self.scheduler.step()
            
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            
            save_name = os.path.join(self.save_path, f"mvmt_net_epoch{i}.pth.tar")
            print(f"Saving weights to {save_name}")
            torch.save({
                'epoch': i,
                'state_dict': self.amodel.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, save_name)

if __name__ == '__main__':
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    
    train_cfg = config["train_cvtnet"]
    data_root = config["data_root"]

    train_handler = trainHandler(
                                 lr=train_cfg["lr"],
                                 step_size=train_cfg["step_size"],
                                 gamma=train_cfg["gamma"],
                                 overlap_th=train_cfg["overlap_th"],
                                 use_shuffle=train_cfg["use_shuffle"],
                                 num_pos=train_cfg["num_pos"],
                                 num_neg=train_cfg["num_neg"],
                                 resume=train_cfg["resume"],
                                 pretrained_weights=train_cfg["weights"],
                                 save_path=train_cfg["save_path"],
                                 train_set=train_cfg["traindata_file"],
                                 ri_bev_root=data_root["ri_bev_database_root"]
                                 )
    train_handler.train()