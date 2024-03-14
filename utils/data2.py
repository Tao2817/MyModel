import pandas as pd
import scipy.sparse as sp
import numpy as np
# from torch_geometric_temporal import dataset as dt
import os, pickle, json, itertools, datetime
from torch_geometric_temporal import StaticGraphTemporalSignal
import torch
from utils.graph_weight import calculate_weights
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt

def load_config(data_name):

    config = {
        'model': 'ARIMA',
        'p_range': [0, 4],
        'd_range': [0, 3],
        'q_range': [0, 4],
        'dataset': data_name,
        'train_rate': 0.7,
        'eval_rate': 0.1,
        'input_window': 12,
        'output_window': 12,
        'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE']
    }
    return config

def date2weekday(date:str) -> tuple[int, int, int]:
    #2012-03-01
    weekday = datetime.datetime.strptime(date[0:10], '%Y-%m-%d').date().isoweekday()
    month = int(date[5:7])
    #07:55:00
    # time step == 5min, so there are 24*60/5 = 288 slices in one day
    hour, minute, secound = tuple(map(int, date[11:-1].split(":")))
    return month, weekday, 12*hour + minute//5

def collate_custom(data_batch):
    pass
    x_batch, y_batch, time_stamp_batch = [], [], []
    for x,y,t,edge_index in data_batch:
        x_batch.append(torch.from_numpy(x))
        y_batch.append(torch.from_numpy(y))
        time_stamp_batch.append(torch.tensor(t))

    # (batch_size,his_len,node_num,feat_dim)
    x_batch = torch.stack(x_batch,dim=0)
    # (batch_size,fut_len,node_num,feat_dim)
    y_batch = torch.stack(y_batch,dim=0)
    time_stamp_batch = torch.stack(time_stamp_batch,dim=0)

    month_batch, weekday_batch, min_batch = time_stamp_batch.split([1,1,1],dim=-1)
    month_batch = one_hot(month_batch-1,num_classes=12)#month:1-12->0-12
    weekday_batch = one_hot(weekday_batch-1,num_classes=7)#weekday:1-7->0-6
    min_batch = one_hot(min_batch,num_classes=288)#min:0-287

    # (batch_size,te_dim==307), 12(month) + 7(weekday) + 288(5min of a day) = 307
    time_stamp_batch = torch.cat([month_batch,weekday_batch,min_batch],dim=-1)
    edge_list = data_batch[0][3]
    return x_batch, y_batch, time_stamp_batch, edge_list

class myDataset(Dataset):
    def __init__(self, args, features, targets, edge_index, time_slots:list, std, mean, augmented=False) -> None:
        super().__init__()
        self.augmented = augmented
        self.his_len, self.fut_len = args.his_len, args.fut_len
        self.std, self.mean = std / 3, mean
        if augmented:
            self.features, self.features_aug = features
            self.targets, self.targets_aug = targets
            self.time_slots, self.time_slots_aug = time_slots
            self.edge_index = edge_index
        else:
            self.features = features
            self.targets = targets
            self.time_slots = time_slots
            self.edge_index = edge_index
    
    def __getitem__(self, index:int) -> torch.Tensor:
        if self.augmented:
            if index < len(self.features):
                # return original data
                return self.features[index], self.targets[index], self.time_slots[index:index + self.his_len + self.fut_len], self.edge_index
            else:
                # return augmented data
                index = index - len(self.features)
                return self.features_aug[index], self.targets_aug[index], self.time_slots_aug[index:index + self.his_len + self.fut_len], self.edge_index
        else:   
            return self.features[index], self.targets[index], self.time_slots[index:index + self.his_len + self.fut_len], self.edge_index
    
    def __len__(self):
        if self.augmented:
            return len(self.features)+len(self.features_aug)
        else:
            return len(self.features)
        
    def norm(self, x):
        x = x - self.mean
        x = x / self.std
        return x
    
    def norm_inv(self, x):
        x = x * self.std
        x = x + self.mean
        return x


                
class process_dataloader(object):
    def __init__(self, args, data_name):
        super(process_dataloader, self).__init__()
        self.args = args
        self.data_dir = os.path.join(f"{self.args.data_path}/raw/", data_name)
        self.data_name = data_name
        self.config = load_config(data_name)
        config_path = os.path.join(self.data_dir, 'config.json')
        self._read_data()
        # read config
        with open(config_path, 'r') as f:
            json_obj = json.load(f)
            for key in json_obj:
                if key not in self.config:
                    self.config[key] = json_obj[key]
    def _read_data(self):
        # read geo
        self.geo_file = pd.read_csv(os.path.join(self.data_dir, self.data_name + ".geo"))  # 存储交通状态信息
        # 传感器的编号
        self.geo_ids = list(self.geo_file['geo_id'])

        # read dyna
        dyna_file = pd.read_csv(os.path.join(self.data_dir, self.data_name + ".dyna"))  # 存储地理实体属性信息
        data_col = self.config.get('data_col', '')  # traffic_speed
        if data_col != '':  # 根据指定的列加载数据集
            if isinstance(data_col, list):
                data_col = data_col.copy()
            else:  # str
                data_col = [data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'entity_id')
            dyna_file = dyna_file[data_col]
        else:  # 不指定则加载所有列
            # Index(['time', 'entity_id', 'traffic_speed'], dtype='object')
            dyna_file = dyna_file[dyna_file.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.time_slots = list(dyna_file['time'][:int(dyna_file.shape[0] / len(self.geo_ids))])
        #2012-03-01T08:15:00Z
        self.time_slots = [date2weekday(time) for time in self.time_slots]

        self.time_seq=np.arange(len(self.time_slots))
        # 转3-d数组
        feature_dim = len(dyna_file.columns) - 2
        # df_sp=dyna_file["traffic_speed"]
        speed_loc=dyna_file.columns.get_loc('traffic_speed')
        df = dyna_file[dyna_file.columns[speed_loc:]]
        # df=df.iloc[:,-1]
        len_time = len(self.time_slots)

        self.X = []
        for i in range(0, df.shape[0], len_time):
            self.X.append(df[i:i + len_time].values)
        self.X = np.array(self.X, dtype=float)  # (N, T, F)
        self.X = self.X.swapaxes(0, 1)  # (Time, Node, Feature)



    def _get_edges_and_weights(self):
        # read rel and constract adj_mat
        save_index_path = self.data_dir + "/{}_edge_index.npy".format(self.data_name)
        save_weights_path = self.data_dir + "/{}_edge_weights.npy".format(self.data_name)
        # if not os.path.isfile(self.data_dir + "/{}_edge_index.npy".format(self.data_name)):
        _geo_to_ind = {}
        weight_col = self.config.get('weight_col', '')
        rel_file = pd.read_csv(os.path.join(self.data_dir, self.data_name + ".rel"))  # 存储实体间的关系信息，如路网
        for index, idx in enumerate(self.geo_ids):
            _geo_to_ind[idx] = index
        if 'weight' in rel_file.columns:
            lst = [(_geo_to_ind[e[0]], _geo_to_ind[e[1]], e[2]) for e in
                   rel_file[['origin_id', 'destination_id', 'weight']].values]
            origin_id = np.array([x[0] for x in lst])
            destination_id = np.array([x[1] for x in lst])
            self.weights = np.array([x[2] for x in lst])
            self.edges_index = np.array([origin_id, destination_id])
            self.num_edges = len(self.edges_index)
        else:
            lst = [(_geo_to_ind[e[0]], _geo_to_ind[e[1]], 1) for e in
                   rel_file[['origin_id', 'destination_id']].values]
            origin_id = np.array([x[0] for x in lst])
            destination_id = np.array([x[1] for x in lst])
            self.weights = np.array([x[2] for x in lst])
            self.edges_index = np.array([origin_id, destination_id])
            self.num_edges = len(self.edges_index)
        if self.data_name in ["PEMS_BAY","METR_LA"]:
            self.weights = calculate_weights(rel_file, self.geo_file)
        #     np.save(save_index_path, self.edges_index)
        #     np.save(save_weights_path, self.weights)
        #     # self.edges_index=self.edges_index.unsqueeze(-1)
        #     # self.weights =self.weights .unsqueeze(0)
        # else:
        #     self.weights = np.load(save_weights_path)
        #     self.edges_index = np.load(save_index_path)

    def seq2sample(self, X, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        features = []
        targets = []
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(X.shape[0] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        for i, j in indices:
            features.append((X[i : i + num_timesteps_in,:, :]))
            targets.append((X[i + num_timesteps_in : j,:, :]))

        return features, targets

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        len_X = self.X.shape[0]     
        # 数据集缩小
        self.X = self.X[0:int(len_X * self.args.shrink_rate)]
        len_X = self.X.shape[0]


        # self.X = self.X - self.mean
        # self.X = self.X / self.std

        
        point1 = int(np.floor(0.7*len_X))
        point2 = int(np.floor(0.9*len_X))

        self.std = np.std(self.X[0:point1])
        self.mean = np.mean(self.X[0:point1])

        # generate seqs len == 24
        self.features_train, self.targets_train = self.seq2sample(self.X[0:point1], self.args.his_len, self.args.fut_len)
        self.time_slots_train = self.time_slots[:point1+24]
        # data augmentation
        self.features_train_aug, self.targets_train_aug = self.seq2sample(self.X[0:point1:self.args.aggregation_size], self.args.his_len, self.args.fut_len)
        self.time_slots_aug = self.time_slots[0:point1:self.args.aggregation_size]
        self.time_slots_train_aug = self.time_slots[:point1:self.args.aggregation_size]


        self.features_vali, self.targets_vali = self.seq2sample(self.X[point1:point2], self.args.his_len, self.args.fut_len)
        self.time_slots_vali = self.time_slots[point1:point2+24]
        self.features_test, self.targets_test = self.seq2sample(self.X[point2:], self.args.his_len, self.args.fut_len)
        self.time_slots_test = self.time_slots[point2:]

        return

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ):

        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        # time_slots = {"time_slots": value for value in self.time_slots}

        # point1 = int(np.floor(0.7*len(self.features)))
        # point2 = int(np.floor(0.9*len(self.features)))
        if self.args.aggregation_size is not None:
            #train_loader = 0.7 * origin data + augmented data
            train_loader = DataLoader(myDataset(self.args, 
                                                (self.features_train, self.features_train_aug),
                                                (self.targets_train, self.targets_train_aug),
                                                self.edges_index,
                                                (self.time_slots_train, self.time_slots_train_aug),
                                                self.std,
                                                self.mean,
                                                augmented=True
                                                ),
                                                shuffle=True,
                                                batch_size=self.args.batch_size,
                                                collate_fn=collate_custom)# do no thing to dataset
        else:
            train_loader = DataLoader(myDataset(self.args,
                                                self.features_train, self.targets_train,
                                                self.edges_index,
                                                self.time_slots_train,
                                                self.std,
                                                self.mean
                                                ),
                                                shuffle=True,
                                                batch_size=self.args.batch_size,
                                                collate_fn=collate_custom)# do no thing to dataset
        #test_loader = 0.2 * origin data
        vali_loader = DataLoader(myDataset(self.args,
                                           self.features_vali, self.targets_vali,
                                           self.edges_index,
                                           self.time_slots_vali,
                                           self.std,
                                           self.mean
                                           ),                                           
                                           shuffle=False,
                                           batch_size=self.args.batch_size,
                                           collate_fn=collate_custom)# do no thing to dataset
        #vali_loader = 0.1 * origin data
        test_loader = DataLoader(myDataset(self.args,
                                           self.features_test, self.targets_test,
                                           self.edges_index,
                                           self.time_slots_test,
                                           self.std,
                                           self.mean
                                           ),
                                           shuffle=False,
                                           batch_size=self.args.batch_size,
                                           collate_fn=collate_custom)# do no thing to dataset

        return (train_loader, vali_loader, test_loader)
    
def load_SE(args, data_name):
    # default ./traffic_dataset/processed/SE/xxx/xxx.pkl
    SE_file_dir = f"{args.data_path}/processed/SE"
    all_SE_path = os.listdir(SE_file_dir)
    # TODO
    all_SE = []
    for index, (p, q) in enumerate(list(itertools.product([1/4,1/2,1,2,4],[1/4,1/2,1,2,4]))):
        this_folder_name = f"p={p},q={q}"
        SE_file_full_name = f"{SE_file_dir}/{this_folder_name}/{data_name}.pkl"
        with open(SE_file_full_name,mode="r") as f:
            all = f.read().split("\n")
            num_dim, id_embeddings = all[0].split(" "), all[1:-1]
            node_num, emb_dim = tuple(map(int,num_dim))
            node_embs = [[] for _ in range(node_num)]
            for i in id_embeddings:
                # last line is " "
                id_embedding = list(map(float, i.split(" ")))
                node_id = int(id_embedding[0])
                node_emb = id_embedding[1:]
                node_embs[node_id] = node_emb
        all_SE.append(node_embs)
    return torch.tensor(all_SE)

def load_multi_domain(args):
    # assert group in ['same_domain', "multi_domain"]
    #PEMSD3没speed
    # datasets = ["METR_LA","PEMSD4"] #交通
    # data_names = ["METR_LA","PEMSD4","PEMSD7(M)","PEMSD8","PEMS_BAY","SZ_TAXI"] #交通
    data_names = args.dataset_names #交通
    # datasets = ["METR_LA","PEMSD4","PEMSD7(M)","PEMSD8","PEMS_BAY"] #交通
    #"METR_LA"：34249,"PEMSD4"：16969,"PEMSD7(M)"：12649,"PEMSD8"：17833,"PEMS_BAY"：52093,"SZ_TAXI：2953"

    client_Data = {}
    df = pd.DataFrame()

    raw_data_path = "./traffic_dataset/raw/"
    processed_data_path = "./traffic_dataset/processed/"
    multi_datasets = {}
    for data_name in data_names:
        loader=process_dataloader(args, data_name=data_name)
        data_loaders = loader.get_dataset(num_timesteps_in=12,num_timesteps_out=12)
        # 有34249个样本，每个样本有207个节点，每个节点包含his的12个值以及pred的12个值
        spacial_embs = load_SE(args, data_name)
        multi_datasets[data_name] = (data_loaders, spacial_embs)
        # dataloader->dataset->[(data, time_stamp), ...]; data->(x, edge_index, ...)
        print(f'''--data_name:{data_name}
                --counts:{len(data_loaders[0])}
                --se_shape:{spacial_embs.shape}'''
              )

    return multi_datasets

if __name__ == "__main__":
    class dummyArgs():
        def __init__(self) -> None:
            self.data_path = "./traffic_dataset"
            self.dataset_names = ["METR_LA"]
            self.current_dataset = "METR_LA"
            self.SE_nums = 25
            self.aggregation_rate = 0.1
            self.aggregation_size = None
            self.shrink_rate = 0.5
            self.batch_size = 20
            self.hid_dim = 64
            self.feature_dim = 1
            self.S_embed_dim = 64
            self.S_embed_num = 25
            self.S_dim = 64
            self.base_num = 16
            # 288 + 12 + 7 = 307
            self.T_dim = 307
            self.ST_hid_dim = 64
            self.node_num = 207
            self.atten_num_heads = 4
            self.drop_out = 0.1
            self.his_len = 12
            self.fut_len = 12
            self.layer_num = 3
            self.layer_num_t = 3
            self.layer_num_g = 3
            self.epoch_num = 300
            self.save_path = f"./runs/model_test/{self.current_dataset}/test_{datetime.datetime.today().strftime(r'%Y-%m-%d-%H-%M')}"
            self.resume = False
            self.resume_epoch = 209
            self.checkpoint_path = f"./runs/model_test/{self.current_dataset}/check_point/model layernorm"
            self.lr = 1e-3
    args = dummyArgs()
    data_loaders = load_multi_domain(args)
    load_SE(args, "METR_LA")
    ((train_loader, vali_loader, test_loader), SE) = data_loaders[args.current_dataset]
    
    for batch in train_loader:
        # print(batch)
        
        pass

