import numpy as np
import itertools,os
import networkx as nx
from gensim.models import Word2Vec
from utils import node2vec
from utils.data import load_multi_domain

is_directed = False
p = [1/4,1/2,1,2,4]
q = [1/4,1/2,1,2,4]
p_q = itertools.product(p,q)
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
epochs = 300

SE_file_dir = "./traffic_dataset/processed/SE"

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())

    return G

def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size = dimensions, window = 10, min_count=0, sg=1,
        workers = 20, epochs = epochs)
    model.wv.save_word2vec_format(output_file)
    return

def generate_spatial_embedding(graphs:dict,p,q):
    # nx_G = read_graph(Adj_file)
    for dataset_name, dataset in graphs.items():
        all_folder = os.listdir(f"{SE_file_dir}")
        this_folder_name = f"p={p},q={q}"
        print(f"{this_folder_name}-{dataset_name}")
        if this_folder_name not in all_folder:
            os.mkdir(f"{SE_file_dir}/{this_folder_name}")
        SE_file_full_name = f"{SE_file_dir}/{this_folder_name}/{dataset_name}.pkl"
        if os.path.exists(SE_file_full_name):
            print("pass")
            continue
        edge_index = dataset.edge_index
        edge_num = edge_index.shape[1]
        #[(u,v,weight)...]
        edgelist = [(edge_index[0][i],edge_index[1][i],dataset.edge_weight[i]) for i in range(edge_num)]
        nx_G = nx.Graph()
        nx_G.add_weighted_edges_from(edgelist)
        G = node2vec.Graph(nx_G, is_directed, p, q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(num_walks, walk_length)
        learn_embeddings(walks, dimensions, SE_file_full_name)

if __name__ == "__main__":
    class dummyArgs():
        def __init__(self) -> None:
            self.data_path = "./traffic_dataset"
            self.SE_nums = 25
            self.aggregation_rate = 0.1
            self.aggregation_size = 5
            self.batch_size = 16
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
            self.atten_num_heads = 8
            self.his_len = 12
            self.fut_len = 12
            self.layer_num = 3
            self.layer_num_g = 3
    multi_datasets = load_multi_domain(dummyArgs())
    for p,q in p_q:
        generate_spatial_embedding(multi_datasets,p,q)