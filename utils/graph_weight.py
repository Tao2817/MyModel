import numpy as np
from numpy import arcsin, sqrt, sin, cos
import pandas as pd
from ast import literal_eval

def calculate_geo_distance(coordinates:((int,int),(int,int))):
    coordinates = np.array(coordinates)
    # to radians
    coor_radians = coordinates * (np.pi / 180)
    # radius of earth is 6471km
    r = 6370996.81
    # 经度纬度 longitude, latitude  
    lo1,lo2 = coor_radians[0][0], coor_radians[1][0]
    la1,la2 = coor_radians[0][1], coor_radians[1][1]
    # 经度差，纬度差
    d_lo = lo2 - lo1
    d_la = la2 - la1
    # https://en.wikipedia.org/wiki/Haversine_formula
    distance = 2 * r * arcsin(
                             sqrt(
                                sin(d_la/2)**2 + cos(la1)*cos(la2)*(sin(d_lo/2)**2)
                                )
                            )
    return distance

def calculate_weights(rel_file, geo_file):
    #calculate the weights by geo distance
    geo_ids = list(geo_file['geo_id'])
    geo_2_ind = {}
    for index, idx in enumerate(geo_ids):
        geo_2_ind[idx] = index
    edges = rel_file[['origin_id', 'destination_id']].values
    geo_ids_to_coordinates = {i[0]:literal_eval(i[1]) for i in geo_file[["geo_id","coordinates"]].values}
    coordinate_pairs = [[geo_ids_to_coordinates[edge[0]],geo_ids_to_coordinates[edge[1]]] for edge in edges]
    distances = np.array([calculate_geo_distance(pair) for pair in coordinate_pairs])
    weights = np.exp(-(distances/distances.std())**2)
    # weights are orderd as same as rel_file
    return weights
    # distance = calculate_geo_distance(coordinates_tuple)
        
if __name__ == "__main__":
    # 计算故宫东西长度 == 886m
    # 左端点(116.398448,39.919224)，右端点(116.408823,39.919649),(经度，纬度)
    assert calculate_geo_distance(((116.398448,39.919224),(116.408823,39.919649))) - 886 <= 1

    geo_file = pd.read_csv(f"./traffic_dataset/raw/METR_LA/METR_LA.geo")#坐标
    rel_file = pd.read_csv(f"./traffic_dataset/raw/METR_LA/METR_LA.rel")#图
    calculate_weights(rel_file, geo_file)