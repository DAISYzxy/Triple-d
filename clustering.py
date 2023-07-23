from utils import *
from tqdm import *
from sklearn.metrics.pairwise import cosine_similarity



# manual annotate the labels
def logN_labels(relation_cluster, data_size):
    idx = 1
    label_dict = dict()
    while idx <= data_size:
        print("Sentence: " + relation_cluster["sen"][idx - 1])
        print("head: " + relation_cluster["head"][idx - 1])
        print("tail: " + relation_cluster["tail"][idx - 1])
        print("DS Label: " + relation_cluster["relation"][idx - 1])
        label_dict[idx - 1] = input("The DS Label is True('1')/Noise('0'): ")
        idx *= 2
    return label_dict




def init(exp_cluster, label_dict):
    for idx in range(len(exp_cluster)):
        exp_cluster["ada_embedding"][idx] = str2lst(exp_cluster["ada_embedding"][idx])
    if '1' in label_dict:
        init_clutser1_idx = list (label_dict.keys()) [list (label_dict.values()).index ('1')]
    else:
        for idx in range(len(exp_cluster)):
            if idx not in label_dict:
                init_clutser1_idx = idx
                break
    if '0' in label_dict:
        init_clutser0_idx = list (label_dict.keys()) [list (label_dict.values()).index ('0')]
    else:
        for idx in range(len(exp_cluster)):
            if idx not in label_dict:
                init_clutser0_idx = idx
                break
    cluster0 = []
    cluster1 = []
    cluster0_idx = []
    cluster1_idx = []
    cluster0_idx.append(init_clutser0_idx)
    cluster1_idx.append(init_clutser1_idx)
    center0 = exp_cluster["ada_embedding"][init_clutser0_idx]
    center1 = exp_cluster["ada_embedding"][init_clutser1_idx]
    cluster0.append(np.array(center0).astype(np.float32))
    cluster1.append(np.array(center1).astype(np.float32))
    return cluster0_idx, cluster1_idx, cluster0, cluster1, exp_cluster




# # Full Neighbors Retrieving
# def cosine_similarity_matrix(dataset):
#     S = np.zeros([len(dataset),len(dataset)])
#     for i in tqdm(range(len(dataset))):
#         for j in range(len(dataset)):
#             point_i = dataset["ada_embedding"][i]
#             point_i = np.array([point_i])
#             point_i = point_i.astype(np.float64)
#             point_j = dataset["ada_embedding"][j]
#             point_j = np.array([point_j])
#             point_j = point_j.astype(np.float64)
#             S[i][j] = cosine_similarity(point_i, point_j)
#             S[j][i] = S[i][j]
#     return S




# def find_neighbor(S_row, c_eps):
#     neighborsIdx = []
#     for idx in range(len(S_row)):
#         if S_row[idx] > c_eps:
#             neighborsIdx.append(idx) 
#     return neighborsIdx






def add_ranked_position(idx, idx_list, position_list, mod_len):
    while True:
        idx_inPos = idx_list.index(idx)
        if idx_inPos in position_list:
            idx += 1
            idx = idx % mod_len
        else:
            position_list.append(idx_inPos)
            return position_list, idx_inPos
    return





def add_random_pos(position_list, length):
    rnd_num = random.randint(0, length - 1)
    while True:
        if rnd_num in position_list:
            rnd_num = random.randint(0, length - 1)
        else:
            position_list.append(rnd_num)
            return position_list, rnd_num
    return



# Sparse Neighbors Retrieving
def three_blocks(data, position_list, idx_list, sw, idx, target_idx, w):
    mod_len = len(data)
    # building block 1
    for i in range(1, w+1):
        if len(position_list) == mod_len:
            return position_list, sw
        position_list.append((idx+i)%mod_len)
        sw.append(data["ada_embedding"][(idx+i)%mod_len])
    # building block 2
    for i in range(1, w+1):
        if len(position_list) == mod_len:
            return position_list, sw
        p_idx = (target_idx + i) % mod_len
        position_list, p_idx = add_ranked_position(p_idx, idx_list, position_list, mod_len)
        sw.append(data["ada_embedding"][p_idx])
    # building block 3
    for i in range(w):
        if len(position_list) == mod_len:
            return position_list, sw
        position_list, rnd_idx = add_random_pos(position_list, mod_len)
        sw.append(data["ada_embedding"][rnd_idx])
    return position_list, sw




def building_blocks(data, w=30):
    mod_len = len(data)
    idx_list = [i for i in range(mod_len)]
    arr = []
    for i in range(len(data)):
        arr.append(count_words(data["sen"][i]))
    quickSort(arr, idx_list, 0, mod_len - 1)
    position_list = dict()
    similarity_sw = dict()
    for idx in tqdm(range(len(data))):
        target_idx = idx_list.index(idx)
        position_list[idx] = []
        sw = []
        position_list[idx].append(idx)
        sw.append(data["ada_embedding"][idx])
        position_list[idx], sw = three_blocks(data, position_list[idx], idx_list, sw, idx, target_idx, w)
        # similarity calculation
        similarity_sw[idx] = cosine_similarity([data["ada_embedding"][idx]], sw)
    return similarity_sw, position_list





        

def retrieve_neighbors(dataset, threshold=0.90):
    similarity_sw, position_list = building_blocks(dataset)
    neighborIdx = {}
    for idx in range(len(dataset)):
        target_sw = similarity_sw[idx].tolist()[0]
        filtered = filter_indices(target_sw, threshold)
        neighborIdx[idx] = filter_objects_by_indices(position_list[idx], filtered)
    return neighborIdx




def calculate_info_entropy_weights(neighbor_embedding):
    r = np.zeros([len(neighbor_embedding[0]),len(neighbor_embedding)])
    H = np.zeros(len(neighbor_embedding[0]))
    V = np.zeros(len(neighbor_embedding[0]))
    max_x = np.max(neighbor_embedding)
    min_x = np.min(neighbor_embedding)
    ln_m = np.log(len(neighbor_embedding))
    sum_H = 0
    for i in range(len(neighbor_embedding[0])):
        sum_r_i = 0
        for j in range(len(neighbor_embedding)):
            r[i][j] = (max_x - neighbor_embedding[j][i]) / (max_x - min_x)
            if r[i][j] != 0:
                sum_r_i += r[i][j] * np.log(r[i][j])
        if ln_m == 0:
            H[i] = 0
        else:
            H[i] = -(sum_r_i / ln_m)
        sum_H += 1 - H[i]
    for i in range(len(neighbor_embedding[0])):
        V[i] = (1 - H[i]) / sum_H
    return V





def neighbor_embedding_process(dataset, density_neighborsIdx):
    m = len(density_neighborsIdx)
    neighbor_embedding = []
    for idx in density_neighborsIdx:
        neighbor_embedding.append(np.array(dataset["ada_embedding"][idx]))
    neighbor_embedding = np.array(neighbor_embedding).astype(np.float64)
    return neighbor_embedding





def predict(point, idx, cluster0, cluster1, clusterIdx0, clusterIdx1):
    sum0 = 0
    for p in cluster0:
        sum0 += np.sqrt(np.sum(np.square(point - p)))
    sum1 = 0
    for p in cluster1:
        sum1 += np.sqrt(np.sum(np.square(point - p)))
    if sum0 > sum1:
        cluster1.append(point)
        clusterIdx1.append(idx)
    else:
        cluster0.append(point)
        clusterIdx0.append(idx)
    return cluster0, cluster1, clusterIdx0, clusterIdx1





def fit(dataset, label_dict, iter_num, cos):
    cluster0_idx, cluster1_idx, cluster0, cluster1, dataset = init(dataset, label_dict)
    if len(cluster0) == 0:
        for idx in range(len(dataset)):
            if idx not in cluster1:
                center0 = idx
                break
    else:
        center0 = cluster0[0]
    if len(cluster1) == 0:
        for idx in range(len(dataset)):
            if idx not in cluster0:
                center1 = idx
                break
    else:
        center1 = cluster1[0]
    neighborIdx = retrieve_neighbors(dataset, threshold=cos) # sparse neighbors retrieving
#     S = cosine_similarity_matrix(dataset) # full neighbors retrieving
    for idx in range(len(dataset)):
#         neighborIdx = find_neighbor(S[idx], cos) # full neighbors retrieving
#         neighbor_embedding = neighbor_embedding_process(dataset, neighborIdx) # full neighbors retrieving
        neighbor_embedding = neighbor_embedding_process(dataset, neighborIdx[idx]) # sparse neighbors retrieving
        neighbor_embedding = np.array(neighbor_embedding).astype(np.float64)
        V = calculate_info_entropy_weights(neighbor_embedding)
        dataset["ada_embedding"][idx] = (V.T) * (np.array(dataset["ada_embedding"][idx]).astype(np.float64))
    for idx in tqdm(range(iter_num)):
        cluster0 = []
        cluster1 = []
        cluster0_idx = []
        cluster1_idx = []
        for idx in range(len(dataset)):
            if idx in label_dict:
                if label_dict[idx] == "1":
                    cluster1.append(dataset["ada_embedding"][idx])
                    cluster1_idx.append(idx)
                else:
                    cluster0.append(dataset["ada_embedding"][idx])
                    cluster0_idx.append(idx)
            else:
                cluster0, cluster1, cluster0_idx, cluster1_idx= predict(dataset["ada_embedding"][idx], idx, cluster0, cluster1, cluster0_idx, cluster1_idx)
        cluster0 = np.array(cluster0).astype(np.float64)
        center0 = np.average(cluster0, axis=0)
        cluster1 = np.array(cluster1).astype(np.float64)
        center1 = np.average(cluster1, axis=0)
    return cluster0_idx, cluster1_idx






