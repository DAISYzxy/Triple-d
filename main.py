from binomial_replace import *
from cpt_text_embedding import *
from clustering import *
from utils import *
import argparse

# 75% True Label (/25% Noise) Experiments
relation_list = [
    'test_religion75.csv',
    'test_company75.csv',
    'test_death75.csv',
    'test_admin_country75.csv',
    'test_administrative_divisions75.csv',
    'test_capital75.csv',
    'test_children75.csv',
    'test_location75.csv',
    'test_major_shareholders75.csv',
    'test_place_lived75.csv',
    'test_place_of_birth75.csv'
]

 # Generate denoised training data
def run_pipeline(path="data/"):
    # 1. adaptive pattern replacement
    value = binomial_value_calculate(relation_list, path)
    binomial_replacement(value, relation_list, path, path)

    # 2. non-parametric denoising model
        # (1) LLM in the loop - generate embedding
    for rel in tqdm(relation_list):
        path_rel = path + rel
        save_rel = path + rel
        example = pd.read_csv(path_rel)
        # Require the billing account to continously run the API
        example['ada_embedding'] = example["sen"].apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
        example.to_csv(save_rel, index=False)
    
        # (2) Model Fitting
    for rel in tqdm(relation_list):
        read_path = path + rel
        exp_cluster = pd.read_csv(read_path)
        label_dict = logN_labels(exp_cluster, len(exp_cluster))
        cluster0_idx, cluster1_idx = fit(exp_cluster, label_dict, 100, 0.88)
        save_true_clustering(cluster1_idx, exp_cluster, path)
    return


# Test the algorithm performance
def test(path):
    exp_cluster = pd.read_csv(path)
    label_dict = retrieve_label(exp_cluster)
    cluster0_idx, cluster1_idx = fit(exp_cluster, label_dict, 100, 0.88)
    precision = get_precision(cluster1_idx, exp_cluster)
    print("Precision: {0}%".format(precision))
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_path", type=str, default=None,
                        help="The path of input test data that you want to load.")
    args = parser.parse_args()

    test(args.test_path)