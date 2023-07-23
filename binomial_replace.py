from tqdm import tqdm
from utils import *
from numpy import *
from scipy.stats import binom



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


def calculate_probability(n, p, start, end):
    probabilities = [binom.pmf(k, n, p) for k in range(start, end+1)]
    total_probability = sum(probabilities)
    return total_probability




def statistics_in_sentence(raw_text, text1, entity_dict):
    pattern_dict = dict()
    for pattern in ner_dict.keys():
        pattern_dict[pattern] = 0
    for word in text1.ents:
        if word.text in raw_text:
            entity = str(word.text)
            pattern = word.label_
            pattern_dict[pattern] += 1
            if pattern not in entity_dict.keys():
                entity_dict[pattern] = dict()
            if entity in entity_dict[pattern].keys():
                entity_dict[pattern][entity] += 1
            else:
                 entity_dict[pattern][entity] = 1
    return pattern_dict, entity_dict



def get_frequency(path):
    dataset_pattern = dict()
    dataset_entity = dict()
    for relation in relation_list:
        tmp_df = pd.read_csv(path + relation)
        dataset_pattern[relation] = dict()
        dataset_entity[relation] = dict()
        for pattern in ner_dict.keys():
            dataset_pattern[relation][pattern] = 0
        for idx in tqdm(range(len(tmp_df))):
            sen = tmp_df["sen"][idx]
            NER_sen = NER(sen)
            pattern_dict, dataset_entity[relation] = statistics_in_sentence(sen, NER_sen, dataset_entity[relation])
            for pattern in ner_dict.keys():
                if pattern_dict[pattern] > 1:
                    dataset_pattern[relation][pattern] += 1
        for pattern in ner_dict.keys():
            dataset_pattern[relation][pattern] = int(dataset_pattern[relation][pattern] * 100 / len(tmp_df))
            if pattern in dataset_entity[relation].keys():
                for entity in dataset_entity[relation][pattern].keys():
                    dataset_entity[relation][pattern][entity] = int(dataset_entity[relation][pattern][entity] * 100 / len(tmp_df))
    return dataset_pattern, dataset_entity




def retrieve_threshold_K(path, statistics_dict, entry, pattern="None", low_bound=0.01):
    entry_list = []
    for relation in relation_list:
        tmp_df = pd.read_csv(path + relation)
        if pattern == "None":
            entry_list.append(statistics_dict[relation][entry])
        else:
            if pattern in statistics_dict[relation].keys():
                if entry in statistics_dict[relation][pattern].keys():
                    entry_list.append(statistics_dict[relation][pattern][entry])
                else: entry_list.append(0)
            else:
                entry_list.append(0)
    X_bar = round(mean(entry_list) / 100, 2)
    if X_bar == 0: return 0
    else:
        for end in range(2, 100):
            prob = calculate_probability(100, X_bar, 1, end)
            if prob >= low_bound:
                K = end
                return K
    return




def step_func(value):
    if value >= 0: return 1
    else: return 0
    return




def replace_or_not(raw_text, text1, value_dict, dataset):
    for word in text1.ents:
        if (word.text in raw_text) and (value_dict[dataset][word.label_][word.text] == 1):
            raw_text = raw_text.replace(word.text, str(ner_dict[word.label_]))
    return raw_text




def binomial_replacement(value_dict, path="data/", path_saved="data/"):
    for relation in relation_list:
        tmp = pd.read_csv(path + relation)
        for idx in tqdm(range(len(tmp))):
            sen = tmp["sen"][idx]
            NER_sen = NER(sen)
            tmp["replaced"][idx] = replace_or_not(sen, NER_sen, value_dict, relation)
        tmp.to_csv(path_saved + relation, index=False)
    return



def binomial_value_calculate(relation_list, path="data/"):
    dataset_pattern, dataset_entity = get_frequency(path)
    value = dict()
    for relation in relation_list:
        value[relation] = dict()
        for pattern in tqdm(ner_dict.keys()):
            value[relation][pattern] = dict()
            pattern_K = retrieve_threshold_K(path, dataset_pattern, pattern)
            pattern_v = step_func(pattern_K - dataset_pattern[relation][pattern])
            if pattern in dataset_entity[relation].keys():
                for entity in dataset_entity[relation][pattern].keys():
                    if pattern_v == 0:
                        value[relation][pattern][entity] = 0
                    else:
                        entity_K = retrieve_threshold_K(path, dataset_entity, entity, pattern)
                        entity_v = step_func(dataset_entity[relation][pattern][entity] - entity_K)
                        value[relation][pattern][entity] = entity_v
    json_str = json.dumps(value)
    with open(path+'binomial_value.json', 'w') as json_file:
        json_file.write(json_str)
    return value













