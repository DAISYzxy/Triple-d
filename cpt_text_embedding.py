import openai
import json
import pandas as pd
import time
import random


# fill your openai key here:
openai.api_key =''



def get_embedding(text, model):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']




def cpt_text_embedding(data, model="text-embedding-ada-002"):
    data['ada_embedding'] = data["replaced"].apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    for idx in range(0, len(data["ada_embedding"])):
        data["ada_embedding"][idx] = np.array(data["ada_embedding"][idx])
    return



def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list



def load_line_json(filepath):
    data = []
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            ins = json.loads(line.strip())
            data.append(ins)
    return data



def json2df(data_json):
    data = dict()
    data["sen"] = list()
    data["head"] = list()
    data["tail"] = list()
    data["relation"] = list()
    for ins in data_json:
        data["sen"].append(ins["sentence"])
        data["head"].append(ins["head"]["word"])
        data["tail"].append(ins["tail"]["word"])
        data["relation"].append(ins["relation"])
    data_df = pd.DataFrame(data)
    return data_df

