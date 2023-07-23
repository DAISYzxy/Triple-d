import pandas as pd
import json
import spacy
from spacy import displacy
import os
import numpy as np
import string
import random
NER = spacy.load("en_core_web_sm")



# The pre-defined NER pattern dictionary
ner_dict = {
    "PERSON": "a person",
    "NORP": "a nationality",
    "FAC": "a facility",
    "ORG": "an organization",
    "GPE": "a country",
    "LOC": "a location",
    "PRODUCT": "a product",
    "EVENT": "an event",
    "WORK_OF_ART": "a work of art",
    "LAW": "a law",
    "LANGUAGE": "a language",
    "DATE": "a date",
    "TIME": "a time",
    "PERCENT": "a percentage",
    "MONEY": "a monetray value",
    "QUANTITY": "a quantity",
    "ORDINAL": "an ordinal",
    "CARDINAL": "a cardinal"
}




# Types of relation in NYT10-test dataset
relation_list = [
 '/people/person/place_of_birth',
 '/people/person/nationality',
 '/people/person/place_lived',
 '/location/location/contains',
 '/sports/sports_team/location',
 '/business/person/company',
 '/location/country/capital',
 '/business/company/founders',
 '/film/film/featured_film_locations',
 '/location/administrative_division/country',
 '/location/us_county/county_seat',
 '/people/deceased_person/place_of_death',
 '/business/company/place_founded',
 '/location/country/languages_spoken',
 '/location/neighborhood/neighborhood_of',
 '/people/person/children',
 '/film/film_location/featured_in_films',
 '/people/ethnicity/geographic_distribution',
 '/location/country/administrative_divisions',
 '/location/us_state/capital',
 '/business/company_advisor/companies_advised',
 '/time/event/locations',
 '/people/person/religion',
 '/people/person/ethnicity',
 '/business/company/major_shareholders',
 '/people/place_of_interment/interred_here',
 '/location/province/capital',
 '/people/deceased_person/place_of_burial',
 '/business/company/advisors',
 '/location/br_state/capital',
 '/base/locations/countries/states_provinces_within'
]



def write_list_to_json(lst, json_file_name):
    with open(json_file_name, 'w') as  f:
        json.dump(lst, f)




def replace_with_pattern(raw_text, text1):
    for word in text1.ents:
        if word.text in raw_text:
            raw_text = raw_text.replace(word.text, str(ner_dict[word.label_]))
    return raw_text




def str2lst(embedding):
    lst = embedding.split(",")
    lst = [lst0[1:] for lst0 in lst]
    lst[-1] = lst[-1][:-1]
    return lst


def partition(arr, idx, low, high):
    i = low - 1
    pivot = arr[high]

    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
            idx[i], idx[j] = idx[j], idx[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    idx[i + 1], idx[high] = idx[high], idx[i + 1]
    return (i + 1)




def quickSort(arr, idx, low, high):
    if len(arr) == 1:
        return arr, idx
    if low < high:
        pi = partition(arr, idx, low, high)
        quickSort(arr, idx, low, pi - 1)
        quickSort(arr, idx, pi + 1, high)
    return





def count_words(sentence):
    words = sentence.split()
    num_words = len(words)
    return num_words




def filter_indices(l, value):
    return [i for i, x in enumerate(l) if x > value]



def filter_objects_by_indices(input_list, indices):
    return [input_list[i] for i in indices]



# Process the NYT dataset
def process_raw_nyt(dataset_path="nyt10_test.txt"):
    sentences = pd.read_table(dataset_path, header=None)[0]
    nyt_df = dict()
    nyt_df["sen"] = list()
    nyt_df["head"] = list()
    nyt_df["tail"] = list()
    nyt_df["relation"] = list()
    nyt_df["replaced"] = list()
    for sen in sentences:
        sen = json.loads(sen)
        nyt_df["sen"].append(sen["text"])
        nyt_df["head"].append(sen["h"]["name"])
        nyt_df["tail"].append(sen["t"]["name"])
        nyt_df["relation"].append(sen["relation"])
        NER_sen = NER(sen["text"])
        nyt_df["replaced"].append(replace_with_pattern(sen["text"], NER_sen))
    nyt_df = pd.DataFrame(nyt_df)
    return nyt_df




# Divide the processed NYT dataset into clusters by the ds relation types
def divide_ds_relation(nyt_df, relation_list=relation_list):
    ds_cluster = dict()
    for rel in relation_list:
        ds_cluster[rel] = nyt_df[nyt_df["relation"] == rel].reset_index().drop(columns="index")
    return ds_cluster



def retrieve_label(dataset):
    label_dict = dict()
    binary_idx = 1
    for idx in range(len(dataset)):
        if idx == binary_idx:
            label_dict[idx - 1] = str(int(dataset["label"][idx - 1]))
            binary_idx = binary_idx * 2
    return label_dict



def get_precision(cluster1_idx, dataset):
    sum_pre = 0
    for idx in cluster1_idx:
        if int(dataset["label"][idx]) == 1:
            sum_pre += 1
    precision = round(sum_pre / len(cluster1_idx) * 100, 3)
    return precision



def save_true_clustering(cluster1_idx, dataset, save_path):
    dataset_true = dataset.loc[cluster1_idx].reset_index(drop=True)
    dataset_true.to_csv(save_path, index=False)
    return