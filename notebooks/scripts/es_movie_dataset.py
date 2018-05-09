from os import listdir
import os
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
import random
from collections import Counter

def get_dataset():
    failure = 0
    max_len = 5000
    data = []
    script_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = script_dir + "/../../../datasets/sentiment/spanish/CriticaCine/"
    for filename in listdir(dataset_root):
        sentiment = None
        content = None
        summary = None
        try:
            e = ET.parse(dataset_root+filename, parser=ET.XMLParser(encoding="iso-8859-1")).getroot()
        except:
            failure += 1
        sentiment_tmp = int(e.attrib["rank"])
        if sentiment_tmp < 3:
            sentiment = 0
        elif sentiment_tmp == 3:
            sentiment = 1
        elif sentiment_tmp > 3:
            sentiment = 2
        for child in e:
            if child.tag == 'body':
                content = child.text
            if child.tag == 'summary':
                summary = child.text
        if content is None or len(content) > 5000:
            failure += 1
            continue
        data.append((sentiment,content, summary))
    return data

# DANITER : Support cross validation in the future
def split_data(data, perc=.2):
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    balance_set = {}
    for label, content, _ in data:
        if random.random() < perc:
            test_data.append(content)
            test_label.append(label)
        else:
            train_data.append(content)
            if label not in balance_set:
                balance_set[label] = []
            balance_set[label].append(content)
            train_label.append(label)

    # Balance training data
    c = Counter(train_label)
    mode_label, mode_count = c.most_common(1)[0]
    to_add = []
    to_add_labels = []
    for label in c.keys():
        num_missing = mode_count - c[label]
        if num_missing > len(balance_set[label]):
            div = int(num_missing/ len(balance_set[label]))
            to_add.extend(balance_set[label]*div)
            to_add_labels.extend([label]*len(balance_set[label]*div))
            num_missing = num_missing % len(balance_set[label])
        to_add.extend(random.sample(balance_set[label], num_missing))
        to_add_labels.extend([label]*num_missing)
    tmp = list(zip(to_add, to_add_labels))
    random.shuffle(tmp)
    to_add, to_add_labels = zip(*tmp)
    train_data.extend(to_add)
    train_label.extend(to_add_labels)

    return (train_data, train_label, test_data, test_label)
