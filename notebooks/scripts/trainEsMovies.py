from es_movie_dataset import get_dataset, split_data
from training import train

es_data = get_dataset()
es_train_data, es_train_labels, es_test_data, es_test_labels = split_data(es_data)

max_len_char = 5000
hist, clf = train(es_train_data, es_test_data, es_train_labels, es_test_labels, 'es')
