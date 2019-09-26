import csv
import numpy as np
from sklearn.model_selection import train_test_split


def load_data_as_sessions_dict(path2submits, path2requests, seconds_per_bucket=60):
    sessions = dict()
    with open(path2submits, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = [row for row in reader][1:]
        worker_ids = [row[-1] for row in data]
        data = np.array([[int(nr) for nr in row[:-1]] for row in data])
        data = make_buckets(data, seconds_per_bucket)
        for i, worker_id in enumerate(worker_ids):
            sessions[worker_id] = {'submits': data[i]}
    with open(path2requests, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = [row for row in reader][1:]
        worker_ids = [row[-1] for row in data]
        data = np.array([[int(nr) for nr in row[:-1]] for row in data])
        data = make_buckets(data, seconds_per_bucket)
        for i, worker_id in enumerate(worker_ids):
            sessions[worker_id]['requests'] = data[i]
    return sessions


def make_buckets(data, seconds_per_bucket):
    bucketed_data = []
    for line in data:
        bucketed_data.append([sum(line[j*seconds_per_bucket:(j+1)*seconds_per_bucket]) for j in range(int(len(line)/seconds_per_bucket))])
    return np.array(bucketed_data)


def add_avoiders_undetermined_and_seekers(sessions):
    #print(median_requests)
    for key in sessions.keys():
        s = sessions[key]
        s['target'] = -1 if sum(s['requests']) < 2  else 0 if sum(s['requests']) < 4 else 1
    return sessions


def split_worker_ids_into_train_test(sessions, train_percentage):
    worker_ids = list(sessions.keys())
    train_worker_ids, test_worker_ids = train_test_split(worker_ids, train_size=train_percentage)
    return train_worker_ids, test_worker_ids


def create_train_test_dataset(nr_of_buckets, train_worker_ids, test_worker_ids, sessions, train_percentage):
    # this takes the sessions dict and splits it into distinct worker sessions
    # then it concatenates the requests and submits into an array for x_train, x_test
    # and it adds the avoiders and seekers into an array for y_train, y_test
    x_train, x_test, y_train, y_test = [], [], [], []
    for worker_id in sessions.keys():
        submits  = sessions[worker_id]['submits' ]
        requests = sessions[worker_id]['requests']
        instance = np.concatenate((submits[:nr_of_buckets], requests[:nr_of_buckets]))
        if worker_id in train_worker_ids:
            x_train.append(instance)
            y_train.append(sessions[worker_id]['target'])
        if worker_id in test_worker_ids:
            x_test.append(instance)
            y_test.append(sessions[worker_id]['target'])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
            
    

if __name__=='__main__':
    sessions = load_data_as_sessions_dict('iui20_ideaSubmits.csv', 'iui20_inspirationRequests.csv')
    print(sessions.keys())
    print(sessions[list(sessions.keys())[0]])









