import numpy as np
import time
import sys

def load_data_old(path,delimiter,label_last,header,y_values):
    
    #x = [line.split(delimiter)[1:] for line in open(path, "r").readlines()]
    x = []
    sorted(y_values)
    i = 0
    for line in open(path, "r").readlines():
        if header and i == 0:
            i = i + 1
            continue
        if label_last:
            z = line.split(delimiter)[:-1]
        else:
            z = line.split(delimiter)[1:]
        i = i + 1
        x.append(z)
    y = []
    i = 0
    for line in open(path, "r").readlines():
        if header and i == 0:
            i = i + 1
            continue
        if label_last:
            label = line.split(delimiter)[-1]
        else:
            label = line.split(delimiter)[0]
        i = i + 1
        y_vec = [0]*len(y_values)
        #print(label)
        y_vec[y_values.index(label)] = 1
        y.append(y_vec)
    return [np.array(x), np.array(y)]

def load_data(path,delimiter,label_last,header,y_values):
    x_list_with_id = []
    y_list = []
    i = 0
    for line in open(path):
        rows = line.replace('\n','').split(delimiter)
        if header and i == 0:
            i = i + 1
            continue
        if label_last:
            z = rows[:-1]
            label = rows[-1]
        else:
            z = rows[1:]
            label = rows[0]
        x_list_with_id.append(z)
        y_vec = [0]*len(y_values)
        #print(label)
        if(label == "neutral"):
            label = "objective"
        y_vec[y_values.index(label)] = 1
        y_list.append(y_vec)

    x_with_id = np.array(x_list_with_id, dtype=float)
    x = x_with_id[:,1:]
    y = np.array(y_list)
    ids = x_with_id[:,0]
    
    return [x, y, ids]
    
def load_data_iter(path,delimiter,label_last,header,y_values, batch_size):
    batch_iter = batch_iter_gen(path, batch_size, 1, label_last, header, delimiter, y_values)
        
    y_list = []
    x_list_with_id = []
    #iterate over generator and append to list
    for batch in batch_iter:
        #unzip batch
        batch_xs_with_id, batch_ys = zip(*batch)
        #append
        y_list = y_list + list(batch_ys)
        x_list_with_id = x_list_with_id + list(batch_xs_with_id)
        
    x_with_id = np.array(x_list_with_id, dtype=float)
    x = x_with_id[:,1:]
    y = np.array(y_list)
    ids = x_with_id[:,0]
    
    return [x, y, ids]
    
def store_data(x_train, y_train, ids, output_path):
    data = np.column_stack((ids, y_train, x_train))
    np.save(output_path, data)
    return None


def load_data_npy(path, n_labels, one_hot=True):
    """
    Load numpy data matrix where first column is ids, followed by label columns and data.

    n_labels specifics number of label columns for one-hot encoded labels (default)
    Returns data, one-hot labels and ids

    :param path: string
        Input data file path
    :param n_labels: int
        Specifics number of label columns
    :param one_hot: boolean
        Flag whether input data is already one hot encoded (default: True)
    :return: array of floats, array of ints, array of ints
        Data array, one hot label array, id array
    """
    data = np.load(path)
    # 0th column for ids
    ids = data[:, 0]
    if not one_hot:
        # Convert to one hot
        y = one_hot_conversion(np.asarray(data[:, 1], dtype=int), n_labels)
        # Data columns
        x = data[:, 2:]
    else:
        # Label column
        y = np.array(data[:, 1:n_labels+1], dtype=int)
        # Data columns
        x = data[:, n_labels+1:]
    return x, y, ids


def one_hot_conversion(labels, n_labels):
    """
    Convert input labels to one hot labels.

    :param labels: array of ints
        Input array of ints
    :param n_labels: int
        Number of unique labels
    :return: array of ints
        One hot label array
    """
    norm_labels = labels - np.min(labels, axis=0)
    # print(y_norm) -> array([0, 1, 2])
    # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    one_hot_labels = np.eye(n_labels, dtype=int)[norm_labels]
    return one_hot_labels
    
    
def load_data_split_npy(path, n_labels, cross_valid_k, one_hot=True):
    """
    Return cross_valid_k datasets of ids, labels, data
    """
    data = np.load(path)
    rows,cols = np.shape(data)
    rem = rows% cross_valid_k
    data = data[:rows-rem]
    np.random.shuffle(data)
    data = np.array_split(data,cross_valid_k)
    for i in range(cross_valid_k):
        data[i] = np.hsplit(data[i],np.array([1,2]))
    return data
    
    
def test_load_data_split_npy(path, n_labels, cross_valid_k, one_hot=True):
    split_data = load_data_split_npy(path, n_labels, cross_valid_k, one_hot=True)
    
    for split_k in split_data:
        #ids k
        ids = split_k[0]
        print(ids[0:10])
        #labels k
        labels = split_k[1]
        print()
        #data k
        x_data = split[2]
        print()
        
    return None


def load_data_lex_npy(path, n_labels, lexicon_dimension, one_hot=True):
    """
    Load numpy data matrix where first column is ids, followed by label columns, data and lexicon data.
    n_labels specifics number of label columns for one-hot encoded labels (default)
    If labels are not one-hot encoded n_labels is irrelevant
    Returns data, lexicon data, one-hot labels and ids
    """
    x_all, y, ids = load_data_npy(path, n_labels, one_hot)
    if lexicon_dimension == 0:
        return x_all, None, y, ids
    #data columns
    x = x_all[:,0:-lexicon_dimension]
    #lexicon data columns
    x_lex = x_all[:,-lexicon_dimension:]
    return x, x_lex, y, ids
           
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch generator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            
def batch_iter_gen(path, batch_size, num_epochs, label_last, header, delimiter, y_values):
    """
    Generates a batch generator for a dataset.
    """
    for epoch in range(num_epochs):
        x = []
        y = []
        i = 0
        for line in open(path):
            if header and i == 0:
                i = i + 1
                continue
            rows = line.replace('\n','').split(delimiter)
            if label_last:
                z = np.array(rows[:-1], dtype=float)
                label = rows[-1]
            else:
                z = np.array(rows[1:], dtype=float)
                label = rows[0]
            x.append(z)
            y_vec = [0]*len(y_values)
            #print(label)
            if(label == "neutral"):
                label = "objective"
            y_vec[y_values.index(label)] = 1
            y.append(y_vec)


            if i%batch_size == batch_size - 1:
                yield zip(x,y)
                x = []
                y = []
            i = i + 1
        if (i-1)%batch_size != batch_size - 1:
            yield zip(x,y)  
        
        

if __name__ == "__main__":
    mode = sys.argv[1]
    data_path = sys.argv[2]
    
    print("Mode: " + mode)
    print("Data: " + data_path)
    
    if mode == "std_load":
        labels = ['positive', 'negative', 'objective']
        labels = sorted(labels)
        
        delimiter = ' '
        label_last = False
        header = False

        print("Loading " + data_path + ", this might take a while....")
        start = time.time()
        x_train, y_train = load_data(data_path,delimiter,label_last,header,labels)
        end = time.time()
        print("Loaded " + str(len(x_train)) + " train data items in " + str(end-start) + " seconds")
    elif mode == "genfromtxt":
        labels = ['positive', 'negative', 'objective']
        labels = sorted(labels)
        
        delimiter = ' '
        label_last = False
        header = False
        print("Loading " + data_path + ", this might take a while....")
        start = time.time()
        data = np.genfromtxt(data_path, delimiter=delimiter, dtype=None)
        end = time.time()
        print("Loaded " + str(len(data)) + " train data items in " + str(end-start) + " seconds")
    elif mode == "np_save":
        output_path = sys.argv[3]

        labels = ['positive', 'negative', 'objective']
        labels = sorted(labels)
        
        delimiter = ' '
        label_last = False
        header = False

        print("Loading " + data_path + ", this might take a while....")
        start = time.time()
        x_train, y_train, ids = load_data(data_path,delimiter,label_last,header,labels)
        print(x_train[0])
        print(len(x_train[0]))
        print(y_train[0])
        print(ids[0])
        store_data(x_train, y_train, ids, output_path)
        end = time.time()
        print("Loaded " + str(len(x_train)) + " train data items in " + str(end-start) + " seconds")
    elif mode == "np_save_iter":
        output_path = sys.argv[3]
        batch_size = 10
        labels = ['positive', 'negative', 'objective']
        labels = sorted(labels)
        
        delimiter = ' '
        label_last = False
        header = False

        print("Loading " + data_path + ", this might take a while....")
        start = time.time()
        x_train, y_train, ids = load_data_iter(data_path,delimiter,label_last,header,labels, batch_size)
        print(x_train[0])
        print(len(x_train[0]))
        print(y_train[0])
        print(ids[0])
        store_data(x_train, y_train, ids, output_path)
        end = time.time()
        print("Loaded " + str(len(x_train)) + " train data items in " + str(end-start) + " seconds")
    elif mode == "np_load_hot":
        n_labels = 3
        print("Loading " + data_path + ", this might take a while....")
        start = time.time()
        x_train, y_train, ids = load_data_npy(data_path, n_labels)
        print(x_train[0])
        print(len(x_train[0]))
        print(y_train[0])
        print(ids[0])
        end = time.time()
        print("Loaded " + str(len(x_train)) + " train data items in " + str(end-start) + " seconds")
    elif mode == "np_load":
        n_labels = 2
        one_hot = False
        print("Loading " + data_path + ", this might take a while....")
        start = time.time()
        x_train, y_train, ids = load_data_npy(data_path, n_labels, one_hot)
        print(x_train[0])
        print(len(x_train[0]))
        print(y_train[0:10])
        print(ids[0])
        end = time.time()
        print("Loaded " + str(len(x_train)) + " train data items in " + str(end-start) + " seconds")
    elif mode =="test_k_fold":
        print("Implement me!")
    
