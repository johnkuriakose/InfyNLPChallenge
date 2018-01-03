import sys
import numpy as np
import resource


def load_data_npy(path, n_labels, one_hot=True):
    """
    Load numpy data matrix where first column is ids, followed by label columns and data.

    Loads all data, use when there are no memory concerns.

    :param path: Path to input npy file
    :param n_labels: Number of labels
    :param one_hot: Flag for one-hot encoding of labels (default: True)
    :return: Tuple of all data, one-hot labels, ids
    """
    data = np.load(path)

    # 0th column for ids
    ids = data[:, 0]

    if not one_hot:
        # non one hot encoded labels, can only be one column
        y_basic = np.array(data[:, 1], dtype=int)
        # normalize to start at 0
        y_basic = y_basic - np.min(y_basic, axis=0)
        # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
        y = np.eye(n_labels, dtype=int)[y_basic]
        x = data[:, 2:]
    else:
        # one hot label columns
        y = np.array(data[:, 1:n_labels+1], dtype=int)
        # data columns
        x = data[:, n_labels+1:]
    return x, y, ids


def load_data_npy_mmap(path, n_labels, one_hot=True):
    """
    DEPRECATED: This is a counter example
    Load numpy data matrix where first column is ids, followed by label columns and data.

    Maps input data to memory, but outputs full sized data tuple.

    :param path: Path to input npy file
    :param n_labels: Number of labels
    :param one_hot: Flag for one-hot encoding of labels (default: True)
    :return: Tuple of all data, one-hot labels, ids
    """
    data = np.load(path, mmap_mode='r')

    # 0th column for ids
    ids = data[:, 0]

    if not one_hot:
        # non one hot encoded labels, can only be one column
        y_basic = np.array(data[:, 1], dtype=int)
        # normalize to start at 0
        y_basic = y_basic - np.min(y_basic, axis=0)
        # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
        y = np.eye(n_labels, dtype=int)[y_basic]
        x = data[:, 2:]
    else:
        # one hot label columns
        y = np.array(data[:, 1:n_labels+1], dtype=int)
        # data columns
        x = data[:, n_labels+1:]
    return x, y, ids


def load_data_npy_gen(path, n_labels, one_hot=True):
    """
    Load numpy data matrix where first column is ids, followed by label columns and data.

    Generator implemntation is memory efficient, use when facing memory limitations.

    :param path: Path to input npy file
    :param n_labels: Number of labels
    :param one_hot: Flag for one-hot encoding of labels (default: True)
    :return: Generator of data, one-hot labels, ids items
    """
    data = np.load(path, mmap_mode='r')

    # Get min label for normalization to 0
    min_label = np.min(np.array(data[:, 1], dtype=int), axis=0)
    for row in data:
        # 0th column for ids
        ids = row[0]

        if not one_hot:
            # non one hot encoded labels, can only be one column
            y_basic = np.array(row[1], dtype=int)
            # normalize to start at 0
            y_basic = y_basic - min_label
            # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
            y = np.eye(n_labels, dtype=int)[y_basic]
            x = row[2:]
        else:
            # one hot label columns
            y = np.array(row[1:n_labels+1], dtype=int)
            # data columns
            x = row[n_labels+1:]
        yield x, y, ids


def _test_load_npy_gen(path, n_labels, one_hot=True):
    """
    Calculates test sums on generator data. Memory efficient.

    :param path:
    :param n_labels:
    :param one_hot:
    :return:
    """
    sum_x = 0
    sum_y = 0
    sum_ids = 0
    for x_element, y_element, ids_element in load_data_npy_gen(path, n_labels, one_hot):
        sum_x += x_element[0]
        sum_y += y_element[0]
        sum_ids += ids_element

    print("sum_x: " + str(sum_x))
    print("sum_y: " + str(sum_y))
    print("sum_ids: " + str(sum_ids))
    return None


def _test_load_npy_mmap_counter(path, n_labels, one_hot=True):
    """
    DEPRECATED: Counter-example

    Calculates test sums on processed memory mapped data. Not memory efficient.

    :param path:
    :param n_labels:
    :param one_hot:
    :return:
    """
    x, y, ids = load_data_npy(path, n_labels, one_hot)
    sum_x = np.sum(x, axis=0)
    sum_y = np.sum(y, axis=0)
    sum_ids = np.sum(ids)

    print("sum_x: " + str(sum_x[0]))
    print("sum_y: " + str(sum_y[0]))
    print("sum_ids: " + str(sum_ids))
    return None


def _test_load_npy_mmap(path, n_labels, one_hot=True):
    """
    Calculates test sums directly on memory mapped data. Memory efficient.

    :param path:
    :param n_labels:
    :param one_hot:
    :return:
    """
    data = np.load(path, mmap_mode='r')

    sum_ids = np.sum(data[:, 0])

    min_label = np.min(np.array(data[:, 1], dtype=int), axis=0)
    if not one_hot:
        sum_x = np.sum(data[:, n_labels + 1])
        sum_y = np.sum(data[:, 1]-min_label == 0)
    else:
        sum_x = np.sum(data[:, n_labels + 1])
        sum_y = np.sum(data[:, 1])

    print("sum_x: " + str(sum_x))
    print("sum_y: " + str(sum_y))
    print("sum_ids: " + str(sum_ids))
    return None


def _test_load_npy(path, n_labels, one_hot=True):
    """
    Calculates test sums on processed memory loaded data. Not memory efficient.

    :param path:
    :param n_labels:
    :param one_hot:
    :return:
    """
    x, y, ids = load_data_npy(path, n_labels, one_hot)
    sum_x = np.sum(x, axis=0)
    sum_y = np.sum(y, axis=0)
    sum_ids = np.sum(ids)

    print("sum_x: " + str(sum_x[0]))
    print("sum_y: " + str(sum_y[0]))
    print("sum_ids: " + str(sum_ids))
    return None


def main():
    mode = sys.argv[1]
    path = sys.argv[2]
    n_labels = int(sys.argv[3])
    reps = int(sys.argv[4])
    if mode == "std":
        for _ in range(reps):
            _test_load_npy(path, n_labels)
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Peak Mem: " + str(peak_mem) + " kb")
    elif mode == "mmap":
        for _ in range(reps):
            _test_load_npy_mmap(path, n_labels)
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Peak Mem: " + str(peak_mem) + " kb")
    elif mode == "gen":
        for _ in range(reps):
            _test_load_npy_gen(path, n_labels)
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Peak Mem: " + str(peak_mem) + " kb")
    else:
        print("Invalid mode: " + str(mode))

    return None


if __name__ == "__main__":
    main()
