import numpy as np
from clean_utils.io import get_features_and_labels
from config import *
from tools.decorators import load_ctx, save_ctx
from os import path
import os


def __random_partial_reverse(flux):
    """
    reverse parts of the flux randomly
    this function is complete copied from "../utils/augments.py"
    the reason is about tqdm
    """
    res = np.copy(flux)
    start = None
    end = None
    q1, q3 = int(len(res) * 0.4), int(len(res) * 0.6)
    threshold = 0.1
    counter = 0  # control how many times to try
    while start is None:
        if counter > 10:
            return flux  # fail to find any reversible part
        flag = np.random.random()
        if flag > 0.5:
            start, end = np.random.randint(q3, len(res), size=2)
            if start > end:
                start, end = end, start
            if start == end or np.median(np.abs(flux[start:end])) > threshold:
                start = None
        else:
            start, end = np.random.randint(0, q1, size=2)
            if start > end:
                start, end = end, start

            if start == end or np.median(np.abs(flux[start:end])) > threshold:
                start = None
        counter += 1

    res[start:end] = res[start:end][::-1]
    return res


def __transit_reverse(flux):
    # not working
    res = flux.copy()
    midpoint = int(0.5 * len(flux))
    max_width = int(0.1 * len(flux))
    q1 = midpoint - max_width
    q3 = midpoint + max_width
    width = None
    counter = 0
    threshold = 0.4
    while width is None:
        if counter > 10:
            return flux
        width = np.random.randint(max_width)

        if np.median(np.abs(res[midpoint - width:midpoint + width])) < threshold:
            width = None
        counter += 1
    res[midpoint - width:midpoint + width] = \
        res[midpoint - width:midpoint + width][::-1]
    return res


def __full_reverse(flux):
    return flux[::-1]


def __translation(flux):
    """
    probabaly translation is not a good idea, because during test,
    all main tce is centered
    """
    idx = np.random.randint(0, len(flux))
    while np.abs(flux[idx]) > 0.1:
        # if idx is in the trainsit, get another idx
        idx = np.random.randint(0, len(flux))
    return np.concatenate(np.split(flux, [idx])[::-1])


def augment_data_transit(global_fluxes,
                         local_fluxes,
                         labels,
                         shuffle=True):
    aug_local_fluxes, aug_labels = None, None
    if local_fluxes is not None:
        assert len(global_fluxes) == len(local_fluxes), \
            "length of global_fluxes and local_fluxes are not equal"
    aug_global_fluxes = np.concatenate([__transit_reverse(flux).reshape(1, -1)
                                        for flux in global_fluxes])

    if local_fluxes is not None:
        aug_local_fluxes = np.concatenate([__transit_reverse(flux).reshape(1, -1)
                                           for flux in local_fluxes])
    if len(global_fluxes.shape) == 3:
        aug_global_fluxes = aug_global_fluxes.reshape(
            *aug_global_fluxes.shape, 1)
        if local_fluxes is not None:
            aug_local_fluxes = aug_local_fluxes.reshape(
                *aug_local_fluxes.shape, 1)

    aug_global_fluxes = np.concatenate([global_fluxes, aug_global_fluxes])
    if local_fluxes is not None:
        aug_local_fluxes = np.concatenate([local_fluxes, aug_local_fluxes])

    if labels is not None:
        aug_labels = np.tile(labels, 2)
        assert len(aug_global_fluxes) == len(
            aug_labels), "length of 'aug_fluxes' and 'aug_labels' is not the same"

    if shuffle:
        shuffle_indices = np.random.choice(
            np.arange(len(aug_global_fluxes)), size=len(aug_global_fluxes),
            replace=False)
        aug_global_fluxes = aug_global_fluxes[shuffle_indices]

        if local_fluxes is not None:
            aug_local_fluxes = aug_local_fluxes[shuffle_indices]
        if labels is not None:
            aug_labels = aug_labels[shuffle_indices]

    return aug_global_fluxes, aug_local_fluxes, aug_labels


def augment_data_partial(global_fluxes,
                         local_fluxes=None,
                         labels=None,
                         n=3,
                         shuffle=True):
    aug_local_fluxes, aug_labels = None, None
    aug_global_fluxes = np.concatenate(
        [__random_partial_reverse(flux).reshape(1, -1)
         for _ in range(n) for flux in global_fluxes]
    )
    if local_fluxes is not None:
        # aug_local_fluxes = np.concatenate([local_fluxes for _ in range(n)])
        aug_local_fluxes = np.concatenate(
            [__random_partial_reverse(flux).reshape(1, -1)
             for _ in range(n) for flux in local_fluxes]
        )
    if labels is not None:
        aug_labels = np.concatenate([labels for _ in range(n)])
    if len(global_fluxes.shape) == 3:
        aug_global_fluxes = aug_global_fluxes.reshape(
            *aug_global_fluxes.shape, 1)

        aug_local_fluxes = aug_local_fluxes.reshape(
            *aug_local_fluxes.shape, 1)

    if shuffle:
        shuffle_idx = np.random.choice(
            np.arange(len(aug_global_fluxes)), size=len(aug_global_fluxes), replace=False)
        aug_global_fluxes = aug_global_fluxes[shuffle_idx]
        if local_fluxes is not None:
            aug_local_fluxes = aug_local_fluxes[shuffle_idx]
        if labels is not None:
            aug_labels = aug_labels[shuffle_idx]

    return aug_global_fluxes, aug_local_fluxes, aug_labels


def augment_data(global_fluxes,
                 local_fluxes=None,
                 labels=None,
                 features=None,
                 shuffle=True):
    """
    call "full_reverse" function \n
    labels for "global_fluxes" and "local_fluxes" are the same,
    so only have 1 labels argument \n
    NOTE: only "random_partial_reverse" the global view,
    the local view is only repeated
    """
    aug_local_fluxes, aug_labels = None, None
    if local_fluxes is not None:
        assert len(global_fluxes) == len(local_fluxes), \
            "length of global_fluxes and local_fluxes are not equal"
    aug_global_fluxes = np.concatenate([__full_reverse(flux).reshape(1, -1)
                                        for flux in global_fluxes])

    if local_fluxes is not None:
        aug_local_fluxes = np.concatenate([__full_reverse(flux).reshape(1, -1)
                                           for flux in local_fluxes])

    if len(global_fluxes.shape) == 3:
        aug_global_fluxes = aug_global_fluxes.reshape(
            *aug_global_fluxes.shape, 1)
        if local_fluxes is not None:
            aug_local_fluxes = aug_local_fluxes.reshape(
                *aug_local_fluxes.shape, 1)

    aug_global_fluxes = np.concatenate([global_fluxes, aug_global_fluxes])
    if local_fluxes is not None:
        aug_local_fluxes = np.concatenate([local_fluxes, aug_local_fluxes])

    if labels is not None:
        aug_labels = np.tile(labels, 2)
        assert len(aug_global_fluxes) == len(
            aug_labels), "length of 'aug_fluxes' and 'aug_labels' is not the same"

    if features is not None:
        aug_features = np.concatenate([features, features])

    if shuffle:
        shuffle_indices = np.random.choice(
            np.arange(len(aug_global_fluxes)), size=len(aug_global_fluxes),
            replace=False)
        aug_global_fluxes = aug_global_fluxes[shuffle_indices]

        if local_fluxes is not None:
            aug_local_fluxes = aug_local_fluxes[shuffle_indices]
        if labels is not None:
            aug_labels = aug_labels[shuffle_indices]
        if features is not None:
            aug_features = aug_features[shuffle_indices]
    if features is not None:
        res = (aug_global_fluxes, aug_local_fluxes,
               aug_labels, aug_features)
    else:
        res = aug_global_fluxes, aug_local_fluxes, aug_labels
    return res


def load_global_view(train_ratio=0.9, shuffle_idx=None):
    """
    NOTE: it is a generator, NOT a normal function \n

    The reason for this is that "load_global_view"
    and "load_local_view" have to share shuffle indices. \n
    The indices is passed by other function, but this function
    don't know the length of the shuffle indices, which is yield
    by the function.\n

    yield twice
        1. length of all training data
        2. (train_x, train_y), (test_x, test_y) of global view (2001 bins)
    """
    with load_ctx(all_pc_flux_filename):
        all_pcs = np.loadtxt(all_pc_flux_filename)
    with load_ctx(all_non_pc_flux_filename):
        all_non_pcs = np.loadtxt(all_non_pc_flux_filename)
    all_x = np.concatenate([all_pcs, all_non_pcs])

    all_x = all_x.reshape(*all_x.shape, 1)
    all_y = np.concatenate([np.ones(len(all_pcs), dtype=np.int),
                            np.zeros(len(all_non_pcs), dtype=np.int)])
    len_all_x, len_all_y = len(all_x), len(all_y)

    assert len_all_x == len_all_y, f"data and label size different ({len_all_x} != {len_all_y})"

    shuffle_idx = yield len_all_x

    if shuffle_idx is not None:
        all_x = all_x[shuffle_idx]
        all_y = all_y[shuffle_idx]

    num_train = int(len_all_x * train_ratio)
    train_x, test_x = all_x[:num_train], all_x[num_train:]
    train_y, test_y = all_y[:num_train], all_y[num_train:]

    yield (train_x, train_y), (test_x, test_y)


def load_local_view(train_ratio=0.9):
    """
    NOTE: this function don't shuffle
    yield twice
        1. length of all training data
        2. (train_x, train_y), (test_x, test_y) of global view (2001 bins)
    """
    # make sure we use the same shuffle indices
    with load_ctx(local_all_pc_flux_filename):
        local_all_pcs = np.loadtxt(local_all_pc_flux_filename)
    with load_ctx(local_all_non_pc_flux_filename):
        local_all_non_pcs = np.loadtxt(local_all_non_pc_flux_filename)

    all_x = np.concatenate([local_all_pcs, local_all_non_pcs])
    all_x = all_x.reshape(*all_x.shape, 1)
    all_y = np.concatenate(
        [np.ones(len(local_all_pcs), dtype=np.int),
         np.zeros(len(local_all_non_pcs), dtype=np.int)])
    len_all_x, len_all_y = len(all_x), len(all_y)
    assert len_all_x == len_all_y, f"data and label size different ({len_all_x} != {len_all_y})"

    shuffle_idx = yield len_all_x
    # need to shuffle all_x and all_y
    if shuffle_idx is not None:
        all_x = all_x[shuffle_idx]
        all_y = all_y[shuffle_idx]

    num_train = int(len_all_x * train_ratio)

    train_x, test_x = all_x[:num_train], all_x[num_train:]
    train_y, test_y = all_y[:num_train], all_y[num_train:]

    yield (train_x, train_y), (test_x, test_y)


def load_data(train_ratio=0.9, more_features=False):
    # main function
    """
    returns:
        (g_train_x, l_train_x, train_y), (g_test_x, l_test_x, test_y
    """
    g1 = load_global_view(train_ratio)
    g2 = load_local_view(train_ratio)
    length1, length2 = next(g1), next(g2)

    assert length1 == length2

    print("shffuling...")
    shuffle_idx = np.random.choice(
        np.arange(length1), size=length1, replace=False)

    (g_train_x, g_train_y), (g_test_x, g_test_y) = g1.send(shuffle_idx)
    (l_train_x, l_train_y), (l_test_x, l_test_y) = g2.send(shuffle_idx)

    num_train = int(train_ratio * length1)

    if more_features:
        features, labels = get_features_and_labels()
        assert length1 == len(features) and length1 == len(labels)

        features = features[shuffle_idx]
        labels = labels[shuffle_idx]

        train_features, test_features = \
            features[:num_train], features[num_train:]
        train_labels, test_labels = labels[:num_train], labels[num_train:]

        assert np.all(g_test_y == test_labels) and \
               np.all(g_train_y == train_labels)

    assert np.all(g_test_y == l_test_y) and \
           np.all(g_train_y == l_train_y), "global labels and local labels are not same"

    # because global/local labels are the same
    train_y = g_train_y
    test_y = g_test_y
    print("finised shuffing")

    assert len(g_train_x) == len(l_train_x) and \
           len(g_test_x) == len(l_test_x), \
        "number of global/local training samples are not the same"

    # train_features =
    if not more_features:
        return (g_train_x, l_train_x, train_y), \
               (g_test_x, l_test_x, test_y)
    else:
        return (g_train_x, l_train_x, train_y), \
               (g_test_x, l_test_x, test_y), \
               (train_features, test_features)


def train_val_split_generator(n_fold, *data):
    num_each_fold = len(data[0]) // n_fold
    i = 0
    while True:
        val = [x[i:i + num_each_fold] for x in data]
        train = [np.concatenate([x[:i], x[i + num_each_fold:]])
                 for x in data]

        yield list(zip(train, val))
        i += 1
        if i == n_fold:
            i = 0


def get_model_summary(model, test_x, test_y, threshold=0.5):
    """
    returns:
        Accuracy, Precision, Recall, F1 and AUC
    """
    if not isinstance(test_x, list):
        print("input should be a list")
        return
    pred = np.ravel(model.predict(test_x))
    pred = np.where(pred > threshold, 1, 0).astype(np.int)
    # compare the pred with test_y
    assert len(pred) == len(test_y), \
        "prediction size is not consistent with test label size"

    pred = pred.astype(np.int)
    test_y = test_y.astype(np.int)

    TP = np.sum((pred == 1) & (test_y == 1))
    TN = np.sum((pred == 0) & (test_y == 0))
    FP = np.sum((pred == 1) & (test_y == 0))
    FN = np.sum((pred == 0) & (test_y == 1))

    acc = np.sum(pred == test_y) / len(pred)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2.0 / (1.0 / precision + 1.0 / recall)
    return {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1': F1}


def save_train_test_data(rootdir,
                         g_train_x,
                         l_train_x,
                         train_y,
                         g_test_x,
                         l_test_x,
                         test_y,
                         train_features=None,
                         test_features=None):
    for data in (g_train_x, l_train_x, g_test_x, l_test_x):
        assert len(data.shape) == 3 and data.shape[-1] == 1

    if not path.exists(rootdir):
        os.makedirs(rootdir)
    with save_ctx(path.join(rootdir, 'global_train_x.txt')):
        np.savetxt(path.join(rootdir, 'global_train_x.txt'),
                   g_train_x, fmt='%.6f')

    with save_ctx(path.join(rootdir, 'local_train_x.txt')):
        np.savetxt(path.join(rootdir, 'local_train_x.txt'),
                   l_train_x, fmt='%.6f')

    if train_features is not None:
        with save_ctx(path.join(rootdir, 'train_features.txt')):
            np.savetxt(path.join(rootdir, 'train_features.txt'),
                       train_features, fmt='%.6f')

    with save_ctx(path.join(rootdir, 'train_y.txt')):
        np.savetxt(path.join(rootdir, 'train_y.txt'), train_y, fmt='%d')

    with save_ctx(path.join(rootdir, 'global_test_x.txt')):
        np.savetxt(path.join(rootdir, 'global_test_x.txt'),
                   g_test_x, fmt='%.6f')

    with save_ctx(path.join(rootdir, 'local_test_x.txt')):
        np.savetxt(path.join(rootdir, 'local_test_x.txt'),
                   l_test_x, fmt='%.6f')

    if test_features is not None:
        with save_ctx(path.join(rootdir, 'test_features.txt')):
            np.savetxt(path.join(rootdir, 'test_features.txt'),
                       test_features, fmt='%.6f')

    with save_ctx(path.join(rootdir, 'test_y.txt')):
        np.savetxt(path.join(rootdir, 'test_y.txt'), test_y, fmt='%d')
