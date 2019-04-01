from itertools import permutations


def triplet_generator(x, y, test_size=0.3, ap_pairs=10, an_pairs=10):

    train_size = 1 - test_size

    train_triplets = []
    test_triplets = []
    for data_class in sorted(set(y)):

        same_class_idx = np.where(y == data_class)[0]
        diff_class_idx = np.where(y != data_class)[0]
        pos_idx = random.sample(list(permutations(same_class_idx, 2)), k=ap_pairs)
        neg_idx = random.sample(list(diff_class_idx), k=an_pairs)

        pos_len = len(pos_idx)
        neg_len = len(neg_idx)

        # train
        for ap in pos_idx[:int(pos_len * train_size)]:
            for n in neg_idx:
                train_triplets.append([x[ap[0]], x[ap[1]], x[n]])

        # test
        for ap in pos_idx[int(pos_len * train_size):]:
            for n in neg_idx:
                test_triplets.append([x[ap[0]], x[ap[1]], x[n]])

    return np.array(train_triplets), np.array(test_triplets)