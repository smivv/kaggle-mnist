import numbers
import tensorflow as tf

""" --------------------------------- Triplet loss implementation ----------------------------------- """


def _all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.
    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).
    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def _cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's _cdist, but symbolic.
    The currently supported metrics can be listed as `_cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.
    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.
    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    with tf.name_scope("_cdist"):
        diffs = _all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `_cdist` yet: {}'.format(metric))


_cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
]


def _get_at_indices(tensor, indices):
    """ Like `tensor[np.arange(len(tensor)), indices]` in numpy. """
    counter = tf.range(tf.shape(indices, out_type=indices.dtype)[0])
    return tf.gather_nd(tensor, tf.stack((counter, indices), -1))


def batch_hard_loss(features, pids, metric='euclidean', margin=0.1):
    """Computes the batch-hard loss from arxiv.org/abs/1703.07737.
    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by _cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        margin: The value of the margin if a number, alternatively the string
            'soft' for using the soft-margin formulation, or `None` for not
            using a margin at all.
    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
        :param margin:
        :param features:
        :param pids:
        :param metric:
    """
    with tf.name_scope("batch_hard_loss"):

        dists = _cdist(features, features, metric=metric)

        pids = tf.argmax(pids, axis=1)

        exp_dims0 = tf.expand_dims(pids, axis=0)
        exp_dims1 = tf.expand_dims(pids, axis=1)

        same_identity_mask = tf.equal(exp_dims1, exp_dims0)

        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))

        furthest_positive = tf.reduce_max(dists*tf.cast(positive_mask, tf.float32), axis=1)
        # closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),
        #                              (dists, negative_mask), tf.float32)
        # Another way of achieving the same, though more hacky:
        closest_negative = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=1)

        diff = furthest_positive - closest_negative
        if isinstance(margin, numbers.Real):
            diff = tf.maximum(diff + margin, 0.0)
        elif margin == 'soft':
            diff = tf.nn.softplus(diff)
        elif margin is None:
            pass
        else:
            raise NotImplementedError('The margin {} is not implemented in batch_hard_loss'.format(margin))

    return diff


def triplet_loss(labels, features):
    # https://github.com/tensorflow/tensorflow/issues/20253
    # from tensorflow.contrib.losses import metric_learning
    # return metric_learning.triplet_semihard_loss(K.argmax(labels, axis=1), embeddings, margin=0.2)
    return tf.reduce_mean(batch_hard_loss(features, labels, margin=0.2))