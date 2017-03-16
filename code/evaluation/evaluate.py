import os
import numpy as np
from scipy.spatial.distance import cosine


def ap_func(distances, target_args):
    """
    Help function to calculate average precision of distances given the target
    arguments.
    Args:
        distances: Vector with distances to a query vector, the distance to the
            query vector is included.
        target_args: Vector with target arguments.
    Returns:
        ap_score: Average precision.
    """
    arg_sorted = distances.argsort()[1:]
    target_function = np.vectorize(lambda x: x in target_args)
    hit = target_function(arg_sorted)
    cum_hit = np.cumsum(hit)
    uni_hit, uni_idx = np.unique(cum_hit, return_index=True)
    if uni_hit[0] == 0:
        uni_idx = uni_idx[1:]
    precision = np.array(range(1, (len(target_args) + 1))) / (uni_idx + 1)
    ap_score = precision.sum() / len(target_args)
    return ap_score


def evaluate(
    model, path_to_eval='../../data/sun/',
    path_to_words='../../data/scenes_vec.npy'
):
    """
    Takes in a model and a path to evaluation data arranged in groups of
    directories. The first 2 images in each directory will serve as queries
    and the name of the directory will serve as explanatory word for
    similarity.
    Args:
        model: Model to be evaluated, class created by the script in
            'code/network_related/slim_inception_v3/slim_inception_v3/.py'
        path_to_eval: String which defines the path to the directory containing
            the evaluation data.
    Returns:
        map_score: Mean average precision score for image retrieval.
    """

    # Load
    group_vecs = np.load(path_to_words).item()
    group_names = list(group_vecs.keys())

    # Build matrix of the vectors produced by the model for each image in the
    # evaluation dataset.
    list_directories = os.listdir(path_to_eval)
    if list_directories[0] == '.DS_Store':
        list_directories = list_directories[1:]
    nr_groups = len(list_directories)
    vectors_matrix = np.zeros([300, 20 * nr_groups])
    for i, explanatory_word in enumerate(list_directories):
        list_images = os.listdir(path_to_eval + explanatory_word + '/')
        if list_images[0] == '.DS_Store':
            list_images = list_images[1:]
        list_images = [
            path_to_eval + explanatory_word + '/' + image
            for image in list_images
        ]
        predictions = model.predict(list_images)
        vectors_matrix[:, (20 * i):(20 * (i + 1))] = predictions.transpose()
        print('Done predicting: ' + explanatory_word)

    # Calculate MAP with the 2 first in each category as query.
    map_score = 0.0
    for i in range(nr_groups):
        print(group_names[i])
        target = list(range(20 * i, 20 * (i + 1)))
        del target[0]
        # Query 1
        query_vec = vectors_matrix[:, (20 * i)]
        dists = np.apply_along_axis(
            cosine, 0, vectors_matrix, query_vec
        )
        ap_1 = ap_func(dists, target)
        # Query 2
        target = list(range(20 * i, 20 * (i + 1)))
        del target[1]
        query_vec = vectors_matrix[:, (20 * i + 1)]
        dists = np.apply_along_axis(
            cosine, 0, vectors_matrix, query_vec
        )
        ap_2 = ap_func(dists, target)
        if ((ap_1 + ap_2) / 2.0) > 0.01459:
            print(
                'AP, ' + group_names[i] + ': ' + str((ap_1 + ap_2) / 2.0) +
                '     OK'
            )
        else:
            print('AP, ' + group_names[i] + ': ' + str((ap_1 + ap_2) / 2.0))
        map_score += (ap_1 + ap_2)

    map_score = map_score / (2.0 * nr_groups)
    return map_score

    # Split each group in subsets of k images, and find nearest explanatory
    # word to the mean vector of the k images.
    # for key, value in d.items():
