import os
import numpy as np
from scipy.spatial.distance import cosine
import sys
sys.path.insert(0, './code/network_related/slim_inception_v3/')
from slim_inception_v3 import SlimInceptionV3


class Evaluation(object):

    def __init__(self):
        pass

    def ap_func(self, distances, target_args):
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
        arg_sorted = distances.argsort()
        if len(target_args) > 1:
            if (target_args[1] - target_args[0]) > 1:
                mask = arg_sorted != (target_args[0] + 1)
                arg_sorted = arg_sorted[mask]
            else:
                mask = arg_sorted != (target_args[0] - 1)
                arg_sorted = arg_sorted[mask]
        target_function = np.vectorize(lambda x: x in target_args)
        hit = target_function(arg_sorted)
        cum_hit = np.cumsum(hit)
        uni_hit, uni_idx = np.unique(cum_hit, return_index=True)
        if uni_hit[0] == 0:
            uni_idx = uni_idx[1:]
        precision = np.array(range(1, (len(target_args) + 1))) / (uni_idx + 1)
        ap_score = precision.sum() / len(target_args)
        if ap_score > 0.3:
            print(ap_score)
            print(uni_idx)
        return ap_score, hit

    def evaluate(self, model, path_to_eval='./data/sun/', path_to_words='./data/new_scene_names.npy', k=5):
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
        # Build matrix of the vectors produced by the model for each image in the
        # evaluation dataset.
        list_directories = os.listdir(path_to_eval)
        if list_directories[0] == '.DS_Store':
            list_directories = list_directories[1:]
        nr_groups = len(list_directories)
        group_names = list_directories
        image_matrix = np.zeros([300, 20 * nr_groups])
        dic_keys = {j:i for i,j in enumerate(list(group_vecs.keys()))}

        word_matrix = np.zeros([300, nr_groups])
        word_list = []
        for i, j in enumerate(list(group_vecs.values())):
            word_matrix[:, i] = j[0][1]
            word_list.append(j[0][0])

        for i, explanatory_word in enumerate(list_directories):
            list_images = os.listdir(path_to_eval + explanatory_word + '/')
            list_images = [
                path_to_eval + explanatory_word + '/' + image
                for image in list_images if image != '.DS_Store'
            ]
            predictions = model.predict(list_images)
            image_matrix[:, (20 * i):(20 * (i + 1))] = predictions.transpose()
            # print('Done predicting: ' + explanatory_word)

        # Calculate MAP with the 2 first in each category as query.
        image_hits = np.zeros(nr_groups * 20 - 1)
        image_map = 0.0
        for i in range(nr_groups):
            target = list(range(20 * i, 20 * (i + 1)))
            del target[0]
            print(list_directories[i])
            # Query 1
            query_vec = image_matrix[:, (20 * i)]
            dists = np.apply_along_axis(
                cosine, 0, image_matrix, query_vec
            )
            ap_1, hit_1 = self.ap_func(dists, target)
            # Query 2
            target = list(range(20 * i, 20 * (i + 1)))
            del target[1]
            query_vec = image_matrix[:, (20 * i + 1)]
            dists = np.apply_along_axis(
                cosine, 0, image_matrix, query_vec
            )
            ap_2, hit_2 = self.ap_func(dists, target)
            # if ((ap_1 + ap_2) / 2.0) > 0.014916813281349084:
            #     print(
            #         'AP, ' + group_names[i] + ': ' + str((ap_1 + ap_2) / 2.0) +
            #         '     Better than random!'
            #     )
            # else:
            #     print('AP, ' + group_names[i] + ': ' + str((ap_1 + ap_2) / 2.0))
            image_map += (ap_1 + ap_2)
            image_hits = image_hits + hit_1 + hit_2

        image_map /= (2.0 * nr_groups)

        # Split each group in subsets of k images, and find nearest explanatory
        # word to the mean vector of the k images.
        sum_k_rows = np.array(
            [
                image_matrix[:, (k * n):(k * n + k)].sum(1)
                for n in range(int(image_matrix.shape[1] / k))
            ]
        )
        word_map = 0.0
        word_hits = np.zeros(nr_groups)
        for i in range(sum_k_rows.shape[0]):
            target_name = list_directories[int((i * k) / 20)]
            target = [dic_keys[target_name]]
            query_vec = sum_k_rows[i, :]
            dists = np.apply_along_axis(
                cosine, 0, word_matrix, query_vec
            )
            ap, hit = self.ap_func(dists, target)
            # print(
            #     'AP, ' + target_name + '(' + word_list[target[0]]
            #      + '): ' + str(ap)
            # )
            word_map += ap
            word_hits = word_hits + hit
        word_map /= float(sum_k_rows.shape[0])
        return (image_map, word_map, image_hits, word_hits)

if __name__ == "__main__":
    count = 1
    for d in os.listdir('./code/network_related/slim_inception_v3/runs'):
        if str(count) == sys.argv[1]:
            print(d)
            net = SlimInceptionV3()
            net.restore_model(d)
            evaluator = Evaluation()
            map_score = evaluator.evaluate(net)

            # Write map_score to file
            with open('./code/network_related/slim_inception_v3/runs/' + d + '/parameters',
                      'a+') as run_info:
                print('Mean Average Precision:')
                run_info.write('Mean Average Precision:\n')
                print('IMAGE_RETRIEVAL={}'.format(map_score[0]))
                run_info.write('IMAGE_RETRIEVAL={}\n'.format(map_score[0]))
                print('SIMILARITY_MOTIVATION={}'.format(map_score[1]))
                run_info.write('SIMILARITY_MOTIVATION={}\n\n'.format(map_score[1]))
        count += 1
