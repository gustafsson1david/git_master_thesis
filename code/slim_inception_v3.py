import os
import sys
import numpy as np
import random
import time
from PIL import Image
from scipy.spatial.distance import cosine
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_arg_scope


class SlimInceptionV3(object):

    def __init__(self):
        self.timestamp = time.strftime('%Y%m%d-%H%M', time.localtime())
        self.graph = tf.get_default_graph()
        self.sess = tf.Session()
        self.saver = None

    def build_graph(self, branch_path='InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0', layers_to_train=[], own_layers=1,
                    init_learning_rate=1e-3, lr_decay_freq=10000.0, lr_decay_factor=0.95,
                    epsilon=0.01,  alpha=1.0, activation='tanh'):
        self.graph.as_default()

        # Write parameters to file
        if not os.path.exists('./runs/'+self.timestamp+'/'):
            os.makedirs('./runs/'+self.timestamp+'/')
        with open('./runs/'+self.timestamp+'/run_info', 'a+') as run_info:
            params = locals()
            print("Graph parameters:")
            run_info.write("Graph parameters:\n")
            for attr, value in params.items():
                if attr != 'self' and attr != 'run_info':
                    print("{}={}".format(attr.upper(), value))
                    run_info.write("{}={}\n".format(attr.upper(), value))
            print("")
            run_info.write("\n")

        # Define preprocessing
        with tf.name_scope('Preprocessing'):
            input_im = tf.placeholder(tf.uint8, shape=[None, None, None, 3], name='input_im')
            resized_im = tf.image.resize_bilinear(tf.image.convert_image_dtype(
                tf.convert_to_tensor(input_im), dtype=tf.float32), [299, 299], name='resized_im')
            normalized_im = tf.multiply(tf.subtract(resized_im, 0.5), 2.0, name='normalized_im')

        # Load Inception-v3 graph from slim
        with slim.arg_scope(inception_v3_arg_scope()):
            inception_v3(normalized_im, num_classes=1001, is_training=False)

        # Create own branch
        with tf.name_scope('Own'):
            y = tf.placeholder(dtype="float", shape=[None, 300], name='y')

            # Branch from original network and shape tensor
            x = self.graph.get_tensor_by_name(branch_path)
            branch_size = int(x.get_shape()[-1])
            x = tf.nn.avg_pool(x, ksize=[1]+list(x.get_shape()[1:3])+[1], strides=[1, 1, 1, 1], padding='VALID')
            x = tf.squeeze(x, [1, 2])

            # Calculate dimensions between own layers
            p = [int(300 + i * (branch_size - 300) / own_layers) for i in range(own_layers + 1)]
            p = p[::-1]

            # Build own fully-connected layers
            w, b, a, n, z = [], [], [], [], []
            training = tf.placeholder(tf.bool, name='phase')
            for i in range(own_layers):
                w.append(tf.Variable(tf.random_normal([p[i], p[i+1]], stddev=float('1e-5')), name='w_'+str(i)))
                b.append(tf.Variable(tf.random_normal([1, p[i+1]]), name='b_'+str(i)))
                if i == 0:
                    a.append(tf.add(tf.matmul(x, w[i]), b[i], name='a_'+str(i)))
                else:
                    a.append(tf.add(tf.matmul(z[i-1], w[i]), b[i], name='a_'+str(i)))
                if i+1 != own_layers:
                    n.append(tf.layers.batch_normalization(a[i], training=training, name='n_'+str(i)))
                    if activation == 'relu':
                        z.append(tf.nn.relu(n[i], name='relu_' + str(i)))
                    elif activation == 'tanh':
                        z.append(tf.nn.tanh(n[i], name='tanh_' + str(i)))
                    else:
                        print("'" + activation + "' is not a valid activation function. Falling back to 'tanh'.")
                        z.append(tf.nn.tanh(n[i], name='tanh_' + str(i)))

            # Log relevant metrics at output layer
            y_pred = tf.identity(a[-1], name='y_pred')
            with tf.name_scope('L2_distance'):
                l2_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y - y_pred), axis=1)), name='l2_dist')
                tf.summary.scalar(name='l2_distance', tensor=l2_dist)
            with tf.name_scope('Cosine_distance'):
                cos_dist = tf.losses.cosine_distance(tf.nn.l2_normalize(y, dim=1),
                                                     tf.nn.l2_normalize(y_pred, dim=1), dim=1)
                tf.summary.scalar(name='cosine_distance', tensor=cos_dist)
            with tf.name_scope('Mixed_distance'):
                mixed_dist = tf.add(alpha*cos_dist, (1-alpha)*l2_dist, name='mixed_dist')
                tf.summary.scalar(name='mixed_distance', tensor=mixed_dist)
            with tf.name_scope('Misc_summaries'):
                y_pred_norm = tf.reduce_mean(tf.norm(y_pred, axis=1), name='y_pred_norm')
                tf.summary.scalar(name='pred_norm', tensor=y_pred_norm)
                norm_diff = tf.reduce_mean(tf.norm(y_pred, axis=1) - tf.norm(y, axis=1))
                tf.summary.scalar(name='norm_diff', tensor=norm_diff)
                mean_tiled = tf.tile(tf.expand_dims(tf.reduce_mean(y, axis=0), 0), tf.stack([tf.shape(y_pred)[0], 1]))
                mean_diff = tf.losses.cosine_distance(tf.nn.l2_normalize(mean_tiled, dim=1),
                                                     tf.nn.l2_normalize(y_pred, dim=1), dim=1)
                tf.summary.scalar(name='mean_diff', tensor=mean_diff)

        # Define trainable variables
        if layers_to_train == ['all']:
            variables_to_train = tf.trainable_variables()
        else:
            variables_to_train = w + b
            for layer in layers_to_train:
                variables_to_train.extend([v for v in tf.global_variables() if
                                           v.name.startswith('InceptionV3/'+layer) and
                                           (v.name.endswith('weights:0') or
                                            v.name.endswith('biases:0') or
                                            v.name.endswith('beta:0'))])
        
        # Define optimization
        with tf.name_scope('Optimization'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step=global_step,
                                                       decay_steps=lr_decay_freq, decay_rate=lr_decay_factor,
                                                       name='learning_rate')
            adam = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon, name='Adam')
            train_step = adam.minimize(mixed_dist, var_list=variables_to_train, global_step=global_step,
                                       name='train_step')

    # Create and save vectors for set of images
    def create_image_space(self, path='./data/val2014/'):
        print('Creating image space for data in {}'.format(path))

        # Create empty vector space
        new_vec_dict = {}
        # Fill vector space
        filename_list = os.listdir(path)
        for i in range(len(filename_list))[::100]:
            try:
                # Form batch and feed through
                image_batch = [path + filename_list[i + j] for j in range(100)]
                temp_return = net.predict(image_batch)
                for j in range(len(temp_return)):
                    new_vec_dict[filename_list[i + j]] = temp_return[j, :]
                print(i + 1, end=' ', flush=True)
            except KeyboardInterrupt:
                break

        # Save image space
        save_path = './runs/' + self.timestamp + '/image_space_' + self.timestamp + '.npy'
        np.save(save_path, new_vec_dict)
        print('\n\nImage space saved to {}'.format(save_path))

    def evaluate(self, k=5):
        print('Evaluating model on data in {}'.format('./data/sun/'))

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
            return ap_score

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
        group_vecs = np.load('./data/new_scene_names.npy').item()
        # Build matrix of the vectors produced by the model for each image in the
        # evaluation dataset.
        list_directories = os.listdir('./data/sun/')
        if list_directories[0] == '.DS_Store':
            list_directories = list_directories[1:]
        nr_groups = len(list_directories)
        image_matrix = np.zeros([300, 20 * nr_groups])
        dic_keys = {j: i for i, j in enumerate(list(group_vecs.keys()))}

        word_matrix = np.zeros([300, nr_groups])
        word_list = []
        for i, j in enumerate(list(group_vecs.values())):
            word_matrix[:, i] = j[0][1]
            word_list.append(j[0][0])

        for i, explanatory_word in enumerate(list_directories):
            list_images = os.listdir('./data/sun/' + explanatory_word + '/')
            list_images = [
                './data/sun/' + explanatory_word + '/' + image
                for image in list_images if image != '.DS_Store'
                ]
            predictions = self.predict(list_images)
            image_matrix[:, (20 * i):(20 * (i + 1))] = predictions.transpose()

        # Calculate MAP with the 2 first in each category as query.
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
            ap_1 = ap_func(dists, target)
            # Query 2
            target = list(range(20 * i, 20 * (i + 1)))
            del target[1]
            query_vec = image_matrix[:, (20 * i + 1)]
            dists = np.apply_along_axis(
                cosine, 0, image_matrix, query_vec
            )
            ap_2 = ap_func(dists, target)
            image_map += (ap_1 + ap_2)
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
        for i in range(sum_k_rows.shape[0]):
            target_name = list_directories[int((i * k) / 20)]
            target = [dic_keys[target_name]]
            query_vec = sum_k_rows[i, :]
            dists = np.apply_along_axis(
                cosine, 0, word_matrix, query_vec
            )
            ap = ap_func(dists, target)
            word_map += ap
        word_map /= float(sum_k_rows.shape[0])

        # Write results to file
        with open('./runs/' + self.timestamp + '/run_info', 'a+') as run_info:
            print('Mean Average Precision:')
            run_info.write('Mean Average Precision:\n')
            print('IMAGE_RETRIEVAL={}'.format(image_map))
            run_info.write('IMAGE_RETRIEVAL={}\n'.format(image_map))
            print('SIMILARITY_MOTIVATION={}'.format(word_map))
            run_info.write('SIMILARITY_MOTIVATION={}\n\n'.format(word_map))

    # Get vector embeddings after training
    def predict(self, path_list):
        # Get variable handles
        input_im = self.graph.get_tensor_by_name('Preprocessing/convert_image/Cast:0')
        normalized_im = self.graph.get_tensor_by_name('InceptionV3/InceptionV3/Conv2d_1a_3x3/convolution:0')
        phase = self.graph.get_tensor_by_name('Own/phase:0')
        y_pred = self.graph.get_tensor_by_name('Own/y_pred:0')

        # Form batch and feed through
        image_batch = []
        for i in range(len(path_list)):
            temp_im = np.array(Image.open(path_list[i]))
            # If image has only one channel
            if np.ndim(temp_im) == 2:
                temp_im = np.stack([temp_im, temp_im, temp_im], axis=2)
            # Resize image
            resized_temp_im = self.sess.run(normalized_im, feed_dict={input_im: [temp_im], phase: False})
            image_batch.append(resized_temp_im[0])

        # Get prediction
        batch_pred = self.sess.run(y_pred, feed_dict={normalized_im: image_batch, phase: False})
        return batch_pred

    # Restore model from meta file and checkpoint
    def restore_model(self, timestamp):
        self.graph.as_default()
        self.timestamp = timestamp
        checkpoint_dir = './runs/'+timestamp+'/checkpoint/'
        new_saver = tf.train.import_meta_graph(checkpoint_dir+'model.meta')
        new_saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
        tf.constant(timestamp, name='restored_model')
        print('Model restored from {}'.format(checkpoint_dir))

    # Save graph layout separately
    def save_graph_layout(self):
        tf.summary.FileWriter('./runs/' + self.timestamp + '/graph/', graph=self.graph)
        print('Graph layout saved to {}'.format('.runs/' + self.timestamp + '/graph/'))

    # Save model to meta file and checkpoint
    def save_model(self):
        save_path = self.saver.save(self.sess, './runs/' + self.timestamp + '/checkpoint/model')
        print('Trained model saved to {}'.format(save_path))

    # Train model
    def train(self, batch_size=1, epochs=range(0, 15), val_freq=100, val_size=10, norm_cap=False):
        self.graph.as_default()

        # Write hyperparameters to file
        with self.graph.device('/cpu:0'):
            with open('./runs/' + self.timestamp + '/run_info', 'a+') as run_info:
                params = locals()
                print("Train parameters:")
                run_info.write("Train parameters:\n")
                for attr, value in params.items():
                    if attr != 'self' and attr != 'run_info':
                        print("{}={}".format(attr.upper(), value))
                        run_info.write("{}={}\n".format(attr.upper(), value))
                print("")
                run_info.write("\n")

        # Read images
        train_image_path = './data/train2014/'
        train_image_list = os.listdir(train_image_path)
        val_image_path = './data/val2014/'
        val_image_list = os.listdir(val_image_path)
        # Read caption dictionaries
        train_dict = np.load('./data/word2vec_train.npy').item()
        val_dict = np.load('./data/word2vec_val.npy').item()

        # Initialize variables if not continuing earlier training session
        if not epochs[0] > 0:
            slim_var_names = [s[0] for s in tf.contrib.framework.list_variables(
                checkpoint_dir='./data/inception-2016-08-28/inception_v3.ckpt')][:-1]
            slim_saver = tf.train.Saver(var_list=[v for v in tf.global_variables() if v.name[:-2] in slim_var_names])
            slim_saver.restore(self.sess, "./data/inception-2016-08-28/inception_v3.ckpt")
            uninitialized = self.sess.run(tf.report_uninitialized_variables())
            uninitialized = [str(v)[2:-1] + ':0' for v in uninitialized]
            uninitialized = [v for v in tf.global_variables() if v.name in uninitialized]
            self.sess.run(tf.variables_initializer(var_list=uninitialized))

        # Get variable handles
        input_im = self.graph.get_tensor_by_name('Preprocessing/convert_image/Cast:0')
        normalized_im = self.graph.get_tensor_by_name('Preprocessing/normalized_im:0')
        y = self.graph.get_tensor_by_name('Own/y:0')
        train_step = self.graph.get_operation_by_name('Optimization/train_step')
        phase = self.graph.get_tensor_by_name('Own/phase:0')
        all_summaries = tf.summary.merge_all()

        # Initialize writers
        train_writer = tf.summary.FileWriter('./runs/' + self.timestamp + '/sums/train/', flush_secs=20)
        val_writer = tf.summary.FileWriter('./runs/' + self.timestamp + '/sums/val/', flush_secs=20)

        # Initialize saver
        try:
            os.mkdir('./runs/' + self.timestamp + '/checkpoint/')
        except FileExistsError:
            pass
        self.saver = tf.train.Saver(max_to_keep=1)

        # Train loop
        for e in epochs:
            try:
                print('Epoch {}'.format(e + 1))
                if e > 0:
                    # Shuffle training samples for each epoch
                    random.shuffle(train_image_list)
                    # Save checkpoint every epoch
                    save_path = self.saver.save(self.sess, './runs/' + self.timestamp + '/checkpoint/model')
                    print('Trained model saved to {}'.format(save_path))

                # Perform minibatch training
                for i in range(len(train_image_list))[::batch_size]:
                    image_batch = []
                    caption_batch = []
                    # Form batch and feed through
                    for j in range(batch_size):
                        try:
                            temp_im = np.array(Image.open(train_image_path + train_image_list[i + j]))
                            # If image has only one channel
                            if np.ndim(temp_im) == 2:
                                temp_im = np.stack([temp_im, temp_im, temp_im], axis=2)
                            # Resize image
                            resized_temp_im = self.sess.run(normalized_im, feed_dict={input_im: [temp_im]})
                            image_batch.append(resized_temp_im[0])

                            # Choose one of the five captions randomly
                            r = random.randrange(len(train_dict[train_image_list[i + j]]))
                            temp_caption = train_dict[train_image_list[i + j]][r]
                            if norm_cap:
                                caption_batch.append(temp_caption / np.linalg.norm(temp_caption))
                            else:
                                caption_batch.append(temp_caption)
                        except IndexError:
                            pass
                    caption_batch = np.stack(caption_batch, axis=0)
                    image_batch = np.stack(image_batch, axis=0)

                    # Run actual training op and write summary
                    [summaries, _] = self.sess.run([all_summaries, train_step],
                                                   feed_dict={normalized_im: image_batch,
                                                              y: caption_batch,
                                                              phase: False})
                    train_writer.add_summary(summary=summaries, global_step=e * len(train_image_list) + i)

                    # Evaluate every val_freq:th step
                    if i % val_freq == 0:
                        val_image_batch = []
                        val_caption_batch = []
                        for j in range(val_size):
                            temp_im = np.array(Image.open(val_image_path + val_image_list[j]))
                            # If image has only one channel
                            if np.ndim(temp_im) == 2:
                                temp_im = np.stack([temp_im, temp_im, temp_im], axis=2)
                            # Resize image
                            resized_temp_im = self.sess.run(normalized_im, feed_dict={input_im: [temp_im]})
                            val_image_batch.append(resized_temp_im[0])

                            # Use first caption in validation set
                            temp_caption = val_dict[val_image_list[j]][0]
                            if norm_cap:
                                val_caption_batch.append(temp_caption / np.linalg.norm(temp_caption))
                            else:
                                val_caption_batch.append(temp_caption)
                        val_caption_batch = np.stack(val_caption_batch, axis=0)
                        val_image_batch = np.stack(val_image_batch, axis=0)

                        # Write summary
                        summaries = self.sess.run(all_summaries,
                                                  feed_dict={normalized_im: val_image_batch,
                                                             y: val_caption_batch,
                                                             phase: False})
                        val_writer.add_summary(summaries, e * len(train_image_list) + i)
                        print(i, end=' ', flush=True)

                print('')

            # Makes it possible to interrupt training in the middle
            except KeyboardInterrupt:
                print('')
                break
        print('')

if __name__ == "__main__":

    # Define model
    branch_path = 'InceptionV3/InceptionV3/Mixed_7b/concat:0'
    layers_to_train = ['Mixed_7b', 'Mixed_7a', 'Mixed_6e']
    own_layers = 1
    alpha = 0.95

    # Build, train, save, and create image space
    net = SlimInceptionV3()
    net.build_graph(branch_path=branch_path, layers_to_train=layers_to_train, own_layers=own_layers,
                    init_learning_rate=1e-3, lr_decay_freq=1, lr_decay_factor=1,
                    epsilon=0.01, alpha=alpha, activation='tanh')
    net.save_graph_layout()
    net.train(batch_size=32, epochs=range(0, 15), val_freq=1000, val_size=200, norm_cap=False)
    net.save_model()
    net.evaluate()
    net.create_image_space()
    sys.exit(0)
