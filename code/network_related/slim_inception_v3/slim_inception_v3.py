import os
import numpy as np
import random
import time
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_arg_scope


class SlimInceptionV3(object):

    def __init__(self):
        self.timestamp = time.strftime('%Y%m%d-%H%M', time.localtime())
        self.graph = tf.get_default_graph()
        self.sess = tf.Session()

    def build_graph(self, branch_path='InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0', layers_to_train=[], own_layers=1,
                    init_learning_rate=1e-5, decay=0.9, momentum=0.0, lr_decay_freq=10000, lr_decay_factor=0.95,
                    epsilon=1.0):
        self.graph.as_default()

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
            try:
                x = tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 2, 2, 1], padding='VALID')
            except:
                pass
            x = tf.squeeze(x, [1, 2])

            # Calculate dimensions between own layers
            p = [int(300 + i * (branch_size - 300) / own_layers) for i in range(own_layers + 1)]
            p = p[::-1]

            # Build own fully-connected layers
            w, b, a, z = [], [], [], []
            for i in range(own_layers):
                w.append(tf.Variable(tf.random_normal([p[i], p[i+1]], stddev=float('1e-5')), name='w_'+str(i)))
                b.append(tf.Variable(tf.random_normal([1, p[i+1]]), name='b_'+str(i)))
                if i == 0:
                    a.append(tf.add(tf.matmul(x, w[i]), b[i], name='a_'+str(i)))
                else:
                    a.append(tf.add(tf.matmul(z[i-1], w[i]), b[i], name='a_'+str(i)))
                if i+1 != own_layers:
                    z.append(tf.nn.relu(a[i], name='relu_'+str(i)))
            y_pred = tf.identity(a[-1], name='y_pred')
            cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), axis=1), name='cost')
            cost_summary = tf.summary.scalar(name='cost_summary', tensor=cost)

        # Define variables to train
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
            rmsprop = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum,
                                                epsilon=epsilon, name='RMSProp')
            train_step = rmsprop.minimize(cost, var_list=variables_to_train, global_step=global_step, name='train_step')
        return tf.global_variables()

    def predict(self, path_list):
        # Get variable handles
        input_im = self.graph.get_tensor_by_name('Preprocessing/convert_image/Cast:0')
        normalized_im = self.graph.get_tensor_by_name('InceptionV3/InceptionV3/Conv2d_1a_3x3/convolution:0')
        y_pred = self.graph.get_tensor_by_name('Own/y_pred:0')

        # Form batch and feed through
        image_batch = []
        for i in range(len(path_list)):
            temp_im = np.array(Image.open(path_list[i]))
            # If image has only one channel
            if len(np.shape(temp_im)) == 2:
                temp_im = np.stack([temp_im, temp_im, temp_im], axis=2)
            # Resize image
            resized_temp_im = self.sess.run(normalized_im, {input_im: [temp_im]})
            image_batch.append(resized_temp_im[0])

        # Get prediction
        batch_pred = self.sess.run(y_pred, {normalized_im: image_batch})
        return batch_pred

    def restore_model(self, meta_graph_path, checkpoint_dir):
        self.graph.as_default()
        new_saver = tf.train.import_meta_graph(meta_graph_path)
        new_saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
        restored_model = tf.constant(checkpoint_dir, name='restored_model')
        print('Model restored from {}'.format(checkpoint_dir))

    def save_graph_layout(self):
        tf.summary.FileWriter('./runs/' + self.timestamp + '/graph/', graph=self.graph)
        print('Graph layout saved to {}'.format('./runs/' + self.timestamp + '/graph/'))

    def save_model(self):
        os.mkdir('./runs/' + self.timestamp + '/checkpoint/')
        new_saver = tf.train.Saver()
        save_path = new_saver.save(self.sess, './runs/' + self.timestamp + '/checkpoint/model')
        print('Trained model saved to {}'.format(save_path))

    def train(self, batch_size=1, epochs=1, val_freq=100, val_size=10):
        self.graph.as_default()

        # Read images
        train_image_path = '../../../data/train2014/'
        train_image_list = os.listdir(train_image_path)
        val_image_path = '../../../data/val2014/'
        val_image_list = os.listdir(val_image_path)
        # Read caption dictionaries
        train_dict = np.load('../../../data/word2vec_train.npy').item()
        val_dict = np.load('../../../data/word2vec_val.npy').item()

        # Initialize variables
        slim_var_names = [s[0] for s in tf.contrib.framework.list_variables(
            checkpoint_dir='../../../data/inception-2016-08-28/inception_v3.ckpt')][:-1]
        saver = tf.train.Saver(var_list=[v for v in tf.global_variables() if v.name[:-2] in slim_var_names])
        saver.restore(self.sess, "../../../data/inception-2016-08-28/inception_v3.ckpt")
        uninitialized = self.sess.run(tf.report_uninitialized_variables())
        uninitialized = [str(v)[2:-1] + ':0' for v in uninitialized]
        uninitialized = [v for v in tf.global_variables() if v.name in uninitialized]
        self.sess.run(tf.variables_initializer(var_list=uninitialized))

        # Get variable handles
        input_im = self.graph.get_tensor_by_name('Preprocessing/convert_image/Cast:0')
        normalized_im = self.graph.get_tensor_by_name('InceptionV3/InceptionV3/Conv2d_1a_3x3/convolution:0')
        y = self.graph.get_tensor_by_name('Own/y:0')
        cost_summary = self.graph.get_tensor_by_name('Own/cost_summary:0')
        train_step = self.graph.get_operation_by_name('Optimization/train_step')

        # Initialize writers
        train_writer = tf.summary.FileWriter('./runs/' + self.timestamp + '/sums/train/', flush_secs=20)
        val_writer = tf.summary.FileWriter('./runs/' + self.timestamp + '/sums/val/', flush_secs=20)

        # Train
        for e in range(epochs):
            try:
                print('Epoch {}'.format(e + 1))
                if e > 0:
                    random.shuffle(train_image_list)

                for i in range(len(train_image_list))[::batch_size]:
                    image_batch = []
                    caption_batch = []
                    # Form batch and feed through
                    for j in range(batch_size):
                        try:
                            temp_im = np.array(Image.open(train_image_path + train_image_list[i + j]))
                            # If image has only one channel
                            if len(np.shape(temp_im)) == 2:
                                temp_im = np.stack([temp_im, temp_im, temp_im], axis=2)
                            # Resize image
                            resized_temp_im = self.sess.run(normalized_im, {input_im: [temp_im]})
                            image_batch.append(resized_temp_im[0])

                            # Choose one of the five captions randomly
                            r = random.randrange(len(train_dict[train_image_list[i + j]]))
                            temp_caption = train_dict[train_image_list[i + j]][r]
                            caption_batch.append(temp_caption / np.linalg.norm(temp_caption))
                        except IndexError:
                            pass

                    caption_batch = np.stack(caption_batch, axis=0)
                    image_batch = np.stack(image_batch, axis=0)
                    [batch_cost, _] = self.sess.run([cost_summary, train_step],
                                                    feed_dict={normalized_im: image_batch, y: caption_batch})
                    train_writer.add_summary(summary=batch_cost, global_step=e * len(train_image_list) + i)

                    # Evaluate every val_freq:th step
                    if i % val_freq == 0:
                        val_image_batch = []
                        val_caption_batch = []
                        for j in range(val_size):
                            temp_im = np.array(Image.open(val_image_path + val_image_list[j]))
                            # If image has only one channel
                            if len(np.shape(temp_im)) == 2:
                                temp_im = np.stack([temp_im, temp_im, temp_im], axis=2)
                            # Resize image
                            resized_temp_im = self.sess.run(normalized_im, {input_im: [temp_im]})
                            val_image_batch.append(resized_temp_im[0])

                            # Use first caption in validation set
                            temp_caption = val_dict[val_image_list[j]][0]
                            val_caption_batch.append(temp_caption / np.linalg.norm(temp_caption))

                        val_caption_batch = np.stack(val_caption_batch, axis=0)
                        val_image_batch = np.stack(val_image_batch, axis=0)
                        val_cost = self.sess.run(cost_summary,
                                                 feed_dict={normalized_im: val_image_batch, y: val_caption_batch})
                        val_writer.add_summary(val_cost, e * len(train_image_list) + i)
                        print(i, end=' ', flush=True)
                print('')
            except KeyboardInterrupt:
                print('')
                break
        print('')
