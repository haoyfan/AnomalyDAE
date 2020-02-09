import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
from pre_import import *

import tensorflow as tf
import numpy as np
from anomaly_detection import AnomalyDetectionRunner
from utils import *
from tensorboardX import SummaryWriter



flags = tf.app.flags
FLAGS = flags.FLAGS

embed_dim=128
print("### embed_dim=", embed_dim)

flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', embed_dim*2, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', embed_dim, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 1, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 100, 'number of iterations.')
flags.DEFINE_float('alpha', 0.8, 'balance parameter') # for attribute cost
flags.DEFINE_float('eta', 0, 'balance parameter') # for attribute
flags.DEFINE_float('theta', 0, 'balance parameter') # for structure

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

# data_list = ['BlogCatalog', 'Flickr', 'ACM']
data_list = ['BlogCatalog']

# eta_list = np.arange(1, 10, 2).astype(np.int)
# theta_list = np.arange(1, 101, 10).astype(np.int)

# alpha_list = [0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
alpha_list = [0.7]
# embed_dims = [8,16,32,64,128,256,512,1024]

model = 'AnomalyDAE'  # 'Dominant' or 'AnomalyDAE'
task = 'anomaly_detection'


for dataset_str in data_list:
    if dataset_str=='BlogCatalog':
        # eta_list = [1,3,5]
        eta_list = [5]
        theta_list = [40]
        decoder_act = [tf.nn.sigmoid, lambda x: x] # [structure_act, attribute_act]
        FLAGS.iterations=180
    elif dataset_str=='Flickr':
        eta_list = [8]
        theta_list = [90]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        FLAGS.iterations=100
    elif dataset_str=='ACM':
        eta_list = [3] # for attribute
        theta_list = [10] # for structure
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        # decoder_act = [lambda x: x, lambda x: x]
        FLAGS.iterations=80
    else:
        print("[ERROR] no such dataset: {}".format(dataset_str))
        continue

    for eta in eta_list:
        for theta in theta_list:
            for alpha in alpha_list:
                FLAGS.eta=eta
                FLAGS.theta=theta
                FLAGS.alpha=alpha

                settings = {'data_name': dataset_str,
                            'iterations': FLAGS.iterations,
                            'model' : model,
                            'decoder_act': decoder_act}

                results_dir = os.path.sep.join(['results', dataset_str, task, model])
                log_dir = os.path.sep.join(['logs', dataset_str, task, model, '{}_{}_{}'.format(eta, theta, alpha)])

                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                file2print = '{}/{}_{}_{}_{}_{}.json'.format(results_dir, dataset_str,
                                                              eta, theta, alpha, embed_dim)

                runner = None
                if task == 'anomaly_detection':
                    runner = AnomalyDetectionRunner(settings)

                writer = SummaryWriter(log_dir)

                runner.erun(writer)
