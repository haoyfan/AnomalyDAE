from __future__ import division
from __future__ import print_function
import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

from constructor import get_placeholder, update
from input_data import format_data
from sklearn.metrics import roc_auc_score
from model import *
from optimizer import *

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

def precision_AT_K(actual, predicted, k, num_anomaly):
    act_set = np.array(actual[:k])
    pred_set = np.array(predicted[:k])
    ll = act_set & pred_set
    tt = np.where(ll == 1)[0]
    prec = len(tt) / float(k)
    rec = len(tt) / float(num_anomaly)
    return round(prec, 4), round(rec, 4)

class AnomalyDetectionRunner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']
        self.decoder_act = settings['decoder_act']

    def erun(self, writer):
        model_str = self.model
        # load data
        feas = format_data(self.data_name)

        print("feature number: {}".format(feas['num_features']))
        # Define placeholders
        placeholders = get_placeholder()

        num_features = feas['num_features']
        features_nonzero = feas['features_nonzero']
        num_nodes = feas['num_nodes']

        if model_str == 'Dominant':
            model = GCNModelAE(placeholders, num_features, features_nonzero)
            opt = OptimizerAE(preds_attribute=model.attribute_reconstructions,
                              labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                              preds_structure=model.structure_reconstructions,
                              labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha)

        elif model_str == 'AnomalyDAE':
            model = AnomalyDAE(placeholders, num_features, num_nodes, features_nonzero, self.decoder_act)
            opt = OptimizerDAE(preds_attribute=model.attribute_reconstructions,
                               labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                               preds_structure=model.structure_reconstructions,
                               labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha,
                               eta=FLAGS.eta, theta=FLAGS.theta)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()

        # Train model
        for epoch in range(1, self.iteration+1):

            train_loss, loss_struc, loss_attr, rec_error = update(model, opt, sess,
                                                                feas['adj_norm'],
                                                                feas['adj_label'],
                                                                feas['features'],
                                                                placeholders, feas['adj'])

            if epoch % 1 == 0:
                y_true = [label[0] for label in feas['labels']]

                auc=0
                try:
                    scores = np.array(rec_error)
                    scores = (scores - np.min(scores)) / (
                            np.max(scores) - np.min(scores))

                    auc = roc_auc_score(y_true, scores)

                except Exception:
                    print("[ERROR] for auc calculation!!!")

                print("Epoch:", '%04d' % (epoch),
                      "AUC={:.5f}".format(round(auc,4)),
                      "train_loss={:.5f}".format(train_loss),
                      "loss_struc={:.5f}".format(loss_struc),
                      "loss_attr={:.5f}".format(loss_attr))

                writer.add_scalar('loss_total', train_loss, epoch)
                writer.add_scalar('loss_struc', loss_struc, epoch)
                writer.add_scalar('loss_attr', loss_attr, epoch)
                writer.add_scalar('auc', auc, epoch)




