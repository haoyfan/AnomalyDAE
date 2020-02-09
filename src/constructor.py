import tensorflow as tf
from model import *
from optimizer import *
from preprocessing import construct_feed_dict
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder():
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    return placeholders


def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero):
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str=='AnomalyDAE':
        model = AnomalyDAE(placeholders, num_features, num_nodes, features_nonzero)
    else:
        print("[ERROR] no such model name: {}".format(model_str))

    return model


def get_optimizer(model_str, model, placeholders, num_nodes, alpha, eta, theta):
    print("alpha:", alpha)

    opt=None
    if model_str == 'gcn_ae'or model_str=='gcn_can':
        opt = OptimizerAE(preds_attribute=model.attribute_reconstructions,
                          labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                          preds_structure=model.structure_reconstructions,
                          labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=alpha)
    elif model_str=='AnomalyDAE':
        opt = OptimizerDAE(preds_attribute=model.attribute_reconstructions,
                           labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                           preds_structure=model.structure_reconstructions,
                           labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=alpha,
                           eta=eta, theta=theta)
    else:
        print("[ERROR] no such model name: {}".format(model_str))

    return opt

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    _, train_loss, loss_struc, loss_attr, rec_error = sess.run([opt.opt_op,
                                         opt.cost,
                                         opt.structure_cost,
                                         opt.attribute_cost,
                                         opt.reconstruction_errors], feed_dict=feed_dict)

    return train_loss, loss_struc, loss_attr, rec_error