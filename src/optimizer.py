import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds_attribute, labels_attribute, preds_structure, labels_structure, alpha):

        # attribute reconstruction loss
        diff_attribute = tf.square(preds_attribute - labels_attribute)
        self.attribute_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_attribute, 1))
        # self.reconstruction_errors =  tf.losses.mean_squared_error(labels= labels, predictions=preds)
        self.attribute_cost = tf.reduce_mean(self.attribute_reconstruction_errors)

        # structure reconstruction loss
        diff_structure = tf.square(preds_structure - labels_structure)

        self.structure_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_structure, 1))
        self.structure_cost = tf.reduce_mean(self.structure_reconstruction_errors)


        self.reconstruction_errors = tf.multiply(alpha, self.attribute_reconstruction_errors) + tf.multiply(1-alpha, self.structure_reconstruction_errors)
        self.cost = alpha * self.attribute_cost + (1-alpha) * self.structure_cost

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        # self.grads_vars = self.optimizer.compute_gradients(self.cost)


class OptimizerDAE(object):
    def __init__(self, preds_attribute, labels_attribute, preds_structure, labels_structure,
                 alpha, eta, theta):

        self.preds_attribute = preds_attribute
        self.labels_attribute = labels_attribute

        # attribute reconstruction loss
        B_attr = labels_attribute * (eta - 1) + 1
        diff_attribute = tf.square(tf.subtract(preds_attribute, labels_attribute)*B_attr)
        self.attribute_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_attribute, 1))
        self.attribute_cost = tf.reduce_mean(self.attribute_reconstruction_errors)

        # structure reconstruction loss
        B_struc = labels_structure * (theta - 1) + 1
        diff_structure = tf.square(tf.subtract(preds_structure, labels_structure)*B_struc)
        self.structure_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_structure, 1))
        self.structure_cost = tf.reduce_mean(self.structure_reconstruction_errors)

        self.reconstruction_errors = tf.multiply(alpha, self.attribute_reconstruction_errors) \
                                     + tf.multiply(1-alpha, self.structure_reconstruction_errors)

        self.cost = alpha * self.attribute_cost + (1-alpha) * self.structure_cost

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        # self.grads_vars = self.optimizer.compute_gradients(self.cost)
