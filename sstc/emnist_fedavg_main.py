"""
Experiments to compare FedAvg+STC and FedAvg+SSTC

This is intended to be a minimal stand-alone experiment script built on top of
core TFF and starting from the simple FedAvg implementation from the TFF repo.
"""

import collections
import functools
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

import simple_fedavg_tf
import simple_fedavg_tff
import os

np.set_printoptions(precision=None, suppress=None)

# Training hyperparameters
flags.DEFINE_integer('total_rounds', 1000, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
flags.DEFINE_integer('train_clients_per_round', 50,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 5,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 16, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 128, 'Minibatch size of test data.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.1, 'Client learning rate.')

flags.DEFINE_float('stc_sparsity', 0.01, 'STC sparsity (0, 1]')
flags.DEFINE_float('sstc_filter_fraction', 0.125, 'Client learning rate (0, 1]')

FLAGS = flags.FLAGS


def get_emnist_dataset():
    """Loads and preprocesses the EMNIST dataset.

  Returns:
    A `(emnist_train, emnist_test)` tuple where `emnist_train` is a
    `tff.simulation.datasets.ClientData` object representing the training data
    and `emnist_test` is a single `tf.data.Dataset` representing the test data
    of all clients.
  """
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        only_digits=False)

    def element_fn(element):
        return collections.OrderedDict(
            x=tf.expand_dims(element['pixels'], -1), y=element['label'])

    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 418 for Federated EMNIST
        return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
            count=FLAGS.client_epochs_per_round)

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(
            FLAGS.test_batch_size, drop_remainder=False)

    emnist_train = emnist_train.preprocess(preprocess_train_dataset)
    emnist_test = preprocess_test_dataset(
        emnist_test.create_tf_dataset_from_all_clients())
    return emnist_train, emnist_test


def create_original_fedavg_cnn_model(only_digits=False):
    """The CNN model used in https://arxiv.org/abs/1602.05629.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
    data_format = 'channels_last'
    input_shape = [28, 28, 1]

    max_pool = functools.partial(
        tf.keras.layers.MaxPooling2D,
        pool_size=(2, 2),
        padding='same',
        data_format=data_format)
    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=5,
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu)

    model = tf.keras.models.Sequential([
        conv2d(filters=32, input_shape=input_shape),
        max_pool(),
        conv2d(filters=64),
        max_pool(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10 if only_digits else 62),
    ])

    return model


def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


def client_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Preparing the directory to save run results
    root_logdir = os.path.abspath(os.path.join("logs", "SSTC"))

    run_id = "C{0}_E{1}_sparsity{2}_filter_fraction{3}".format(str(FLAGS.train_clients_per_round),
                                     str(FLAGS.client_epochs_per_round), str(round(FLAGS.stc_sparsity, 3)),
                                                               str(round(FLAGS.sstc_filter_fraction, 4)))
    run_logdir_test = os.path.join(root_logdir, run_id, "test")
    test_summary_writer = tf.summary.create_file_writer(run_logdir_test)

    # If GPU is provided, TFF will by default use the first GPU like TF. The
    # following lines will configure TFF to use multi-GPUs and distribute client
    # computation on the GPUs. Note that we put server computatoin on CPU to avoid
    # potential out of memory issue when a large number of clients is sampled per
    # round. The client devices below can be an empty list when no GPU could be
    # detected by TF.
    client_devices = tf.config.list_logical_devices('GPU')
    server_device = tf.config.list_logical_devices('CPU')[0]
    tff.backends.native.set_local_execution_context(
        server_tf_device=server_device, client_tf_devices=client_devices)

    # Load the dataset
    train_data, test_data = get_emnist_dataset()

    def tff_model_fn():
        """Constructs a fully initialized model for use in federated averaging."""
        keras_model = create_original_fedavg_cnn_model(only_digits=False)
        # keras_model.summary()
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return simple_fedavg_tf.KerasModelWrapper(keras_model,
                                                  test_data.element_spec, loss)

    iterative_process = simple_fedavg_tff.build_federated_averaging_process(
        tff_model_fn, server_optimizer_fn, client_optimizer_fn, FLAGS.stc_sparsity, FLAGS.sstc_filter_fraction)
    server_state = iterative_process.initialize()

    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    model = tff_model_fn()

    seed = 0
    for round_num in range(FLAGS.total_rounds):
        # For fair comparison among algorithms, we select the same pool of per-round participants using a seed
        np.random.seed(seed)
        seed = seed + 1
        sampled_clients = np.random.choice(
            train_data.client_ids,
            size=FLAGS.train_clients_per_round,
            replace=False)
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client).batch(
                FLAGS.batch_size, drop_remainder=False)
            for client in sampled_clients
        ]

        # Run the next round of the training
        server_state, train_metrics, time_metrics = iterative_process.next(
            server_state, sampled_train_data)

        print(f'Round {round_num} training loss: {train_metrics} mean time: {time_metrics}')
        print(" Local Training Average Time {:.7f}".format(time_metrics))

        # Evaluating the global model on the test set
        if round_num % FLAGS.rounds_per_eval == 0:
            model.from_weights(server_state.model_weights)
            accuracy = simple_fedavg_tf.keras_evaluate(model.keras_model, test_data,
                                                       metric)
            print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')

        # Writing the results on disk
        # To be then visualized or post-processed
        with test_summary_writer.as_default():
            tf.summary.scalar("sparse_categorical_accuracy", accuracy, step=round_num)
            test_summary_writer.flush()


if __name__ == '__main__':
    app.run(main)
