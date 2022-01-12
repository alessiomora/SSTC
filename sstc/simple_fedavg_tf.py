"""An implementation of the Federated Averaging algorithm +
STC or SSTC of updates.

Based on the papers:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629

Robust and communication-efficient federated learning from non-iid data
    Felix Sattler, Simon Wiedemann, Klaus-Robert M¨uller, and Wojciech Samek
    IEEE transactions on neural networks and learning systems, 2019

Structured Sparse Ternary Compression for Convolutional Layers in Cross-Device Federated Learning
    Alessio Mora, Luca Foschini, Paolo Bellavista
    Under review.
"""

import collections
import attr
import tensorflow as tf
import tensorflow_federated as tff

ModelWeights = collections.namedtuple('ModelWeights', 'trainable non_trainable')
ModelOutputs = collections.namedtuple('ModelOutputs', 'loss')


class KerasModelWrapper(object):
    """A standalone keras wrapper to be used in TFF."""

    def __init__(self, keras_model, input_spec, loss):
        """A wrapper class that provides necessary API handles for TFF.

    Args:
      keras_model: A `tf.keras.Model` to be trained.
      input_spec: Metadata of dataset that desribes the input tensors, which
        will be converted to `tff.Type` specifying the expected type of input
        and output of the model.
      loss: A `tf.keras.losses.Loss` instance to be used for training.
    """
        self.keras_model = keras_model
        self.input_spec = input_spec
        self.loss = loss

    def forward_pass(self, batch_input, training=True):
        """Forward pass of the model to get loss for a batch of data.

    Args:
      batch_input: A `collections.abc.Mapping` with two keys, `x` for inputs and
        `y` for labels.
      training: Boolean scalar indicating training or inference mode.

    Returns:
      A scalar tf.float32 `tf.Tensor` loss for current batch input.
    """
        preds = self.keras_model(batch_input['x'], training=training)
        loss = self.loss(batch_input['y'], preds)
        return ModelOutputs(loss=loss)

    @property
    def weights(self):
        return ModelWeights(
            trainable=self.keras_model.trainable_variables,
            non_trainable=self.keras_model.non_trainable_variables)

    def from_weights(self, model_weights):
        tf.nest.map_structure(lambda v, t: v.assign(t),
                              self.keras_model.trainable_variables,
                              list(model_weights.trainable))
        tf.nest.map_structure(lambda v, t: v.assign(t),
                              self.keras_model.non_trainable_variables,
                              list(model_weights.non_trainable))


def keras_evaluate(model, test_data, metric):
    metric.reset_states()
    for batch in test_data:
        preds = model(batch['x'], training=False)
        metric.update_state(y_true=batch['y'], y_pred=preds)
    return metric.result()


@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput(object):
    """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `client_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  """
    weights_delta = attr.ib()
    client_weight = attr.ib()
    model_output = attr.ib()
    time_client_update = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
    """Structure for state on the server.

  Fields:
  -   `model_weights`: A dictionary of model's trainable variables.
  -   `optimizer_state`: Variables of optimizer.
  -   'round_num': Current round index
  """
    model_weights = attr.ib()
    optimizer_state = attr.ib()
    round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
    """Structure for tensors broadcasted by server during federated optimization.

  Fields:
  -   `model_weights`: A dictionary of model's trainable tensors.
  -   `round_num`: Round index to broadcast. We use `round_num` as an example to
          show how to broadcast auxiliary information that can be helpful on
          clients. It is not explicitly used, but can be applied to enable
          learning rate scheduling.
  """
    model_weights = attr.ib()
    round_num = attr.ib()


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta):
    """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `KerasModelWrapper` or `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`. If the optimizer
      creates variables, they must have already been created.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: A nested structure of tensors holding the updates to the
      trainable variables of the model.

  Returns:
    An updated `ServerState`.
  """
    # Initialize the model with the current state.
    model_weights = model.weights
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          server_state.model_weights)
    tf.nest.map_structure(lambda v, t: v.assign(t), server_optimizer.variables(),
                          server_state.optimizer_state)

    # Apply the update to the model.
    neg_weights_delta = [-1.0 * x for x in weights_delta]
    server_optimizer.apply_gradients(
        zip(neg_weights_delta, model_weights.trainable), name='server_update')

    # Create a new state based on the updated model.
    return tff.structure.update_struct(
        server_state,
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=server_state.round_num + 1)


@tf.function
def build_server_broadcast_message(server_state):
    """Builds `BroadcastMessage` for broadcasting.

  This method can be used to post-process `ServerState` before broadcasting.
  For example, perform model compression on `ServerState` to obtain a compressed
  state that is sent in a `BroadcastMessage`.

  Args:
    server_state: A `ServerState`.

  Returns:
    A `BroadcastMessage`.
  """
    return BroadcastMessage(
        model_weights=server_state.model_weights,
        round_num=server_state.round_num)


@tf.function
def client_update(model, dataset, server_message, client_optimizer):
    """Performans client local training of `model` on `dataset`.

  Args:
    model: A `tff.learning.Model`.
    dataset: A 'tf.data.Dataset'.
    server_message: A `BroadcastMessage` from server.
    client_optimizer: A `tf.keras.optimizers.Optimizer`.

  Returns:
    A 'ClientOutput`.
  """

    def difference_model_norm_2_square(global_model, local_model):
        """
        Calculates the squared l2 norm of a model difference (i.e.
        local_model - global_model)
        Args:
            global_model: the model broadcast by the server
            local_model: the current, in-training model

        Returns: the squared norm
        """
        difference = tf.nest.map_structure(lambda a, b: a - b,
                                           local_model,
                                           global_model)
        squared_norm = tf.square(tf.linalg.global_norm(difference))
        return squared_norm

    init_time = tf.timestamp()
    model_weights = model.weights
    initial_weights = server_message.model_weights
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          initial_weights)

    num_examples = tf.constant(0, dtype=tf.int32)
    loss_sum = tf.constant(0, dtype=tf.float32)
    # Explicit use `iter` for dataset is a trick that makes TFF more robust in
    # GPU simulation and slightly more performant in the unconventional usage
    # of large number of small datasets.

    for batch in iter(dataset):
        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch)

            # ------ FedProx ------
            # mu = tf.constant(0.3, dtype=tf.float32)
            # prox_term = (mu/2)*difference_model_norm_2_square(model_weights.trainable, initial_weights.trainable)
            # fedprox_loss = outputs.loss + prox_term
            #----------------------

        grads = tape.gradient(outputs.loss, model_weights.trainable)
        # If FedProx is decommented, letting GradientTape dealing with the FedProx's loss
        #grads = tape.gradient(fedprox_loss, model_weights.trainable)

        client_optimizer.apply_gradients(zip(grads, model_weights.trainable))

        batch_size = tf.shape(batch['x'])[0]
        num_examples += batch_size
        #tf.print("BATCH SIZE ", batch_size)
        loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)

    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)
    client_weight = tf.cast(num_examples, tf.float32)

    weight_delta_copy = []
    for ww in weights_delta:
        weight_delta_copy.append(tf.identity(ww))

    # Sparse Ternary Compression (STC) - excluding biases
    # De-comment this for STC
    #weights_delta_no_bias_flatten = flatten_ad_hoc(weights_delta)
    #stc_updates_no_bias, idx = sparse_ternary_compression(weights_delta_no_bias_flatten, percentage=0.01)
    #weights_delta[0], weights_delta[2], weights_delta[4], weights_delta[6] = reshape_ad_hoc(stc_updates_no_bias, weights_delta)

    # Structured STC - excluding biases
    conv_updates = [weight_delta_copy[0], weight_delta_copy[2]] 
    fc_updates = [weight_delta_copy[4], weight_delta_copy[6]]
    weights_delta[0], weights_delta[2], weights_delta[4], weights_delta[6] = structured_sparse_ternary_compression(conv_updates, fc_updates, 260, 0.01)

    # Just a time metric
    time_client_update = tf.subtract(tf.timestamp(), init_time)

    return ClientOutput(weights_delta, client_weight, loss_sum / client_weight, time_client_update)

"""
for batch in iter(dataset):
    with tf.GradientTape() as tape:
      outputs = model.forward_pass(batch)
    grads = tape.gradient(outputs.loss, model_weights.trainable)
    client_optimizer.apply_gradients(zip(grads, model_weights.trainable))
    batch_size = tf.shape(batch['x'])[0]
    num_examples += batch_size
    loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)
"""

# assign_sub(var, lr_t * (grad + mu_t * (var - vstar)))
# tf.print("LEARNING RATE ", client_optimizer.learning_rate)
# l_r = client_optimizer.learning_rate
# proximal_term = tf.multiply(l_r, tf.multiply(
#    mu, tf.subtract(model_weights.trainable, initial_weights.trainable)))

# Check if l_r is already applied by optimizer
# prox_term = tf.nest.map_structure(lambda a, b: (-1)*mu*(a - b),
#                      model_weights.trainable,
#                      initial_weights.trainable)
# tf.print("PROX TERM ", prox_term)

# client_optimizer.apply_gradients(zip(prox_term,
#                                     model_weights.trainable))


def sparse_ternary_compression(weights_delta, percentage):
    dW = tf.reshape(weights_delta, [-1])

    dW_size = tf.size(dW)
    k_float = tf.multiply(percentage, tf.cast(dW_size, tf.float32))
    k_int = tf.cast(tf.math.round(k_float), dtype=tf.int32)

    dW_abs = tf.math.abs(dW)
    values_for_calculating_mu, idx = tf.math.top_k(dW_abs, k=k_int)

    k_float = tf.cast(k_int, dtype=tf.float32)
    mu = tf.math.divide_no_nan(tf.reduce_sum(values_for_calculating_mu), k_float)
    mu = tf.reshape(mu, [])
    mu_neg = tf.math.negative(mu)

    # TODO: here add encoding-decoding to 8 bits for mu and mu_neg
    # excluding indices from the compression

    indices = tf.expand_dims(idx, 1)
    values = tf.gather_nd(dW, indices)

    sparse_dW = tf.scatter_nd(indices, values, tf.shape(dW))

    zero = tf.constant(0, dtype=tf.float32)
    where_positive = tf.math.greater(sparse_dW, zero)
    where_negative = tf.math.less(sparse_dW, zero)

    indices_of_pos = tf.where(where_positive)
    indices_of_neg = tf.where(where_negative)

    mu_pos_tensor = tf.fill([tf.size(indices_of_pos)], mu)
    mu_neg_tensor = tf.fill([tf.size(indices_of_neg)], mu_neg)

    decoded_values = tf.tensor_scatter_nd_add(tf.zeros(dW_size, dtype=tf.float32), indices_of_neg,
                                              mu_neg_tensor)
    decoded_values = tf.tensor_scatter_nd_add(decoded_values, indices_of_pos, mu_pos_tensor)

    return decoded_values, idx


def flatten_ad_hoc(weights_delta):
    weights_delta_no_bias_flatten = tf.concat(
            [tf.reshape(weights_delta[0], [-1]),
             tf.reshape(weights_delta[2], [-1]),
             tf.reshape(weights_delta[4], [-1]),
             tf.reshape(weights_delta[6], [-1])], axis=0)
    #tf.print("Dimension FLATTEN")
    #tf.print(tf.size(weights_delta_no_bias_flatten))
    return weights_delta_no_bias_flatten

def reshape_ad_hoc(weights_delta_no_bias_flatten, weights_delta):

    conv1 = tf.reshape(tf.slice(weights_delta_no_bias_flatten, begin=[0],
                                               size=[tf.size(weights_delta[0])]),
                                      tf.shape(weights_delta[0]))
    conv2 = tf.reshape(tf.slice(weights_delta_no_bias_flatten, begin=[tf.size(weights_delta[0])],
                                               size=[tf.size(weights_delta[2])]),
                                      tf.shape(weights_delta[2]))
    fc1 = tf.reshape(
            tf.slice(weights_delta_no_bias_flatten, begin=[tf.size(weights_delta[2]) + tf.size(weights_delta[0])],
                     size=[tf.size(weights_delta[4])]),
            tf.shape(weights_delta[4]))
    fc2 = tf.reshape(tf.slice(weights_delta_no_bias_flatten, begin=[
            tf.size(weights_delta[4]) + tf.size(weights_delta[2]) + tf.size(weights_delta[0])],
                                               size=[tf.size(weights_delta[6])]),
                                      tf.shape(weights_delta[6]))
    return conv1, conv2, fc1, fc2

def sample_without_replacement(max_val, dim, seed):
    """
    Sampling without replacement with tf operations. Extract values in the range (0, max]
    Inputs:
    max_val, the maximum value of the extracted values
    K, tf.constant that represents the dimension of the resulting tensor
    seed, seed for reproducible results
    Returns a int32 tf.tensor
    """

    logits = tf.zeros([max_val])
    tf.random.set_seed(5)
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1, seed=seed)))
    #z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1, seed=int(tf.reduce_sum(seed)))))
    _, indices = tf.nn.top_k(logits + z, dim)
    return indices


def my_numpy_func(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  print(x)
  #return np.random.choice(5, x, replace=False)[0]
  max_val = tf.constant(10)
  dim = tf.constant(8)
  return sample_without_replacement(max_val, dim, x)

@tf.function(input_signature=[tf.TensorSpec(None, tf.int64)])
def tf_function(input):
  res = tf.numpy_function(my_numpy_func, [input], tf.int32)
  return res
"""
def structured_sparse_ternary_compression(t, n_filters):
    reshaped_t = tf.reshape(t, (25, 32))
    means = tf.math.reduce_mean(tf.math.abs(reshaped_t), axis=0)
    print("means ", means)
    values, idx = tf.math.top_k(means, k=n_filters)
    #print("idx ", idx)
    #print("values ", values)
    #print("idx reshaped", idx)
    #print(tf.shape(t))
    extracted_filters = tf.gather(reshaped_t, idx, axis=1)
    #extracted_filters_list = [tf.slice(t, [0, f], [25, 1])
    #                            for f in idx]

    #extracted_filters = tf.concat(extracted_filters_list, axis=1)

    #print(tf.shape(extracted_filters))
    #print("extr filter ", extracted_filters)

    tensor_original = tf.zeros(tf.shape(reshaped_t), dtype=tf.float32)
    indices = tf.expand_dims(idx, axis=1)

    #print(tf.shape(extracted_filters))
    topk_extracted_filters_val, topk_extracted_filters_idx = tf.math.top_k(tf.math.abs(tf.reshape(extracted_filters, [-1])), k=10)
    topk_extracted_filters_idx_exp = tf.expand_dims(topk_extracted_filters_idx, axis=1)

    topk_extracted_filters_val_with_sign = tf.gather(tf.reshape(extracted_filters, [-1]), topk_extracted_filters_idx)

    extracted_filters_sstc = tf.scatter_nd(topk_extracted_filters_idx_exp, topk_extracted_filters_val_with_sign, tf.shape(tf.reshape(extracted_filters, [-1])))
    extracted_filters_sstc_reshaped = tf.reshape(extracted_filters_sstc, tf.shape(extracted_filters))

    compressed_t = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(tensor_original), indices, tf.transpose(extracted_filters_sstc_reshaped)))

    topk_mean_abs = tf.math.reduce_mean(topk_extracted_filters_val)
    #topk_mean_abs = tf.math.reduce_mean(tf.math.abs(extracted_filters))
    #print("Mean ", topk_mean_abs)

    where_positive = tf.math.greater(compressed_t, 0)
    indices_of_pos = tf.where(where_positive)

    where_negative = tf.math.less(compressed_t, 0)
    indices_of_neg = tf.where(where_negative)

    #tf.print("idx of pos ", tf.reduce_prod(tf.shape(indices_of_pos)))
    #tf.print("idx of neg ", indices_of_neg)
    #print("idx of pos ", indices_of_pos)

    are_positive_present = tf.reduce_prod(tf.shape(indices_of_pos))
    compressed_t_ternary = tf.cond(tf.equal(are_positive_present, 0),
                                 lambda: compressed_t,
                                 lambda: tf.tensor_scatter_nd_update(compressed_t, indices_of_pos, tf.fill(tf.slice(tf.shape(indices_of_pos), [0], [1]), topk_mean_abs)))
    #compressed_t_ternary = tf.tensor_scatter_nd_update(compressed_t, indices_of_pos, tf.fill([tf.reduce_max(tf.shape(indices_of_pos))], topk_mean_abs))
    are_negative_present = tf.reduce_prod(tf.shape(indices_of_neg))
    compressed_t_ternary = tf.cond(tf.equal(are_negative_present, 0),
                                 lambda: compressed_t,
                                 lambda: tf.tensor_scatter_nd_update(compressed_t_ternary, indices_of_neg, tf.fill(tf.slice(tf.shape(indices_of_neg), [0], [1]), tf.math.negative(topk_mean_abs))))
    #compressed_t_ternary = tf.tensor_scatter_nd_update(compressed_t_ternary, indices_of_neg, tf.fill([tf.reduce_max(tf.shape(indices_of_neg))], tf.math.negative(topk_mean_abs)))

    return tf.reshape(compressed_t_ternary, tf.shape(t))
"""
"""
def structured_sparse_ternary_compression(conv_updates, fc_updates, n_filters, spars_rate):

  flatten_conv_updates = tf.concat([tf.reshape(w, [25, -1]) for w in conv_updates], axis=1)
  #print("flatten_w ", flatten_w)
  #shape_to_reproduce = [tf.shape(w) for w in weights_delta]
  #means = tf.concat([tf.math.reduce_mean(tf.math.abs(w), axis=0) for w in flatten_conv_updates], axis=0)
  means = tf.math.reduce_mean(tf.math.abs(flatten_conv_updates), axis=0)
  #print("means ", means)
  #n_filters = 10
  values, idx = tf.math.top_k(means, k=n_filters)
  #idx_gather = tf.expand_dims(idx, axis=1)
  extracted_filters = tf.gather(flatten_conv_updates, idx, axis=1)
  #   tf.print("Post gather")
  #tensors_original_list = [tf.zeros(tf.shape(s), dtype=tf.float32) for s in shape_to_reproduce]
  #tensor_original = tf.zeros(tf.shape(flatten_conv_updates), dtype=tf.float32)
  indices = tf.expand_dims(idx, axis=1)
  
  #print(tf.shape(extracted_filters))
  # HERE CHANGE K
  # reshape conv1 e conv2.
  # concat them with the other fc layers.
  # proceed with ternary comrpression.
  extracted_filters_flatten = tf.reshape(extracted_filters, [-1])
  flatten_updates_fc = tf.concat([tf.reshape(w, [-1]) for w in fc_updates], axis=0)
  flatten_updates = tf.concat([extracted_filters_flatten, flatten_updates_fc], axis=0)
  # Here change
  #k = 10
  spars_rate_tensor = tf.constant(spars_rate, tf.float32)
  k = tf.cast(tf.math.round(spars_rate_tensor*tf.cast(tf.size(flatten_updates), tf.float32) + spars_rate_tensor*tf.cast(tf.size(flatten_conv_updates), tf.float32)), tf.int32)
  #print("K ",k)
  topk_val, topk_idx = tf.math.top_k(tf.math.abs(flatten_updates), k=k)
  topk_idx_exp = tf.expand_dims(topk_idx, axis=1)

  topk_val_with_sign = tf.gather(flatten_updates, topk_idx)
  ternary_val = tf.math.reduce_mean(topk_val)
  #print("Ternary val ", ternary_val)
  ternary_updates = tf.math.multiply(tf.sign(topk_val_with_sign), tf.fill(tf.shape(topk_val_with_sign), ternary_val))

  updates_sstc = tf.scatter_nd(topk_idx_exp, ternary_updates, tf.shape(flatten_updates))
  
  # updates_sstc_reshaped = tf.reshape(extracted_filters_sstc, tf.shape(extracted_filters))
  flatten_conv_upd_ternary = tf.slice(updates_sstc, [0], [tf.size(extracted_filters)])
  # TO CHANGE SHAPE
  #tf.print("Indices ", tf.shape(indices))
  #tf.print("Updates", tf.shape(tf.reshape(flatten_conv_upd_ternary, tf.shape(extracted_filters))))
  conv_upd_ternary = tf.transpose(tf.scatter_nd(indices, tf.transpose(tf.reshape(flatten_conv_upd_ternary, tf.shape(extracted_filters))), tf.shape(tf.zeros((64*32+32, 25)))))
  conv1 = tf.reshape(tf.slice(conv_upd_ternary, [0, 0], [25, 32]), tf.shape(conv_updates[0]))
  conv2 = tf.reshape(tf.slice(conv_upd_ternary, [0, 32], [25, 32*64]), tf.shape(conv_updates[1]))
  fc1 = tf.reshape(tf.slice(updates_sstc, [tf.size(extracted_filters)], [3136*512]), tf.shape(fc_updates[0]))
  fc2 = tf.reshape(tf.slice(updates_sstc, [tf.size(extracted_filters)+ 3136*512], [512*62]), tf.shape(fc_updates[1]))
  #tf.print("fc1 size: ", tf.size(fc1), "fc2 size: ", tf.size(fc2), "versus ", 3136*512, " ", 512*62)
  
  return conv1, conv2, fc1, fc2
"""
def max_means_filters_by_sampling(tt, p, n_filters):
    max_idx = 25
    pp = tf.cast(tf.math.round(tf.math.multiply(tf.cast(p, tf.float32), tf.cast(max_idx, tf.float32))), tf.int32)
    #print(pp)
    def sample_without_replacement(max_val, dim, seed):
      """
      Sampling without replacement with tf operations. Extract values in the range (0, max]
      Inputs:
      max_val, the maximum value of the extracted values
      K, tf.constant that represents the dimension of the resulting tensor
      seed, seed for reproducible results
      Returns a int32 tf.tensor
      """
      logits = tf.zeros([max_val])
      #tf.random.set_seed(5)
      z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
      _, indices = tf.nn.top_k(logits + z, dim)
      return indices

    extracted = sample_without_replacement(max_idx, pp, None)
    res = tf.gather(tt, extracted, axis=0)
    # res rows are non ordered; for the scope of calculating the mean this does not matter
    means = tf.math.reduce_mean(tf.math.abs(res), axis=0)
    _, idx = tf.math.top_k(means, k=n_filters)

    return idx

def structured_sparse_ternary_compression(conv_updates, fc_updates, n_filters, spars_rate):
  #tf.print("Structured")
  flatten_conv_updates = tf.concat([tf.reshape(w, [25, -1]) for w in conv_updates], axis=1)
  sampling = 1.0
  idx = max_means_filters_by_sampling(flatten_conv_updates, sampling, n_filters)

  extracted_filters = tf.gather(flatten_conv_updates, idx, axis=1)

  indices = tf.expand_dims(idx, axis=1)

  # reshape conv1 e conv2
  # concat them with the other fc layers
  # proceed with ternary comrpression
  extracted_filters_flatten = tf.reshape(extracted_filters, [-1])
  flatten_updates_fc = tf.concat([tf.reshape(w, [-1]) for w in fc_updates], axis=0)
  flatten_updates = tf.concat([extracted_filters_flatten, flatten_updates_fc], axis=0)

  spars_rate_tensor = tf.constant(spars_rate, tf.float32)
  k = tf.cast(tf.math.round(spars_rate_tensor*tf.cast(tf.size(flatten_updates_fc), tf.float32) + spars_rate_tensor*tf.cast(tf.size(flatten_conv_updates), tf.float32)), tf.int32)

  topk_val, topk_idx = tf.math.top_k(tf.math.abs(flatten_updates), k=k)
  topk_idx_exp = tf.expand_dims(topk_idx, axis=1)

  topk_val_with_sign = tf.gather(flatten_updates, topk_idx)
  ternary_val = tf.math.reduce_mean(topk_val)

  ternary_updates = tf.math.multiply(tf.sign(topk_val_with_sign), tf.fill(tf.shape(topk_val_with_sign), ternary_val))

  updates_sstc = tf.scatter_nd(topk_idx_exp, ternary_updates, tf.shape(flatten_updates))
  

  flatten_conv_upd_ternary = tf.slice(updates_sstc, [0], [tf.size(extracted_filters)])

  conv_upd_ternary = tf.transpose(tf.scatter_nd(indices, tf.transpose(tf.reshape(flatten_conv_upd_ternary, tf.shape(extracted_filters))), tf.shape(tf.zeros((64*32+32, 25)))))
  conv1 = tf.reshape(tf.slice(conv_upd_ternary, [0, 0], [25, 32]), tf.shape(conv_updates[0]))
  conv2 = tf.reshape(tf.slice(conv_upd_ternary, [0, 32], [25, 32*64]), tf.shape(conv_updates[1]))
  fc1 = tf.reshape(tf.slice(updates_sstc, [tf.size(extracted_filters)], [3136*512]), tf.shape(fc_updates[0]))
  fc2 = tf.reshape(tf.slice(updates_sstc, [tf.size(extracted_filters) + 3136*512], [512*62]), tf.shape(fc_updates[1]))
  
  return conv1, conv2, fc1, fc2

def count_idx_less_than(t, less_than, starting_idx):
    sliced_t = tf.slice(t, [starting_idx], [less_than])
    where_less = tf.math.less(sliced_t, less_than)
    indices = tf.where(where_less)
    #print(indices)
    return tf.cast(tf.size(indices), tf.float32)


def change_stc_in_structured_stc(update_tensor, idx):
    n_filters = tf.cast(tf.round(tf.math.divide(count_idx_less_than(idx, 800, 0), 25.0)), tf.int32)
    #tf.print("Number of elements in conv1 ", idx_less_than_conv1)
    # #tf.print("Number of filters corresponding to structured stc: ", n_filters)
    compressed_conv1 = structured_sparse_ternary_compression(update_tensor, n_filters)
    #structured_sparse_ternary_compression(update_tensor, n_filters)
    return compressed_conv1

def count_no_zero(t):
  #sliced_t = tf.slice(t, [starting_idx], [less_than])
  where_diverse = tf.math.not_equal(t, tf.constant(0, tf.float32))
  indices = tf.where(where_diverse)
  tf.print("Not0,", tf.strings.as_string(tf.size(indices)), output_stream="file://update_count_in_conv1_2.out",summarize=-1)
  return tf.size(indices)
