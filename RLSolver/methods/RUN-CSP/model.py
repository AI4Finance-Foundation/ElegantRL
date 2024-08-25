import tensorflow as tf
import numpy as np
import json
import os
from tqdm import tqdm

from csp_utils import Constraint_Language, CSP_Instance, max_2sat_language, is_language


class Message_Network:
    """ Message Network that sends messages between variables """

    def __init__(self, out_units, activation='linear'):
        """
        :param out_units: Length of the message vectors. We usually use the variables state size for this.
        :param activation: The activation of each layer.
        """
        self.out_units = out_units
        self.activation = activation
        
        # Output layer for generating both messages
        self.out_layer = tf.keras.layers.Dense(2 * self.out_units,
                                               activation=activation,
                                               use_bias=False,
                                               kernel_regularizer=tf.keras.regularizers.l2())
        self.out_norm = tf.keras.layers.BatchNormalization()

    def __call__(self, in_left, in_right):
        """
        :param in_left: A tensor of shape (m, h) and type float32. m is the number of constraints to which this messaging function is applied.
                        h is the length of the input vectors. in_left[i,:] is the input vector from the left end point of the i-th constraint.
        :param in_right: The same as in_left, but for the other endpoint of each constraint
        :return: Two tensors msg_left and msg_right with shape (m, h) that contain the messages send to each endpoint of each clause.
        """
        # combine inputs for both endpoints
        y = tf.concat([in_left, in_right], axis=1)

        # call output layer and batch normalization
        y = self.out_layer(y)
        y = self.out_norm(y)

        # split output into two different messages
        msg_left = y[:, :self.out_units]
        msg_right = y[:, self.out_units:2 * self.out_units]

        return msg_left, msg_right


class Symmetric_Message_Network():
    """ Symmetric Version of the Messaging Network """

    def __init__(self, out_units, activation='linear'):
        """
        :param out_units: Length of the message vectors. We usually use the variables state size for this.
        :param activation: The activation of each layer.
        """
        self.out_units = out_units
        self.activation = activation

        # Output layer for generating both messages
        self.out_layer = tf.keras.layers.Dense(self.out_units,
                                               activation=activation,
                                               use_bias=False,
                                               kernel_regularizer=tf.keras.regularizers.l2())
        self.out_norm = tf.keras.layers.BatchNormalization()

    def __call__(self, in_right, in_left):
        """
        :param in_left: A tensor of shape (m, h) and type float32. m is the number of constraints to which this messaging function is applied.
                        h is the length of the input vectors. in_left[i,:] is the input vector from the left end point of the i-th constraint.
        :param in_right: The same as in_left, but for the other endpoint of each constraint
        :return: Two tensors msg_left and msg_right with shape (m, h) that contain the messages send to each endpoint of each clause.
        """

        # combine inputs in both directions
        in_lr = tf.concat([in_left, in_right], axis=1)
        in_rl = tf.concat([in_right, in_left], axis=1)

        # stack combined tensors along batch axis
        y = tf.concat([in_lr, in_rl], axis=0)

        # call output layer and batch normalization
        y = self.out_layer(y)
        y = self.out_norm(y)

        # split tensor to obtain messages in both directions
        n_edges = tf.shape(in_right)[0]
        msg_left = y[:n_edges, :]
        msg_right = y[n_edges:, :]

        return msg_left, msg_right


def get_message_function(M):
    """
    Helper function to to return correct message network constructor for a given relation
    :param M: The characteristic matrix of a relation R
    :returns: Symmetric_Message_Network if R is symmetric, otherwise Message_Network
    """
    symmetric = np.allclose(M, M.T, rtol=1e-05, atol=1e-08)
    return Symmetric_Message_Network if symmetric else Message_Network


class RUN_CSP_Cell:
    """ The RNN Cell used by RUN-CSP. Implements the cell of the network as specified for tf.keras.layers.RNN """

    def __init__(self, network):
        """
        :param network: The RUN_CSP instance that the cell belongs to
        """

        self.network = network
        self.output_units = network.domain_size if network.domain_size > 2 else 1
        self.relations = network.language.relation_names
        self.message_networks = network.message_networks
        self.n_variables = network.n_variables
        self.n_clauses = network.n_clauses

        # since there are two state vectors for each variable, this has to be an array to fit the keras specification
        self.state_size = [network.state_size, network.state_size]

        self.clauses = network.clauses
        self.idx_left = network.idx_left
        self.idx_right = network.idx_right
        
        self.degrees = tf.cast(tf.reshape(network.degrees, [self.n_variables, 1]), dtype=tf.float32)

        # Batch normalization layer to normalize recieved messages
        self.normalize = tf.keras.layers.BatchNormalization()

        # LSTM Cell to update variable states
        self.update = tf.keras.layers.LSTMCell(network.state_size,
                                               use_bias=True,
                                               bias_regularizer=tf.keras.regularizers.l2(),
                                               kernel_regularizer=tf.keras.regularizers.l2(),
                                               recurrent_regularizer=tf.keras.regularizers.l2())

        # Trainable linear reduction to map variable states to logits before softmax/sigmoid
        self.out_reduction = tf.keras.layers.Dense(self.output_units,
                                                   activation='linear',
                                                   use_bias=False,
                                                   kernel_regularizer=tf.keras.regularizers.l2())

        # Initializer for long term memory states of the LSTM Cell
        self.long_state_init = tf.initializers.zeros()

        # Random Initializer for variable states
        self.var_state_init = tf.random_normal_initializer()

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # generate initial states for each variable in the given instance
        var_states = self.var_state_init([self.n_variables, self.state_size[0]])
        long_states = self.long_state_init([self.n_variables, self.state_size[1]])
        return var_states, long_states

    def call(self, x, states):
        """
        :param x: The temporal input of the cell, which we do not use. Required by keras.
        :param states: The tuple with the current state tensors
        :return: A tensor with the next output for each variable and a tuple with the next states
        """

        # retrieve states
        var_states = states[0]
        long_states = states[1]

        variable_input_tensors = []
        for r in self.relations:
            # send variable states to incident clauses
            clause_in_left = tf.reshape(tf.gather_nd(var_states, self.idx_left[r]), [-1, self.state_size[0]])
            clause_in_right = tf.reshape(tf.gather_nd(var_states, self.idx_right[r]), [-1, self.state_size[0]])

            # call the message network of the current relation to compute messages
            message_network = self.message_networks[r]
            msg_left, msg_right = message_network(clause_in_left, clause_in_right)
            
            # sum up messages for each node
            variable_in_left = tf.scatter_nd(self.idx_left[r], msg_left, shape=[self.n_variables, self.state_size[0]])
            variable_in_right = tf.scatter_nd(self.idx_right[r], msg_right, shape=[self.n_variables, self.state_size[0]])
            variable_in = variable_in_right + variable_in_left
            variable_input_tensors.append(variable_in)

        # sum up messages across all relations and normalize with inverse degrees
        rec = tf.add_n(variable_input_tensors)
        rec = tf.math.divide_no_nan(rec, self.degrees)
        rec = self.normalize(rec)

        # apply LSTM cell to update states
        _, (var_states, long_states) = self.update(rec, [var_states, long_states])

        # compute soft assignments
        logits = self.out_reduction(var_states)

        return logits, (var_states, long_states)


class RUN_CSP:
    """ A Tensorflow implementation of RUN-CSP """

    def __init__(self, model_dir, language, state_size=128):
        """
        :param model_dir: The directory to store the trained model in
        :param language: A Constraint_Language instance that specifies the underlying constraint language
        :param state_size: The length of the variable state vectors
        """
        # create session
        self.session = tf.Session()
        self.session.as_default()

        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        self.language = language
        self.domain_size = language.domain_size
        self.domain = list(range(self.domain_size))

        # get characteristic relation matrices in numpy format and define corresponding tensorflow matrices
        self.relations_matrices = language.relation_matrices
        self.relation_tensors = {r: tf.constant(M, dtype=tf.float32) for r, M in self.relations_matrices.items()}

        # construct the message network for each relation
        self.message_networks = {r: (get_message_function(M))(state_size) for r, M in self.relations_matrices.items()}

        self.state_size = state_size

        self.learning_rate = 0.001
        self.decay_steps = 2000
        self.decay_rate = 0.1

        # placeholder for the number of iterations t_max
        self.iterations = tf.compat.v1.placeholder(dtype=tf.int32)

        """ 
        Placeholders that store the clauses for each iteration.
        For each relation type r, the clauses of this type are stored as tuples in a tensor of shape (n_r, 2),
        where n_r is the number of clauses of type r. 
        """
        self.clauses = {r: tf.compat.v1.placeholder(dtype=tf.int32) for r in self.language.relation_names}
        
        """ 
        Split the clause tensors into single column matrices that each contain the left and right variables of each constraint, respectively.
        These are needed for the gather and scatter operations in the messaging process.
        """
        self.idx_left = {r: tf.reshape(c[:, 0], [-1, 1]) for r, c in self.clauses.items()}
        self.idx_right = {r: tf.reshape(c[:, 1], [-1, 1]) for r, c in self.clauses.items()}

        # placeholder for the degrees, number of variables and clauses
        self.degrees = tf.compat.v1.placeholder(dtype=tf.int32)
        self.n_variables = tf.compat.v1.placeholder(dtype=tf.int32)
        self.n_clauses = tf.compat.v1.placeholder(dtype=tf.int32)

        # initializer for the dummy input of the network
        self.x_init = tf.zeros_initializer()

        # Construct the Cell of the RNN
        self.cell = RUN_CSP_Cell(self)

        # use keras RNN class for the recurrent neural network
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)

        # build the network
        self.build()

        # init writers for summaries
        self.trainWriter = tf.compat.v1.summary.FileWriter(self.model_dir + '/train', self.session.graph)
        self.testWriter = tf.compat.v1.summary.FileWriter(self.model_dir + '/test', self.session.graph)
        self.summaries = tf.compat.v1.summary.merge_all()

        var = [v for v in tf.compat.v1.local_variables()]
        self.rolling_variable_init = tf.compat.v1.variables_initializer(var)

        if self.has_checkpoint():
            # reload checkpoint if this model has been trained already
            self.load_checkpoint()
        else:
            # initialize tensorflow variables otherwise
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            self.session.run([init_global, init_local])
            self.save_parameters()

    def build(self):
        """ Builds the Networks Computational Graph """

        # create dummy for x to call network for the number of iterations. This value is not actually used in the network.
        x = self.x_init(shape=([self.n_variables, self.iterations, 1]))

        # call rnn to get the color probabilites for each node and iteration
        logits = self.rnn(x)

        if self.domain_size == 2:
            self.p = tf.reshape(tf.nn.sigmoid(logits), [self.n_variables, self.iterations, 1])
            self.phi = tf.concat([1.0 - self.p, self.p], axis=2)
        else:
            self.phi = tf.nn.softmax(logits, axis=2)

        # compute loss for each iteration
        loss = tf.reduce_sum(self.build_loss())
        # add result to collection loss
        tf.compat.v1.losses.add_loss(loss)

        # compute mean loss metric and add it to summaries
        self.mean_loss, self.mean_loss_op = tf.compat.v1.metrics.mean(loss)
        with tf.name_scope('summaries'):
            tf.compat.v1.summary.scalar('loss', self.mean_loss_op)

        # define global step and define learning rate
        self.global_step = tf.Variable(0, trainable=False)
        rate = tf.compat.v1.train.exponential_decay(self.learning_rate,
                                                    self.global_step, decay_steps=self.decay_steps,
                                                    decay_rate=self.decay_rate, staircase=True)

        # use adam optimizer with default parameters  and gradient clipping for training
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=rate)
        #self.train_op = optimizer.minimize(tf.compat.v1.losses.get_total_loss(), self.global_step)
        gvs = optimizer.compute_gradients(tf.compat.v1.losses.get_total_loss())
        gvs = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(gvs, self.global_step)

        # build predictions and additional metrics
        self.build_predictions()

    def build_loss(self):
        """
        Computes the loss for training RUN-CSP
        :return: A 1 dimensional tensor that contains the loss of each iteration, weighted by a discount factor
        """

        # reshape phi to combine all iterations for each variable
        all_phi = tf.reshape(self.phi, [self.n_variables, self.iterations * self.domain_size])

        relation_losses = []
        for r, M in self.relation_tensors.items():
            # order opposing predictions according to clauses
            phi_left = tf.gather_nd(all_phi, self.idx_left[r])
            phi_left = tf.reshape(phi_left, [-1, self.domain_size])
            phi_right = tf.gather_nd(all_phi, self.idx_right[r])
            phi_right = tf.reshape(phi_right, [-1, self.domain_size])

            # compute matrix product for each clause
            clause_relation_loss = tf.reduce_sum(tf.matmul(phi_left, M) * phi_right, axis=1)
            clause_relation_loss = tf.reshape(clause_relation_loss, [-1, self.iterations])

            # compute combined loss of clauses of the current relation
            relation_loss = -tf.math.log(clause_relation_loss)
            relation_loss = tf.reduce_sum(relation_loss, axis=0)
            relation_losses.append(relation_loss)

        # sum up losses across all relations
        loss = tf.add_n(relation_losses)
        loss = loss / tf.cast(self.n_clauses, tf.float32)

        # compute and apply discount factor
        discount = tf.tile(tf.constant([0.95]), [self.iterations])
        exp = tf.cast(tf.range(self.iterations - 1, tf.constant(-1), tf.constant(-1)), dtype=tf.float32)
        factor = tf.pow(discount, exp)
        loss = factor * loss

        return loss

    def build_predictions(self):
        """ Constructs the predictions and additional metrics """

        # compute hard assignment from final iteration
        self.assignment = tf.cast(tf.argmax(self.phi, axis=2), dtype=tf.int32)

        # compute number of conflicting clauses for the assignment
        relation_conflicts = []
        self.edge_conflicts = {}
        assignment = tf.reshape(self.assignment, [self.n_variables, self.iterations, 1])
        for r, M in self.relation_tensors.items():
            # get values of the endpoints of each clause of type r
            val_left = tf.gather_nd(assignment, self.idx_left[r])
            val_right = tf.gather_nd(assignment, self.idx_right[r])
            val_clause = tf.concat([val_left, val_right], axis=2)

            # Count conflicting clauses of type r
            valid = tf.gather_nd(M, val_clause)
            conflicts = 1.0 - valid

            self.edge_conflicts[r] = conflicts
            n_conflicts = tf.reduce_sum(conflicts[:, self.iterations-1])
            relation_conflicts.append(n_conflicts)

        # sum up conflicts across all relations
        self.conflicts = tf.add_n(relation_conflicts)

        # Add metric for relative number of conflicting clauses
        n_clauses = tf.cast(self.n_clauses, tf.float32)
        self.conflict_ratio, self.conflict_ratio_op = tf.compat.v1.metrics.mean(self.conflicts / n_clauses)

        # Add summaries
        with tf.name_scope('summaries'):
            tf.compat.v1.summary.scalar('conflict_ratio', self.conflict_ratio_op)

    def get_feed_dict(self, instance, iterations):
        """ Creates a Tensorflow feed dict for a given csp instance """
        feed_dict = {self.iterations: iterations,
                     self.n_variables: instance.n_variables,
                     self.n_clauses: instance.n_clauses,
                     self.degrees: instance.degrees}

        for r in self.language.relation_names:
            feed_dict[self.clauses[r]] = instance.clauses[r]
            
        return feed_dict

    def train(self, instances, iterations):
        """
        Performs one training epoch.
        :param instances: A list of CSP_Instance objects to perform training on.
        :param iterations: The number of iterations that RUN-CSP performs on each instances.
        :return: A dictionary that contains the mean ratio of conflicting edges across all instances.
        """
        self.session.run(self.rolling_variable_init)

        print('Training...')
        for instance in tqdm(instances):
            feed_dict = self.get_feed_dict(instance, iterations)
            out = [self.train_op, self.conflict_ratio_op, self.summaries, self.global_step]
            res = self.session.run(out, feed_dict=feed_dict)

        self.trainWriter.add_summary(res[2], res[3])

        output = {'conflict_ratio': res[1]}
        return output

    def predict(self, instance, iterations):
        """
        Generates predictions for a given instance.
        :param instance: A CSP_Instance object.
        :param iterations: The number of iterations that RUN-CSP performs on each instances.
        :return: A dictionary that contains the final hard assignment as well as the number of conflicts.
        """
        self.session.run(self.rolling_variable_init)

        feed_dict = self.get_feed_dict(instance, iterations)

        out = [self.assignment, self.conflicts, self.conflict_ratio_op, self.phi, self.edge_conflicts]
        res = self.session.run(out, feed_dict=feed_dict)

        output = {'assignment': res[0],
                  'conflicts': res[1],
                  'conflict_ratio': res[2],
                  'phi': res[3],
                  'edge_conflicts': res[4]}
        return output

    def predict_boosted(self, instance, iterations, attempts):
        """
        Generate predictions with boosted performance by making multiple runs in paralleland using the best results.
        :param instance: A CSP_Instance object.
        :param iterations: The number of iterations that RUN-CSP performs on each instances.
        :param attempts: The number of parallel runs.
        :return: The predictions for the run with the least conflicts
        """
        # duplicate instance and generate predictions in parallel
        combined = CSP_Instance.merge([instance for _ in range(attempts)])
        output_dict = self.predict(combined, iterations=iterations)

        # soft assignments for all iterations
        phi = output_dict['phi']
        phi = np.reshape(phi, (attempts, instance.n_variables, iterations, instance.language.domain_size))

        # compute hard assignments
        assignments = np.argmax(phi, axis=3)

        # compute number of conflicts in each attempts at each iteration
        conf = np.zeros([attempts, iterations], np.int64)
        for r in instance.language.relations:
            # get binary encoding of whether or not each constraint has a conflict
            edge_conf = output_dict['edge_conflicts'][r]
            edge_conf = np.reshape(edge_conf, [attempts, len(instance.clauses[r]), iterations])
            conf += np.int64(np.sum(edge_conf, axis=1))

        # select solution with fewest conflicts as final output
        best = np.unravel_index(np.argmin(conf, axis=None), conf.shape)
        best_assignment = assignments[best[0], :, best[1]]
        best_conflicts = conf[best]
        best_conflict_ratio = best_conflicts / instance.n_clauses
        
        output = {'assignment': best_assignment,
                  'conflicts': best_conflicts,
                  'conflict_ratio': best_conflict_ratio,
                  'all_assignments': assignments,
                  'all_conflicts': conf}
        return output
        
    def save_checkpoint(self, name='best'):
        """
        Save the current graph and summaries in the model directory
        :param name: Name of the checkpoint
        """
        path = os.path.join(self.model_dir, f"model_{name}.ckpt")

        session = self.session
        saver = tf.compat.v1.train.Saver()
        path = saver.save(session, path)
        print("Model saved in file: %s" % path)

    def load_checkpoint(self, name='best'):
        """
        Load a checkpoint from the model directory
        :param name: Name of the checkpoint
        """
        path = os.path.join(self.model_dir, f"model_{name}.ckpt")
        session = self.session
        saver = tf.compat.v1.train.Saver()
        saver.restore(session, path)

    def has_checkpoint(self):
        """ Check if network has some checkpoint stored in the model directory """
        return os.path.exists(os.path.join(self.model_dir, "checkpoint"))

    def save_parameters(self):
        """ Saves the constraint language and state size in the model directory """

        parameters = {'state_size': self.state_size}
        with open(os.path.join(self.model_dir, "parameters.json"), 'w') as f:
            json.dump(parameters, f)

        self.language.save(os.path.join(self.model_dir, 'language.json'))

    @staticmethod
    def load(model_dir):
        """
        Loads a network from its model directory
        :param model_dir: The directory
        :return: The loaded RUN-CSP Network
        """
        with open(os.path.join(model_dir, "parameters.json"), 'r') as f:
            parameters = json.load(f)

        state_size = parameters['state_size']
        language = Constraint_Language.load(os.path.join(model_dir, 'language.json'))

        network = RUN_CSP(model_dir, language, state_size)
        return network


class Coloring_Network(RUN_CSP):
    """ A RUN-CSP instance that performs 3 coloring on graphs """
    def __init__(self, model_dir, colors=3, state_size=128):
        super().__init__(model_dir, Constraint_Language.get_coloring_language(colors), state_size=state_size)


class Max_2SAT_Network(RUN_CSP):
    """ A RUN-CSP instance for the Max2Sat problem """
    def __init__(self, model_dir, state_size=128):
        super().__init__(model_dir, max_2sat_language, state_size=state_size)


class Max_IS_Network(RUN_CSP):
    """ A Modified RUN-CSP instance for the Max Independent Set Problem """
    def __init__(self, model_dir, kappa=1.0, state_size=128):
        self.kappa = kappa
        super().__init__(model_dir, is_language, state_size=state_size)

    def build_loss(self):
        """
        Overload loss function to add additional loss that maximizes the IS size
        """
        # get standard run-csp loss first
        is_loss = super().build_loss()

        discount = tf.tile(tf.constant([0.95]), [self.iterations])
        exp = tf.cast(tf.range(self.iterations - 1, tf.constant(-1), tf.constant(-1)), dtype=tf.float32)
        factor = tf.pow(discount, exp)

        # loss that rewards larger sets
        max_loss = factor * (1.0 - tf.reduce_mean(self.p, axis=0))

        # product to combine losses
        loss = (self.kappa + is_loss) * (1.0 + max_loss)
        return loss

    def build_predictions(self):
        """ Overload prediction function to measure the size of the IS """
        super().build_predictions()

        self.size_IS = tf.count_nonzero(self.assignment[:, self.iterations-1], dtype=tf.float32)
        self.IS_ratio, self.IS_ratio_op = tf.compat.v1.metrics.mean(self.size_IS / tf.cast(self.n_variables, dtype=tf.float32))

        corrected_ratio = (tf.cast(self.size_IS, tf.float32) - self.conflicts) / tf.cast(self.n_variables, tf.float32)
        self.corrected_ratio, self.corrected_ratio_op = tf.compat.v1.metrics.mean(corrected_ratio)

        # Add summaries
        with tf.name_scope('summaries'):
            tf.compat.v1.summary.scalar('is_ratio', self.IS_ratio_op)

    def train(self, batches, iterations):
        """ Add Independent Set size to output """
        self.session.run(self.rolling_variable_init)

        print('Training Network...')
        for batch in tqdm(batches):
            feed_dict = self.get_feed_dict(batch, iterations)
            out = [self.train_op, self.conflict_ratio_op, self.summaries, self.global_step, self.IS_ratio_op, self.corrected_ratio_op]
            res = self.session.run(out, feed_dict=feed_dict)

        self.trainWriter.add_summary(res[2], res[3])

        output = {'conflict_ratio': res[1], 'is_ratio': res[4], 'corrected_ratio': res[5]}
        return output
    
    def predict_boosted_and_corrected(self, instance, iterations, attempts):
        """
        Generate predictions with boosted performance by making multiple runs in parallel and using the best result.
        This method also computes the IS size after a simple post-processing step,
        where one end point of each conflicting edge is removed from the IS to enforce fully valid independent sets.
        :param instance: A CSP_Instance object.
        :param iterations: The number of iterations that RUN-CSP performs on each instances.
        :param attempts: The number of parallel runs.
        :return: The predictions for the run with the least conflicts
        """
        # duplicate instance and generate predictions in parallel
        output_dict = super().predict_boosted(instance, iterations=iterations, attempts=attempts)

        assignments = output_dict['all_assignments']
        is_sizes = np.sum(assignments, axis=1)
        conflicts = output_dict['all_conflicts']

        # correct is sizes by removing conflicts
        corrected_sizes = is_sizes - conflicts
        # ignore early assignments with many conflicts
        corrected_sizes[:, :iterations-10] = 0 * corrected_sizes[:, :iterations-10]

        # choose attempt with best corrected IS size
        best_attempt = np.argmax(corrected_sizes)

        best = np.unravel_index(np.argmax(corrected_sizes, axis=None), corrected_sizes.shape)
        best_assignment = assignments[best[0], :, best[1]]
        best_conflicts = conflicts[best]
        best_conflict_ratio = best_conflicts / instance.n_clauses
        best_size = corrected_sizes[best]
        best_is_ratio = best_size / instance.n_variables

        output = {'assignment': best_assignment,
                  'conflicts': best_conflicts,
                  'conflict_ratio': best_conflict_ratio,
                  'is_ratio': best_is_ratio,
                  'is_size': best_size}
        return output

    @staticmethod
    def load(model_dir):
        return Max_IS_Network(model_dir)