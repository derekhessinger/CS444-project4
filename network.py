'''network.py
Deep neural network core functionality implemented with the low-level TensorFlow API.
Alex and Derek
CS444: Deep Learning
'''
import time
import numpy as np
import tensorflow as tf

from tf_util import arange_index

class DeepNetwork:
    '''The DeepNetwork class is the parent class for specific networks (e.g. VGG).
    '''
    def __init__(self, input_feats_shape, reg=0):
        '''DeepNetwork constructor.

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        reg: float.
            The regularization strength.

        TODO: Set instance variables for the parameters passed into the constructor.
        '''
        # Keep these instance vars:
        self.optimizer_name = None
        self.loss_name = None
        self.output_layer = None
        self.all_net_params = []
        self.opt = None

        if isinstance(input_feats_shape, int):
            self.input_feats_shape = (input_feats_shape,)
        else:
            self.input_feats_shape = input_feats_shape
        self.reg = reg

    def compile(self, loss='cross_entropy', optimizer='adam', lr=1e-3, beta_1=0.9, print_summary=True):
        '''Compiles the neural network to prepare for training.

        This involves performing the following tasks:
        1. Storing instance vars for the loss function and optimizer that will be used when training.
        2. Initializing the optimizer.
        3. Doing a "pilot run" forward pass with a single fake data sample that has the same shape as those that will be
        used when training. This will trigger each weight layer's lazy initialization to initialize weights, biases, and
        any other parameters.
        4. (Optional) Print a summary of the network architecture (layers + shapes) now that we have initialized all the
        layer parameters and know what the shapes will be.
        5. Get references to all the trainable parameters (e.g. wts, biases) from all network layers. This list will be
        used during backpropogation to efficiently update all the network parameters.

        Parameters:
        -----------
        loss: str.
            Loss function to use during training.
        optimizer: str.
            Optimizer to use to train trainable parameters in the network. Initially supported options: 'adam'.
            NOTE: the 'adamw' option will be added later when instructed.
        lr: float.
            Learning rate used by the optimizer during training.
        beta_1: float.
            Hyperparameter in Adam and AdamW optimizers that controls the accumulation of gradients across successive
            parameter updates (in moving average).
        print_summary: bool.
            Whether to print a summary of the network architecture and shapes of activations in each layer.

        TODO: Fill in the section below that should create the supported optimizer. Use TensorFlow Keras optimizers.
        Assign the optimizer to the instance variable `opt`.
        '''
        self.loss_name = loss
        self.optimizer_name = optimizer

        # Initialize optimizer
        #TODO: Fill this section in
        if optimizer == 'adam':
            self.opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, weight_decay=self.reg)
        elif optimizer == 'adamw':
            self.opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=self.reg, beta_1=beta_1)
        else:
            raise ValueError(f'Unknown optimizer {optimizer}')

        # Do 'fake' forward pass through net to create wts/bias
        x_fake = self.get_one_fake_input()
        self(x_fake)

        # Initialize batch norm vars
        self.init_batchnorm_params()

        # Print network arch
        if print_summary:
            self.summary()

        # Get reference to all net params
        self.all_net_params = self.get_all_params()

    def get_one_fake_input(self):
        '''Generates a fake mini-batch of one sample to forward through the network when it is compiled to trigger
        lazy initialization to instantiate the weights and biases in each layer.

        This method is provided to you, so you should not need to modify it.
        '''
        return tf.zeros(shape=(1, *self.input_feats_shape))

    def summary(self):
        '''Traverses the network backward from output layer to print a summary of each layer's name and shape.

        This method is provided to you, so you should not need to modify it.
        '''
        print(75*'-')
        layer = self.output_layer
        while layer is not None:
            print(layer)
            layer = layer.get_prev_layer_or_block()
        print(75*'-')

    def set_layer_training_mode(self, is_training):
        '''Sets the training mode in each network layer.

        Parameters:
        -----------
        is_training: bool.
            True if the network is currently in training mode, False otherwise.

        TODO: Starting with the output layer, traverse the net backward, calling the appropriate method to
        set the training mode in each network layer. Model this process around the summary method.
        '''
        layer = self.output_layer
        while layer is not None:
            layer.set_mode(is_training)
            layer = layer.get_prev_layer_or_block()

    def init_batchnorm_params(self):
        '''Initializes batch norm related parameters in all layers that are using batch normalization.

        (Week 3)

        TODO: Starting with the output layer, traverse the net backward, calling the appropriate method to
        initialize the batch norm parameters in each network layer. Model this process around the summary method.
        '''
        layer = self.output_layer
        while layer is not None:
            layer.init_batchnorm_params() # This method will determine whether it is appropriate or not!
            layer = layer.get_prev_layer_or_block()
        pass

    def get_all_params(self, wts_only=False):
        '''Traverses the network backward from the output layer to compile a list of all trainable network paramters.

        This method is provided to you, so you should not need to modify it.

        Parameters:
        -----------
        wts_only: bool.
            Do we only collect a list of only weights (i.e. no biases or other parameters).

        Returns:
        --------
        Python list.
            List of all trainable parameters across all network layers.
        '''
        all_net_params = []

        layer = self.output_layer
        while layer is not None:
            # print(layer.layer_name)
            # if (layer.b is not None):
            #     print(f"Trainable b: {layer.b.trainable}")
            if wts_only:
                params = layer.get_wts()

                if params is None:
                    params = []
                if not isinstance(params, list):
                    params = [params]
            else:
                params = layer.get_params()

            all_net_params.extend(params)
            layer = layer.get_prev_layer_or_block()
        return all_net_params

    def accuracy(self, y_true, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct.

        Parameters:
        -----------
        y_true: tf.constant. shape=(B,).
            int-coded true classes.
        y_pred: tf.constant. shape=(B,).
            int-coded predicted classes by the network.

        Returns:
        -----------
        float.
            The accuracy in range [0, 1]

        Hint: tf.where might be helpful.
        '''
        correct = tf.where(y_true == y_pred, 1, 0)  # Converts True → 1, False → 0
        return tf.reduce_mean(tf.cast(correct, tf.float32))
        correct = tf.where(y_true == y_pred, 1, 0)
        
        return tf.reduce_mean(correct)
        pass

    def predict(self, x, output_layer_net_act=None):
        '''Predicts the class of each data sample in `x` using the passed in `output_layer_net_act`.
        If `output_layer_net_act` is not passed in, the method should compute it in order to perform the prediction.

        Parameters:
        -----------
        x: tf.constant. shape=(B, ...). Data samples
        output_layer_net_act: tf.constant. shape=(B, C) or None. Network activation.

        Returns:
        -----------
        tf.constant. tf.ints32. shape=(B,).
            int-coded predicted class for each sample in the mini-batch.
        '''
        net_act = output_layer_net_act
        if net_act is None:
            net_act = self(x)
        tf_max_index = tf.argmax(net_act, axis=1, output_type=tf.int32)  # Get class index per sample
        return tf_max_index

    def loss(self, out_net_act, y, eps=1e-16):
        '''Computes the loss for the current minibatch based on the output layer activations `out_net_act` and int-coded
        class labels `y`.

        Parameters:
        -----------
        output_layer_net_act: tf.constant. shape=(B, C) or None.
            Net activation in the output layer for the current mini-batch.
        y: tf.constant. shape=(B,). tf.int32s.
            int-coded true classes for the current mini-batch.

        Returns:
        -----------
        float.
            The loss.

        TODO:
        1. Compute the loss that the user specified when calling compile. As of Project 1, the only option that
        should be supported/implemented is 'cross_entropy' for general cross-entropy loss.
        2. Throw an error if the the user specified loss is not supported.

        NOTE: I would like you to implement cross-entropy loss "from scratch" here — i.e. using the equation provided
        in the notebook, NOT using a TF high level function. For your convenience, I am providing the `arange_index`
        function in tf_util.py that offers functionality that is similar to arange indexing in NumPy (which you cannot
        do in TensorFlow). Use it!
        '''
        # method from Professor, just like arange indexing
        post_chosen_acts = arange_index(out_net_act, y)


        if self.loss_name == 'cross_entropy':
            loss = -tf.reduce_mean(tf.math.log(post_chosen_acts + eps))
        else:
            raise ValueError(f'Unknown loss function {self.loss_name}')

        # Keep the following code
        # Handles the regularization for Adam
        if self.optimizer_name.lower() == 'adam':
            all_net_wts = self.get_all_params(wts_only=True)
            reg_term = self.reg*0.5*tf.reduce_sum([tf.reduce_sum(wts**2) for wts in all_net_wts])
            loss = loss + reg_term

        return loss

    def update_params(self, tape, loss):
        '''Do backpropogation: have the optimizer update the network parameters recorded on `tape` based on the
        gradients computed of `loss` with respect to each of the parameters. The variable `self.all_net_params`
        represents a 1D list of references to ALL trainable parameters in every layer of the network
        (see compile method).

        Parameters:
        -----------
        tape: tf.GradientTape.
            Gradient tape object on which all the gradients have been recorded for the most recent forward pass.
        loss: tf.Variable. float.
            The loss computed over the current mini-batch.
        '''
        grads = tape.gradient(loss, self.all_net_params)
        self.opt.apply_gradients(zip(grads, self.all_net_params))

    # @tf.function(jit_compile=True)
    @tf.function
    def train_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during training. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Updating the network parameters using backprop (via update_params method).

        Parameters:
        -----------
        x_batch: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The loss.

        NOTE: Don't forget to record gradients on a gradient tape!
        '''
        with tf.GradientTape() as tape: #NOTE FOR EXAM Know gradient tape perhaps??
            net_act = self(x_batch)
            loss = self.loss(net_act, y_batch)
            self.update_params(tape, loss)
        return(loss)

    # @tf.function(jit_compile=True)
    @tf.function
    def test_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during test/validation time. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Obtaining the predicted classes for the mini-batch samples.
        4. Compute the accuracy of the predictions.

        Parameters:
        -----------
        x_batch: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.

        NOTE: There should not be any gradient tapes here.
        '''
        net_act = self(x_batch)
        loss = self.loss(net_act, y_batch)
        y_pred = self.predict(x_batch, net_act)
        acc = self.accuracy(y_batch, y_pred)

        return(acc, loss)

    def fit(self, x, y, x_val=None, y_val=None, batch_size=128, max_epochs=10000, val_every=1, verbose=True,
            patience=999, lr_patience=999, lr_decay_factor=0.5, lr_max_decays=12):
        '''Trains the neural network on the training samples `x` (and associated int-coded labels `y`).

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(N, Iy, Ix, n_chans).
            The data samples.
        y: tf.constant. tf.int32s. shape=(N,).
            int-coded class labels
        x_val: tf.constant. tf.float32. shape=(N_val, Iy, Ix, n_chans).
            Validation set samples.
        y_val: tf.constant. tf.float32. shape=(N_val,).
            int-coded validation set class labels.
        batch_size: int.
            Number of samples to include in each mini-batch.
        max_epochs: int.
            Network should train no more than this many epochs.
            Why it is not just called `epochs` will be revealed in Week 2.
        val_every: int.
            How often (in epoches) to compute validation set accuracy and loss.
        verbose: bool.
            If `False`, there should be no print outs during training. Messages indicating start and end of training are
            fine.
        patience: int.
            Number of most recent computations of the validation set loss to consider when deciding whether to stop
            training early (before `max_epochs` is reached).
            NOTE: Ignore Week 1 and until instructed otherwise.
        lr_patience: int.
            Number of most recent computations of the validation set loss to consider when deciding whether to decay the
            optimizer learning rate.
            NOTE: Ignore Week 1 and 2 and until instructed otherwise.
        lr_decay_factor: float.
            A value between 0.0. and 1.0 that represents the proportion of the current learning rate that the learning
            rate should be set to. For example, 0.7 would mean the learning rate is set to 70% its previous value.
            NOTE: Ignore Week 1 and 2 and until instructed otherwise.
        lr_max_decays: int.
            Number of times we allow the lr to decay during training.
            NOTE: Ignore Week 1 and 2 and until instructed otherwise.

        Returns:
        -----------
        train_loss_hist: Python list of floats. len=num_epochs.
            Training loss computed on each training mini-batch and averaged across all mini-batchs in one epoch.
        val_loss_hist: Python list of floats. len=num_epochs/val_freq.
            Loss computed on the validation set every time it is checked (`val_every`).
        val_acc_hist: Python list of floats. len=num_epochs/val_freq.
            Accuracy computed on the validation every time it is checked  (`val_every`).
        e: int.
            The number of training epochs used

        TODO:
        0. To properly handle Dropout layers in your network, set the mode of all layers in the network to train mode
        before the training loop begins.
        1. Process the data in mini-batches of size `batch_size` for each training epoch. Use the strategy recommended
        in CS343 for sampling the dataset randomly WITH replacement.
            NOTE: I suggest using NumPy to create a RNG (before the training loop) with a fixed random seed to
            generate mini-batch indices. That way you can ensure that differing results you get across training runs are
            not just due to your random choice of samples in mini-batches. This should probably be your ONLY use of
            NumPy in `DeepNetwork`.
        2. Call `train_step` to handle the forward and backward pass on the mini-batch.
        3. Average and record training loss values across all mini-batches in each epoch (i.e. one avg per epoch).
        4. If we are at the end of an appropriate epoch (determined by `val_every`):
            - Check and record the acc/loss on the validation set.
            - Print out: current epoch, training loss, val loss, val acc
        5. Regardless of `val_every`, print out the current epoch number (and the total number). Use the time module to
        also print out the time it took to complete the current epoch. Try to print the time and epoch number on the
        same line to reduce clutter.

        NOTE:
        - The provided `evaluate` method (below) should be useful for computing the validation acc+loss ;)
        - `evaluate` kicks all the network layers out of training mode (as is required bc it is doing prediction).
        Be sure to bring the network layers back into training mode after you are doing computing val acc+loss.
        '''
        # Set all layers to training mode
        self.set_layer_training_mode(is_training=True)

        # Histories to record loss and accuracy over epochs
        train_loss_hist = []
        val_loss_hist = []
        val_acc_hist = []

        # Number of training samples (assumes x is a tensor or array-like with shape[0] = N)
        N = x.shape[0]
        
        # Use a fixed random seed for reproducibility when sampling mini-batches
        np.random.seed(0)
        
        # val early stop list
        val_early_stop = []
        
        # val early stop list
        val_decay_lr = []
        num_prev_lr_decays = 0

        # Main training loop
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            # Compute number of mini-batches for this epoch
            num_batches = int(np.ceil(N / batch_size))
            epoch_train_loss = 0.0
            
            # Process mini-batches with replacement
            for i in range(num_batches):
                # Sample indices with replacement
                batch_indices = np.random.choice(N, size=batch_size, replace=True)
                x_batch = tf.gather(x, batch_indices)
                y_batch = tf.gather(y, batch_indices)
                
                # Perform one training step (forward + backward) on this mini-batch
                loss_value = self.train_step(x_batch, y_batch)
                epoch_train_loss += float(loss_value)
            
            # Average training loss for this epoch
            avg_train_loss = epoch_train_loss / num_batches
            train_loss_hist.append(avg_train_loss)
            
            # Every val_every epochs, compute validation metrics
            if (epoch + 1) % val_every == 0:
                # Evaluate on validation set if provided; otherwise, evaluate on the training set
                if x_val is not None and y_val is not None:
                    val_acc, val_loss = self.evaluate(x_val, y_val, batch_sz=batch_size)
                else:
                    val_acc, val_loss = self.evaluate(x, y, batch_sz=batch_size)
                
                val_loss_hist.append(float(val_loss))
                val_acc_hist.append(float(val_acc))
                
                # After evaluation, set layers back to training mode (since evaluate() sets them to False)
                self.set_layer_training_mode(is_training=True)
                
                if verbose:
                    epoch_time = time.time() - epoch_start_time
                    print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s")
                    
                val_early_stop, stop = self.early_stopping(val_early_stop, val_loss, patience)
                if stop:
                    break

                val_decay_lr, decay_bool = self.early_stopping(val_decay_lr, val_loss, lr_patience)
                if (num_prev_lr_decays < lr_max_decays) and decay_bool:
                    self.update_lr(lr_decay_factor)
                    num_prev_lr_decays += 1

            else:
                if verbose:
                    epoch_time = time.time() - epoch_start_time
                    print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # (For Week 1, early stopping and learning rate decay are ignored.)
        
        e = epoch + 1  # Total number of epochs executed
        print(f'Finished training after {e} epochs!')
        return train_loss_hist, val_loss_hist, val_acc_hist, e

    def evaluate(self, x, y, batch_sz=64):
        '''Evaluates the accuracy and loss on the data `x` and labels `y`. Breaks the dataset into mini-batches for you
        for efficiency.

        This method is provided to you, so you should not need to modify it.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(N, Iy, Ix, n_chans).
            The complete dataset or one of its splits (train/val/test/dev).
        y: tf.constant. tf.ints32. shape=(N,).
            int-coded labels of samples in the complete dataset or one of its splits (train/val/test/dev).
        batch_sz: int.
            The batch size used to process the provided dataset. Larger numbers will generally execute faster, but
            all samples (and activations they create in the net) in the batch need to be maintained in memory at a time,
            which can result in crashes/strange behavior due to running out of memory.
            The default batch size should work fine throughout the semester and its unlikely you will need to change it.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.
        '''
        # Set the mode in all layers to the non-training mode
        self.set_layer_training_mode(is_training=False)

        # Make sure the mini-batch size isn't larger than the number of available samples
        N = len(x)
        if batch_sz > N:
            batch_sz = N

        num_batches = N // batch_sz

        # Make sure the mini-batch size is positive...
        if num_batches < 1:
            num_batches = 1

        # Process the dataset in mini-batches by the network, evaluating and avging the acc and loss across batches.
        loss = acc = 0
        for b in range(num_batches):
            curr_x = x[b*batch_sz:(b+1)*batch_sz]
            curr_y = y[b*batch_sz:(b+1)*batch_sz]

            curr_acc, curr_loss = self.test_step(curr_x, curr_y)
            acc += curr_acc
            loss += curr_loss
        acc /= num_batches
        loss /= num_batches

        return acc, loss

    def early_stopping(self, recent_val_losses, curr_val_loss, patience):
        '''Helper method used during training to determine whether training should stop before the maximum number of
        training epochs is reached based on the most recent loss values computed on the validation set
        (`recent_val_losses`) the validation loss on the current epoch (`curr_val_loss`) and `patience`.

        (Week 2)

        - When training begins, the recent history of validation loss values `recent_val_losses` is empty (i.e. `[]`).
        When we have fewer entries in `recent_val_losses` than the `patience`, then we just insert the current val loss.
        - The length of `recent_val_losses` should not exceed `patience` (only the most recent `patience` loss values
        are considered).
        - The recent history of validation loss values (`recent_val_losses`) is assumed to be a "rolling list" or queue.
        Remove the oldest loss value and insert the current validation loss into the list. You may keep track of the
        full history of validation loss values during training, but maintain a separate list in `fit()` for this.

        Conditions that determine whether to stop training early:
        - We never stop early when the number of validation loss values in the recent history list is less than patience
        (training is just starting out).
        - We stop early when the OLDEST rolling validation loss (`curr_val_loss`) is smaller than all recent validation
        loss values. IMPORTANT: Assume that `curr_val_loss` IS one of the recent loss values — so the oldest loss value
        should be compared with `patience`-1 other more recent loss values.

        Parameters:
        -----------
        recent_val_losses: Python list of floats. len between 0 and `patience` (inclusive).
            Recently computed losses on the validation set.
        curr_val_loss: float
            The loss computed on the validation set on the current training epoch.
        patience: int.
            The patience: how many recent loss values computed on the validation set we should consider when deciding
            whether to stop training early.

        Returns:
        -----------
        recent_val_losses: Python list of floats. len between 1 and `patience` (inclusive).
            The list of recent validation loss values passsed into this method updated to include the current validation
            loss.
        stop. bool.
            Should we stop training based on the recent validation loss values and the patience value?

        NOTE:
        - This method can be concisely implemented entirely with regular Python (TensorFlow/Numpy not needed).
        - It may be helpful to think of `recent_val_losses` as a queue: the current loss value always gets inserted
        either at the beginning or end. The oldest value is then always on the other end of the list.
        '''
        # if fewer losses than patience, add current loss and continue training
        if len(recent_val_losses) < patience:
            recent_val_losses.append(curr_val_loss)
            return recent_val_losses, False
            
        # remove oldest loss if we've reached patience limit
        if len(recent_val_losses) == patience:
            recent_val_losses.pop(0)
        
        # add current loss to end of list
        recent_val_losses.append(curr_val_loss)
        
        # get the oldest loss (first in list) to compare against more recent ones
        oldest_loss = recent_val_losses[0]
        
        # check if oldest loss is smaller than all more recent losses
        stop = all(oldest_loss < loss for loss in recent_val_losses[1:])
        
        return recent_val_losses, stop
    
    def update_lr(self, lr_decay_rate):
        '''Adjusts the learning rate used by the optimizer to be a proportion `lr_decay_rate` of the current learning
        rate.

        (Week 3)

        Paramters:
        ----------
        lr_decay_rate: float.
            A value between 0.0. and 1.0 that represents the proportion of the current learning rate that the learning
            rate should be set to. For example, 0.7 would mean the learning rate is set to 70% its previous value.

        NOTE: TensorFlow optimizer objects store the learning rate as a field called learning_rate.
        Example: self.opt.learning_rate for the optimizer object named self.opt. You are allowed to modify it with
        regular Python assignment.

        TODO:
        1. Update the optimizer's learning rate.
        2. Print out the optimizer's learning rate before and after the change.
        '''
        print('Current lr=', self.opt.learning_rate.numpy(), end=' ')
        self.opt.learning_rate = self.opt.learning_rate * lr_decay_rate
        print('Updated lr=', self.opt.learning_rate.numpy())
