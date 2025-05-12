'''gpts.py
Family of Generated Pretrained Transformer (GPT) neural networks.
YOUR NAMES HERE
CS 444: Deep Learning
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import network
from tf_util import arange_index
from layers import Dense
from transformer_layers import Embedding
from transformer_blocks import TransformerBlock, PositionalEncodingBlock


class GPT(network.DeepNetwork):
    '''The parent class of all GPT networks. Customizes the functionality of DeepNetwork for transformers.
    '''
    def __init__(self, seq_len, padding_char_enc, reg=0):
        '''GPT constructor

        Parameters:
        -----------
        seq_len: int.
            The length of sequences processed by the transformer (`T`).
        padding_char_enc: int.
            The INT-CODED padding char. Needed to compute the temporal cross-entropy loss.
        reg: float.
            Regularization strength.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create instance variables for the method parameters as needed.
        '''
        super().__init__(seq_len, reg)
        self.padding_char_enc = padding_char_enc
        pass

    def get_one_fake_input(self):
        '''Generates a fake input for use in the `compile` method to trigger lazy initialization to initialize network
        parameters.

        This function is provided to you. You should not need to modify it.
        '''
        return tf.zeros(shape=(1, *self.input_feats_shape), dtype=tf.int32)

    def loss(self, out_net_act, y, eps=1e-16, mask_padding_preds=True):
        '''Computes the loss for the current minibatch based on the output layer activations `out_net_act` and int-coded
        class labels `y`.

        This function is provided to you. You should not need to modify it.

        Parameters:
        -----------
        output_layer_net_act: tf.constant. shape=(B, T, C) or None.
            Net activation in the output layer for the current mini-batch.
        y: tf.constant. shape=(B,). tf.int32s.
            int-coded true classes for the current mini-batch.
        eps: float.
            Small fudge factor to use when evaluating the cross entropy log to prevent taking log of 0.
        mask_padding_preds: bool.
            Should we not count time steps in mini-batch sequences when the current character is the padding character
            and the prediction would then be meaningless?1

        Returns:
        -----------
        float.
            The loss.

        NOTE:
        - We treat the time dimension as additional data samples from the perspective of the loss function.
        - When we are not masking out padding chars from the temporal cross entropy loss, the loss simplifies to
        regular cross entropy loss.
        '''
        N, T, vocab_sz = out_net_act.shape

        if self.loss_name == 'cross_entropy':
            act_at_correct = arange_index(out_net_act, y)
            loss = tf.reduce_mean(-tf.math.log(act_at_correct + eps)) # sparse int based
        elif self.loss_name == 'temporal_cross_entropy': # NEW TRANSFORMERS
            # Our input is 3D so we need to combine/treat time as more batches.
            out_net_act_flat = tf.reshape(out_net_act, [N*T, vocab_sz])
            y_flat = tf.reshape(y, (N*T,))

            # Compute the per-sample loss like usual
            act_at_correct = arange_index(out_net_act_flat, y_flat)
            loss = -tf.math.log(act_at_correct + eps)
            #print(tf.reshape(act_at_correct, (N, T)))

            if mask_padding_preds:
                # We want to nix padding chars from the loss, so mask them out
                mask = tf.cast(y_flat != self.padding_char_enc, dtype=tf.float32)
                N_effective = tf.reduce_sum(mask)  # We want to avg loss by number of real non-pad contributions
                loss = mask * loss  # (N*T,) but only non-padded chars contribute
                loss = (1/N_effective) * tf.reduce_sum(loss) # sparse int based
            else:
                # Average all per-sample loss values together
                loss = tf.reduce_mean(loss)
        else:
            raise ValueError(f'Unknown loss function {self.loss_name}')

        return loss

    def __call__(self, x):
        '''Forward pass through the Transformer with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, M).
            B data samples (seqs) of length T. Each time step has `M`=vocab_sz features.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, C).
            Activations produced by the output layer to the mini-batch for all time steps and possible tokens in the
            vocab.

        Hint: All layers/blocks in all GPTs are connected sequentially at the macro network level...;)
        '''
        net_act = x
        for cur_layer in self.layers:
            net_act = cur_layer(net_act)
        return net_act

    def generate_sequence(self, prompt, length, char2ind_map, ind2char_map, end_char=None, method='max',
                          plot_probs=False, live_print=True):
        '''Generates/predicts a sequence of chars of length `length` chars that follow the provided prompt.
        It is helpful remember that the transformer generates chars one at a time sequentially. Therefore in
        prediction/generation mode, the network processes tokens in mini-batches of one item for one time step.

        This function is provided to you. You should not need to modify it.

        Parameters:
        -----------
        prompt: str.
            Chars that the transformer should process and predict chars after. Treated as a single sequence (B=1) with
            a length Tp that is probably != T (seq_len used to train transformer). So this method right-pads prompt
            until it is length T, which is the length the transformer is used to dealing with.
        length: int.
            Number of chars that the transformer generates after the prompt chars.
        char2ind_map: Python dictionary.
            Keys: chars in vocab. Values: int code of a char in the vocab.
        ind2char_map: Python dictionary.
            Keys: int code of a char in the vocab. Values: Which char it corresponds to in the vocab.

        Returns:
        -----------
        str. len=(len(prompt) + `length`).
            The provided prompt concatenated with the set of transformer generated chars.
        '''
        # Turn off Dropout
        self.set_layer_training_mode(is_training=False)

        prompt_ints = []
        # We want to convert prompt (str) to B,T (where B = 1)
        # Convert chars -> int
        prompt_ints = [char2ind_map[char] for char in prompt]  # (N_gen,)

        # response (int coded) we will return
        response_int = []

        # RNG for distributed multinomial sampling
        rng = np.random.default_rng()

        # print prompt
        if live_print:
            print(prompt, sep='', end='')

        # Where we get the next char prediction from the output layer net_acts
        t_ind = len(prompt_ints)-1

        # Second: now that state has developed, generate new chars using a feedback loop (prev pred = next input)
        for t in range(length):
            # Now we assemble the context that is fed to the transformer. This is the user's prompt + what has been
            # generated
            context = prompt_ints + response_int

            # Ensure we keep the context length to the T used to train the transformer. Use the T most recent tokens as
            # a context (rolling window)
            if len(context) > self.seq_len:
                context = context[-self.seq_len:]

            # Make sure we left-pad so that the context input we give the transformer is always seq_len long
            # and the prompt is next to the generated text
            while len(context) < self.seq_len:
                context.append(self.padding_char_enc)

            # Convert int into Tensor format (1, T)
            x_input_tf = tf.expand_dims(context, axis=0)  # N, T
            # Get net_acts from output layer
            out_net_act = self(x_input_tf)

            # Important: Squeeze / convert to numpy before drawing from softmax dist.
            out_probs_np = out_net_act[0, t_ind].numpy()  # probs from only last time step
            # Draw predicted char index from vocab proportional to the softmax prob
            if method == 'max':
                curr_pred_int = tf.argmax(out_net_act[0, t_ind, :]).numpy()
            else:
                curr_pred_int = int(rng.choice(np.arange(len(out_probs_np)), p=out_probs_np))

            if plot_probs:
                plt.plot(out_probs_np)
                plt.title(f'{t}th char: pred char: {ind2char_map[curr_pred_int]} ({curr_pred_int})')
                plt.show()

            # Append to generated seq, the char corresponding to the predicted index in the vocab
            response_int.append(curr_pred_int)

            # Advance where we get our prediction next time.
            if t_ind < self.seq_len - 1 and t_ind > 0:
                t_ind += 1
            else:
                t_ind = -1

            # live next char live as we go
            if live_print:
                print(ind2char_map[curr_pred_int], sep='', end='')

            if end_char is not None and curr_pred_int == char2ind_map[end_char]:
                break

        if live_print:
            print()

        seq_char = [ind2char_map[index] for index in response_int]

        if plot_probs:
            plt.show()

        return seq_char


class GPTPico1(GPT):
    '''GPTPico1 is a "baby GPT" — the shallowest transformer in the family. It consists of only one transformer block.

    Here is a summary the architecture:

    Embedding layer → Position Encoding block → Transformer block (1x) → Dense output layer

    In the output layer:
    - Use He init.
    - Use layer norm.
    '''
    def __init__(self, vocab_sz, seq_len, padding_char_enc=None, num_heads=4, embed_dim=24, dropout_rate=0.0):
        '''GPTPico1 constructor

        Parameters:
        -----------
        vocab_sz: int.
            The vocab size (`M`).
        seq_len: int.
            Length of the sequences, in time steps, that the transformer processes (`T`)
        padding_char_enc: int.
            The INT-CODED padding char. Needed to compute the temporal cross-entropy loss.
        num_heads: int.
            Number of attention heads to use in the attention blocks.
        embed_dim: int.
            The number of units in the Embedding layer.
        dropout_rate: float.
            The dropout rate (R) to use throughout the net.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        super().__init__(seq_len, padding_char_enc)
        self.seq_len = seq_len
        self.vocab_sz = vocab_sz
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        # Create all layers and blocks
        self.layers = []

        # embedding layer
        embed_layer = Embedding('Embedding_Layer_0', vocab_sz, self.embed_dim)
        self.layers.append(embed_layer)
        # pos enc block
        pos_block = PositionalEncodingBlock('Positional_Encoding_Block_0', self.embed_dim, prev_layer_or_block=embed_layer, 
                                            dropout_rate=self.dropout_rate)
        self.layers.append(pos_block)
        # transformer 0
        trans_0 = TransformerBlock('Transformer_Block_0', embed_dim, self.num_heads, prev_layer_or_block=pos_block, 
                                   dropout_rate=self.dropout_rate)
        self.layers.append(trans_0)

        # Add after the transformer block:
        output_layer = Dense(f"Output_Layer", units=self.vocab_sz, 
                            activation='softmax', prev_layer_or_block=trans_0, 
                            wt_init='he', do_batch_norm=False, do_layer_norm=True)
        self.layers.append(output_layer)

        # Then set the output layer correctly:
        self.output_layer = output_layer


class GPTMini6(GPT):
    '''GPTMini6 is a moderate-sized GPT in the family. It consists of 6 transformer blocks.

    Here is a summary the architecture:

    Embedding layer → Position Encoding block → Transformer block (6x) → Dense output layer

    In the output layer:
    - Use He init.
    - Use layer norm.
    '''
    def __init__(self, vocab_sz, seq_len, padding_char_enc=None, num_heads=6, embed_dim=384, dropout_rate=0.2):
        '''GPTMini6 constructor

        Parameters:
        -----------
        vocab_sz: int.
            The vocab size (`M`).
        seq_len: int.
            Length of the sequences, in time steps, that the transformer processes (`T`)
        padding_char_enc: int.
            The INT-CODED padding char. Needed to compute the temporal cross-entropy loss.
        num_heads: int.
            Number of attention heads to use in the attention blocks.
        embed_dim: int.
            The number of units in the Embedding layer.
        dropout_rate: float.
            The dropout rate (R) to use throughout the net.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        super().__init__(seq_len, padding_char_enc)
        self.seq_len = seq_len
        self.vocab_sz = vocab_sz
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        
        # Create all layers and blocks
        self.layers = []

        # embedding layer
        embed_layer = Embedding('Embedding_Layer_0', vocab_sz, self.embed_dim)
        self.layers.append(embed_layer)
        # pos enc block
        pos_block = PositionalEncodingBlock('Positional_Encoding_Block_0', self.embed_dim, prev_layer_or_block=embed_layer, 
                                            dropout_rate=self.dropout_rate)
        self.layers.append(pos_block)
        # transformer 0
        trans_0 = TransformerBlock('Transformer_Block_0', embed_dim, self.num_heads, prev_layer_or_block=pos_block, 
                                   dropout_rate=self.dropout_rate)
        self.layers.append(trans_0)
        # transformer 1
        trans_1 = TransformerBlock('Transformer_Block_1', embed_dim, self.num_heads, prev_layer_or_block=trans_0, 
                                   dropout_rate=self.dropout_rate)
        self.layers.append(trans_1)
        # transformer 2
        trans_2 = TransformerBlock('Transformer_Block_2', embed_dim, self.num_heads, prev_layer_or_block=trans_1, 
                                   dropout_rate=self.dropout_rate)
        self.layers.append(trans_2)
        # transformer 3
        trans_3 = TransformerBlock('Transformer_Block_3', embed_dim, self.num_heads, prev_layer_or_block=trans_2, 
                                   dropout_rate=self.dropout_rate)
        self.layers.append(trans_3)
        # transformer 4
        trans_4 = TransformerBlock('Transformer_Block_4', embed_dim, self.num_heads, prev_layer_or_block=trans_3, 
                                   dropout_rate=self.dropout_rate)  
        self.layers.append(trans_4)
        # transformer 5
        trans_5 = TransformerBlock('Transformer_Block_5', embed_dim, self.num_heads, prev_layer_or_block=trans_4, 
                                   dropout_rate=self.dropout_rate)  
        self.layers.append(trans_5)

        # Add to the end of your GPTMini6.__init__ method
        # Output layer - converts from embed_dim to vocab_sz at each position
        output_layer = Dense('Output_layer', units=self.vocab_sz, 
                            activation='softmax', prev_layer_or_block=trans_5, 
                            wt_init='he', do_batch_norm=False, do_layer_norm=True)
        self.layers.append(output_layer)

        # Store reference to the output layer for the network
        self.output_layer = output_layer

