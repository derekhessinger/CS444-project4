'''transformer_blocks.py
Blocks related to transformer neural networks.
YOUR NAMES HERE
CS 444: Deep Learning
'''
import tensorflow as tf

import block
from layers import Dense, Dropout
from transformer_layers import PositionalEncoding
from tf_util import tril


class QueryKeyValueBlock(block.Block):
    '''Block that encapsulates the Dense layers that generate the queries, keys, and values.'''
    def __init__(self, blockname, units, prev_layer_or_block):
        '''QueryKeyValueBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        units. int.
            The number of neurons in each of the Dense layers in the block. All Dense layers have the same number of
            units (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        Properties of all layers:
        ---------------------------
        - They are along separate branches. Think about what this means for their previous layer/block reference.
        - He initialization.
        - Layer normalization.
        - Linear/identity activation.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Assemble layers in the block.
        '''
        super().__init__(blockname, prev_layer_or_block=prev_layer_or_block)
        self.units = units
        query = Dense('QKVBlock_Query', units, activation='linear', prev_layer_or_block=prev_layer_or_block, wt_init='he',
                      do_batch_norm = False, do_layer_norm = True)
        self.layers.append(query)
        
        key = Dense('QKVBlock_Key', units, activation='linear', prev_layer_or_block=prev_layer_or_block, wt_init='he',
                      do_batch_norm = False, do_layer_norm = True)
        self.layers.append(key)
        
        value = Dense('QKVBlock_Value', units, activation='linear', prev_layer_or_block=prev_layer_or_block, wt_init='he',
                      do_batch_norm = False, do_layer_norm = True)
        self.layers.append(value)
        pass

    def __call__(self, query_input, key_input, value_input):
        '''Forward pass through the QKV Block with activations that should represent the input to respective QKV layers.

        Parameters:
        -----------
        query_input: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from a prev layer/block that are the input to the query layer.
        key_input: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from a prev layer/block that are the input to the key layer.
        value_input: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from a prev layer/block that are the input to the value layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            Activations produced by the query layer.
        tf.constant. tf.float32s. shape=(B, T, H).
            Activations produced by the key layer.
        tf.constant. tf.float32s. shape=(B, T, H).
            Activations produced by the value layer.
        '''
        layers = self.layers
        query_act = layers[0](query_input)
        key_act = layers[1](key_input)
        value_act = layers[2](value_input)
        return query_act, key_act, value_act
        pass


class AttentionBlock(block.Block):
    '''Block that encapsulates the fundamental attention mechanism.'''
    def __init__(self, blockname, num_heads, units, prev_layer_or_block, dropout_rate=0.1, causal=True):
        '''AttentionBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        num_heads: int.
            Number of attention heads to use in the attention block.
        units: int.
            Number of neurons in the attention block (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use in the Dropout layer that is in the attention block that is applied to the
            attention values.
        causal: bool.
            Whether to apply a causal mask to remove/mask out the ability for the layer to pay attention to tokens
            that are in the future of the current one in the sequence.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create any instance variables to save any information that will be helpful to access during the forward pass.
        3. Create the Dropout layer.
        4. For efficiency, it is helpful to pre-compute the attention gain and assign it to an instance variable
        (e.g. as self.gain) so that you can use it during the forward pass. You have all the info here that is needed
        to compute the gain.

        NOTE: Remember to add your dropout layer to the layers list (otherwise the dropout mode will not get set) and
        to make the dropout layer's prev reference whatever is passed into this block.
        '''
        super().__init__(blockname, prev_layer_or_block)
        self.H = units
        self.A = num_heads
        self.causal = causal
        # precompute attention gain
        self.gain = 1 / tf.sqrt(tf.cast(self.H / self.A, dtype=tf.float32))
        # create dropout layer
        self.dropout_layer = Dropout('attention_dropout', dropout_rate, prev_layer_or_block)
        self.layers.append(self.dropout_layer)

    def __call__(self, queries, keys, values):
        '''Forward pass through the attention block with activations from the query, key, and value layers.

        Parameters:
        -----------
        queries: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the query layer.
        keys: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the keys layer.
        values: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the values layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The attended values.

        NOTE:
        1. Follow the blueprint from class on computing the various phases of attention (i.e. A_1, A_2, A_3, and A_4).
        2. Refer to the notebook for a refresher on the big-picture equations.
        3. You will need to rely on batch matrix multiplication. The code does not differ from regular multiplication,
        but it affects the setup of the shapes.
        4. It is important to keep track of shapes at the various phases. I suggest keeping track of shapes in code
        comments above each line of code you write.
        5. Don't forget that you pre-computed the attention gain.
        6. Don't forget to incorporate the causal mask to implement causal attention (if that option is turned on).
        The function `tril` from tf_util should be very helpful.11
        '''
        B = tf.shape(queries)[0]
        T = tf.shape(queries)[1]
        H = self.H
        A = self.A 
        H_A = H // A
        
        # reshape queries, keys, values to separate attention heads
        # shape: (B, T, H) -> (B, T, A, H/A)
        queries = tf.reshape(queries, [B, T, A, H_A])
        keys = tf.reshape(keys, [B, T, A, H_A])
        values = tf.reshape(values, [B, T, A, H_A])
        
        # transpose to (B, A, T, H/A) for queries and values, and (B, A, H/A, T) for keys
        queries = tf.transpose(queries, [0, 2, 1, 3])  # (B, A, T, H/A)
        keys = tf.transpose(keys, [0, 2, 3, 1])        # (B, A, H/A, T)
        values = tf.transpose(values, [0, 2, 1, 3])    # (B, A, T, H/A)
        
        # compute attention scores
        A_1 = tf.matmul(queries, keys) * self.gain  # (B, A, T, T)
        
        # apply causal mask if needed
        if self.causal:
            mask = tril(T)
            mask = tf.cast(mask, dtype=tf.float32)
            mask2 = -1e9 * (1.0 - mask)
            A_2 = A_1 + mask2
        else:
            A_2 = A_1
        
        # apply softmax to compute attention weights
        A_3 = tf.nn.softmax(A_2, axis=-1)  # (B, A, T, T)
        
        # apply dropout to attention weights
        A_4 = self.dropout_layer(A_3)  # (B, A, T, T)
        
        # apply attention weights to values
        A_5 = tf.matmul(A_4, values)  # (B, A, T, H/A)
        
        # transpose back
        A_5 = tf.transpose(A_5, [0, 2, 1, 3])  # (B, T, A, H/A)
        
        # reshape back to (B, T, H)
        output = tf.reshape(A_5, [B, T, H])  # (B, T, H)
        
        return output


class MultiHeadAttentionBlock(block.Block):
    '''Block that encapsulates MultiHeadAttention and related blocks. Here is a summary of the block:

    QueryKeyValueBlock → MultiHead Attention → Dense → Dropout

    All the layers/subblocks have H (i.e. num_embed) neurons. The Dense layer uses He init and a linear act fun.

    NOTE: The Dense layer in this block (according to the paper) does NOT use layer norm.
    '''
    def __init__(self, blockname, num_heads, units, prev_layer_or_block, dropout_rate=0.1, causal=True):
        '''MultiHeadAttentionBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        num_heads: int.
            Number of attention heads to use in the attention block.
        units: int.
            Number of neurons in the attention block (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use in the Dropout layer that is in the attention block that is applied to the
            attention values. The dropout rate is the same for the dropout layer in this block and the attention
            subblock.
        causal: bool.
            Whether to apply a causal mask to remove/mask out the ability for the layer to pay attention to tokens
            that are in the future of the current one in the sequence.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        super().__init__(blockname, prev_layer_or_block)
        
        # create the QKV block
        self.qkv_block = QueryKeyValueBlock(f"{blockname}_QKV", units, prev_layer_or_block)
        
        # create the attention block
        self.attention_block = AttentionBlock(f"{blockname}_Attention", num_heads, units, 
                                             prev_layer_or_block, dropout_rate, causal)
        
        # create the final dense layer (with He init and linear activation, no layer norm)
        self.dense = Dense(f"{blockname}_Dense", units, activation='linear', 
                           prev_layer_or_block=prev_layer_or_block, wt_init='he',
                           do_batch_norm=False, do_layer_norm=False)
        
        # create the dropout layer
        self.dropout = Dropout(f"{blockname}_Dropout", dropout_rate, prev_layer_or_block)
        
        # add all components to the layers list
        self.layers.extend([self.qkv_block, self.attention_block, self.dense, self.dropout])



    def __call__(self, x):
        '''Forward pass through the MultiHead Attention Block.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs
        '''
        # generate queries, keys, and values from the input
        queries, keys, values = self.qkv_block(x, x, x)
        
        # pass through the attention block
        attention_output = self.attention_block(queries, keys, values)
        
        # pass through the dense layer
        dense_output = self.dense(attention_output)
        
        # apply dropout and return
        output = self.dropout(dense_output)
        
        return output


class MLPBlock(block.Block):
    '''MLP block that tends to follow the attention block. Composed of the following layers:

    Dense → Dense → Dropout

    Implements a bottleneck design: 1st Dense layer has 4x the units and the 2nd Dense layer has 1x.

    1st Dense layer:
    ----------------
    - Uses the gelu activation function, layernorm

    2nd Dense layer:
    ----------------
    - Uses the linear/identity activation function, no layernorm
    '''
    def __init__(self, blockname, units, prev_layer_or_block, exp_factor=4, dropout_rate=0.1):
        '''MLPBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        units: int.
            Number of neurons in the MLP block dense layers (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        exp_factor: int.
            The expansion factor that scales the number of units in the 1st Dense layer. Controls how large the
            bottleneck is in the block.
        dropout_rate: float.
            The dropout rate (R) to use in the Dropout layer.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        super().__init__(blockname, prev_layer_or_block)
        
        # first dense layer - expanded dimension with gelu activation and layer norm
        self.dense1 = Dense(f"{blockname}_Dense1", units * exp_factor, activation='gelu',
                            prev_layer_or_block=prev_layer_or_block, wt_init='he',
                            do_batch_norm=False, do_layer_norm=True)
        
        # second dense layer - no layer norm
        self.dense2 = Dense(f"{blockname}_Dense2", units, activation='linear',
                            prev_layer_or_block=self.dense1, wt_init='he',
                            do_batch_norm=False, do_layer_norm=False)
        
        # dropout layer
        self.dropout = Dropout(f"{blockname}_Dropout", dropout_rate, prev_layer_or_block=self.dense2)
        
        # add the layers to the block's layer list
        self.layers.extend([self.dense1, self.dense2, self.dropout])

    def __call__(self, x):
        '''Forward pass through the MLPBlock with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs
        '''
        # pass through first dense layer
        x = self.dense1(x)
        
        # pass through second dense layer
        x = self.dense2(x)
        
        # apply dropout
        x = self.dropout(x)
        
        return x


class TransformerBlock(block.Block):
    '''The Transformer Block, composed of a single MultiHeadAtention Block followed by a single MLP Block.'''
    def __init__(self, blockname, units, num_heads, prev_layer_or_block, dropout_rate=0.1):
        '''TransformerBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        units: int.
            Number of neurons in the Transformer block (H — i.e. embed_dim).
        num_heads: int.
            Number of attention heads to use in the attention block.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use throughout the block.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        super().__init__(blockname, prev_layer_or_block)
        
        # # layer normalization for attention branch
        # self.ln1 = Dense(f"{blockname}_LN1", units, activation='linear',
        #                  prev_layer_or_block=prev_layer_or_block, wt_init='he',
        #                  do_batch_norm=False, do_layer_norm=True)
        
        # multihead attention block
        self.mha = MultiHeadAttentionBlock(blockname=f"{blockname}_MHA", num_heads=num_heads, units=units, 
                                          prev_layer_or_block=prev_layer_or_block, dropout_rate=dropout_rate)
        
        # # layer normalization for MLP branch
        # self.ln2 = Dense(f"{blockname}_LN2", units, activation='linear',
        #                  prev_layer_or_block=prev_layer_or_block, wt_init='he',
        #                  do_batch_norm=False, do_layer_norm=True)
        
        # MLP block
        self.mlp = MLPBlock(f"{blockname}_MLP", units, self.mha, dropout_rate=dropout_rate)
        
        # add the components to the block's layer list
        self.layers.extend([self.mha, self.mlp])
        # self.layers.extend([self.ln1, self.mha, self.ln2, self.mlp])


    def __call__(self, x):
        '''Forward pass through the Transformer block with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs

        NOTE: Don't forget the residual connections that allows the input to skip to the end of each block.
        '''
        # # first normal layer
        # norm1 = self.ln1(x)
        
        # multi-head attention with residual connection
        attn_output = self.mha(x)
        attn_output = x + attn_output  # Residual connection
        
        # second normalization layer 
        # norm2 = self.ln2(attn_output)
        
        # MLP with residual connection
        mlp_output = self.mlp(attn_output)
        output = attn_output + mlp_output  # Residual connection
        
        return output


class PositionalEncodingBlock(block.Block):
    '''Block that combines PositionalEncoding layer and a Dropout layer in the following order:

    PositionalEncoding → Dropout
    '''
    def __init__(self, blockname, embed_dim, prev_layer_or_block, dropout_rate=0.1):
        '''PositionalEncodingBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        embed_dim: int.
            Number of neurons in the Embedding layer (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use in the dropout layer1.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers.
        '''
        super().__init__(blockname, prev_layer_or_block)
        pe_layer = PositionalEncoding(f'{blockname}_PE', embed_dim=embed_dim, prev_layer_or_block=prev_layer_or_block)
        self.layers.append(pe_layer)
        
        dropout = Dropout(f"{blockname}_Dropout", dropout_rate, prev_layer_or_block=pe_layer)
        self.layers.append(dropout)

        pass

    def __call__(self, x):
        '''Forward pass through the block with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs
        '''
        netin_pre_drop = self.layers[0](x)
        return self.layers[1](netin_pre_drop)
        pass
