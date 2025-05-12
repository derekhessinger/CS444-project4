'''preprocess_corpus.py
Loads text data and preprocesses it into a char representation to train character-level models
YOUR NAMES HERE
CS 444: Deep Learning
Project 4: Transformers
'''
import numpy as np
import tensorflow as tf


def load_document(path2data='data/shakespeare.txt'):
    '''Reads in the document located at `path2data` and determines the vocabulary for a character-level model.

    This function is provided to you. You should not need to modify it.

    Parameters:
    -----------
    path2data: str.
        Path to the text file that should be read in and used for the corpus.

    Returns:
    --------
    str.
        The corpus, defined as the entire document represented as a large single string.
    Python list of str.
        The vocabulary, the list of all the unique chars in the corpus.
    '''
    # Read in text file as a single string
    with open(path2data, 'r') as fp:
        corpus = fp.read()

    # Get a list of unique chars
    vocab = sorted(list(set(corpus)))

    return corpus, vocab


def make_char2ind_map(vocab):
    '''Makes the dictionary that maps char (str) → int index.

    This function is provided to you. You should not need to modify it.

    Parameters:
    -----------
    vocab: Python list of str.
        The vocabulary, the list of all the unique chars in the corpus.

    Returns:
    --------
    Dictionary mapping str → int.
        Maps each char to its position in the vocabulary.
    '''
    return dict((word, i) for i, word in enumerate(vocab))


def make_seqs_and_labels(corpus, char2ind_map, seq_len):
    '''Makes the sequences and labels from the text corpus `corpus`, a large single string. The labels are the next
    chars for each char in the sequences.

    This function is provided to you. You should not need to modify it.

    Here is the strategy to determine the seqs and labels:
    - Walk the corpus from start to finish.
    - Process the corpus in non-overlapping segments/"windows" of `seq_len`
    - The sequences are simply the corpus chars within the current window of size `seq_len`.
    - The labels are the chars one-to-the-right of the chars grabbed for each sequence.
    - If the final few chars do not fit into a full window, just truncate and ignore them.
    - All sequences and labels should be int-coded.

    Example (without int-coding): corpus='abcdefgh'. seq_len=3.
    seq1 = ['a', 'b', 'c'], labels1 = ['b', 'c', 'd']
    seq2 = ['d', 'e', 'f'], labels2 = ['e', 'f', 'g']
    done.

    Parameters:
    -----------
    corpus: str.
        The corpus of text.
    char2ind_map: Dictionary mapping str → int.
        Maps each char to its position in the vocabulary.
    seq_len: int.
        The length of sequences of tokens to create.

    Returns:
    --------
    x_int: tf.constant. tf.int32s. shape=(num_windows, T).
        The `N` int-coded sequences with `T` time steps.
    y_int: tf.constant. tf.int32s. shape=(num_windows, T).
        The `N` int-coded labels with `T` time steps.
    '''
    # Number of nonoverlapping windows, accounting for labels being shifted shifted right by 1
    num_windows = (len(corpus) - 1) // seq_len
    x = np.zeros([num_windows, seq_len], dtype=int)
    y = np.zeros([num_windows, seq_len], dtype=int)

    for i in range(num_windows):
        x[i] = [char2ind_map[char] for char in corpus[i*seq_len:(i+1)*seq_len]]
        y[i] = [char2ind_map[char] for char in corpus[i*seq_len+1:(i+1)*seq_len+1]]


    x = tf.constant(x, dtype=tf.int32)
    y = tf.constant(y, dtype=tf.int32)
    return x, y
