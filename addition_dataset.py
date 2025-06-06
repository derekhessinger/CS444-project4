'''addition_dataset.py
Generates and preprocesses the Addition Dataset and the Addition and Multiplication Dataset
YOUR NAMES HERE
CS 444: Deep Learning
'''
import numpy as np
import tensorflow as tf

def make_addition_expressions(N, max_operand_digits=2, seed=1):
    '''Generates arithmetic expressions involving the addition operator and two `max_operand_digits` digit operands.
    By default, each operand is a 2 digits long positive integer, so the largest sum that can be included is
    99+99=198.

    Includes the special characters:
    - '.' to indicate the end of an expression has been reached. int code = 12.
    - '#' to pad an expression so that all expressions have the same length in chars. int code = 13.

    This function is provided to you. You should not need to modify it.

    Parameters:
    -----------
    N: int.
        Number of expressions to generate.
    max_operand_digits: int.
        Maximum number of digits of each positive integer operand.
        For example if max_operand_digits=1, 9+1=10 would be possible, but not 90+1=91.
    seed: int.
        Random seed that controls the reproducability of generated expressions.

    Returns:
    --------
    Python list of list of chars (str).
        The fixed-length addition expressions. Each expression represented as a list of chars.
        For example: 47+51=98 is represented as: ['4', '7', '+', '5', '1', '=', '9', '8', '.', '#']
    Dictionary mapping str → int.
        Maps each char to its position in the vocabulary.
    '''
    # What is the largest number we can add?
    max_operand = int(max_operand_digits*'9')

    # Determine how long the longest equation will be in characters
    addition_eq_len = 2  # operator + equals sign
    addition_eq_len += 2*max_operand_digits  # worst case: two of the max operands added together
    addition_eq_len += len(str(max_operand+max_operand))  # add in max size of answer
    addition_eq_len += 1  # For end token

    # Generate the operands (e.g. [1, 1]) and their solution (21)
    rng = np.random.default_rng(seed)
    operands = rng.integers(low=0, high=max_operand+1, size=(N, 2))
    answers = np.sum(operands, axis=1)

    # Our "vocabulary": all the chars we can possibly process
    char2ind_map = {}
    for i in range(10):
        char2ind_map[str(i)] = i
    char2ind_map['+'] = 10
    char2ind_map['='] = 11
    char2ind_map['.'] = 12  # Special token: end of equation.
    char2ind_map['#'] = 13  # Special token padding for uneven length expressions

    expressions = []
    for i in range(N):
        # form strings like '2+3=5'
        expression = f'{operands[i][0]}+{operands[i][1]}={answers[i]}.'
        # Pad to ensure all expressions have same length
        expression = f'{expression:#<{addition_eq_len}}'
        expressions.append(list(expression))

    print(f'First 5/{N} expressions:')
    for i in range(5):
        print(' ', expressions[i])

    return expressions, char2ind_map


def make_addition_samples_and_labels(expressionLists, char2ind_map):
    '''Makes the addition dataset int-coded samples and labels. The labels (y_int) are lists of each next char in each
    addition expression. The samples (x_int) are just each char in each addition expression.

    NOTE: The samples cannot include the last char in each expression (bc there is no next char).

    Parameters:
    -----------
    expressionLists: Python list of lists of chars (str).
        The fixed-length addition expressions. Each expression represented as a list of chars.
        Example of one sublist: ['4', '7', '+', '5', '1', '=', '9', '8', '.', '#']
    char2ind_map: Dictionary mapping str → int.
        Maps each char to its position in the vocabulary.

    Returns:
    --------
    x_int: Python list of list of int. len(x_int)=len(y_int).
        The int-coded chars in each addition expression.
    y_int: Python list of list of int. len(x_int)=len(y_int).
        The int-coded labels/targets for each addition expression (i.e. the next chars).
    '''
    # create x_int, y_int lists
    x_int = []
    y_int = []
    # loop through (0, n-1) in EL, map to int, append to x_int
    for expression in expressionLists:
        x_cur = []
        y_cur = []
        for i in range(len(expression)-1):
            cur = char2ind_map[expression[i]]
            x_cur.append(cur)
        x_int.append(x_cur)
        # loop through (1, n) in EL, map to int, append to y_int
        for i in range(1, len(expression)):
            cur = char2ind_map[expression[i]]
            y_cur.append(cur)
        y_int.append(y_cur)

    return x_int, y_int


def make_ind2char_mapping(char2ind_map):
    '''Makes the dictionary that maps int index → char (str).

    Parameters:
    -----------
    char2ind_map: Dictionary mapping str → int.
        Maps each char to its position in the vocabulary.

    Returns:
    --------
    Dictionary mapping int → str.
        Maps ints back into the original chars in the vocabulary.
    '''
    ind2char_map = {}
    for key in char2ind_map.keys():
        ind2char_map[char2ind_map[key]] = key
    return ind2char_map


def convert_int2str(x_int, ind2char_map):
    '''Converts int-coded tokens back to human-readable string representations.

    Parameters:
    -----------
    x_int: Python list of list of int.
        A list of expressions, where each expression is int-coded.
        Example: [4, 7, 10, 5, 1, 11, 9, 8, 12]
    ind2char_map: Dictionary.
        A dictionary mapping integer indices to their corresponding character representations.

    Returns:
    --------
    list of list of str.
        A list of addition expressions, where each expression is represented as a list of characters (str).
        Example: ['4', '7', '+', '5', '1', '=', '9', '8', '.']
    '''
    str_list = []
    for l in x_int:
        cur = []
        for ind in l:
            cur.append(ind2char_map[ind])
        str_list.append(cur)
    return str_list


def make_train_val_split(x, y, val_prop=0.1):
    '''Splits the sequences and associated labels into training and validation sets.

    Parameters:
    -----------
    x: Python list of list of int.
        Each expression is int-coded.
    y: Python list of list of int.
        Target/label for each int-coded token in `x`. Each expression is int-coded.
    val_prop: float.
        The proportion of the data to be used for validation. Assumed to be the last portion of `x` and `y`.

    Returns:
    --------
    x_train: tf.constant. tf.int32s.
        Training set tokens.
    y_train: tf.constant. tf.int32s.
        Training set labels.
    x_val: tf.constant. tf.int32s.
        Validation set tokens.
    y_val: tf.constant. tf.int32s.
        Validation set labels.
    '''
    val_size = int(len(x) * val_prop)
    
    train_size = len(x) - val_size

    x_train = x[:train_size]
    y_train = y[:train_size]
    x_val = x[train_size:]
    y_val = y[train_size:]

    x_train = tf.constant(x_train, dtype=tf.int32)
    y_train = tf.constant(y_train, dtype=tf.int32)
    x_val = tf.constant(x_val, dtype=tf.int32)
    y_val = tf.constant(y_val, dtype=tf.int32)
    
    return x_train, y_train, x_val, y_val

def split_sum_and_answer(x_str):
    '''Splits a list of mathematical expressions into their left-hand side (LHS), everything to the left side and
    including the = char, and answer, everything to the right of the =, components.

    Parameters:
    -----------
    x_str: Python list of list of chars (str).
        The fixed-length addition expressions. Each expression represented as a list of chars.
        For example: 47+51=98 is represented as: ['4', '7', '+', '5', '1', '=', '9', '8', '.', '#']

    Returns:
    --------
    List of str:
        All characters to the left and including the = in each expression, represented as single strings.
        Example: '47+51='
    List of str:
        All characters to the right of the = in each expression, represented as single strings.
        Example: '98.'
    '''
    lhs_list = []
    rhs_list = []
    
    for expression in x_str:
        # Find the index of the equals sign
        equals_index = expression.index('=')
        
        # Get all characters up to and including the equals sign
        lhs = ''.join(expression[:equals_index+1])
        lhs_list.append(lhs)
        
        # Get all characters after the equals sign (excluding padding characters)
        rhs_chars = []
        for char in expression[equals_index+1:]:
            rhs_chars.append(char)
        
        rhs = ''.join(rhs_chars)
        rhs_list.append(rhs)
    
    return lhs_list, rhs_list


def get_addition_dataset(N, max_operand_digits=2, seed=1, val_prop=0.1):
    '''Automates the process of generating and preprocessing the Addition dataset.

    Parameters:
    -----------
    N: int.
        Number of expressions to generate.
    max_operand_digits: int.
        Maximum number of digits of each positive integer operand.
        For example if max_operand_digits=1, 9+1=10 would be possible, but not 90+1=91.
    seed: int.
        Random seed that controls the reproducability of generated expressions.
    val_prop: float.
        The proportion of the data to be used for validation. Assumed to be the last portion of `x` and `y`.

    Returns:
    --------
    x_train: tf.constant. tf.int32s.
        Training set tokens.
    y_train: tf.constant. tf.int32s.
        Training set labels.
    x_val: tf.constant. tf.int32s.
        Validation set tokens.
    y_val: tf.constant. tf.int32s.
        Validation set labels.
    Dictionary mapping str → int.
        Maps each char to its position in the vocabulary.
    '''
    expressions, char2ind_map = make_addition_expressions(N, max_operand_digits, seed)
    
    x_int, y_int = make_addition_samples_and_labels(expressions, char2ind_map)
    
    x_train, y_train, x_val, y_val = make_train_val_split(x_int, y_int, val_prop)
    
    return x_train, y_train, x_val, y_val, char2ind_map

def make_multiplication_expressions(N, max_operand_digits=2, seed=1):
    '''Generates arithmetic expressions using the multiplication operator (*) and two operands.'''
    import numpy as np

    # Largest operand (e.g. 99 if max_operand_digits=2)
    max_operand = int('9' * max_operand_digits)

    # Compute maximum length of any product string
    max_product_len = len(str(max_operand * max_operand))

    # Total expression length: operand1 + operator + operand2 + '=' + product + '.' 
    expr_len = max_operand_digits + 1 + max_operand_digits + 1 + max_product_len + 1

    # RNG for reproducibility
    rng = np.random.default_rng(seed)
    operands = rng.integers(0, max_operand + 1, size=(N, 2))
    products = operands[:, 0] * operands[:, 1]

    # Build vocab mapping
    char2ind_map = {str(d): d for d in range(10)}
    char2ind_map['*'] = 10
    char2ind_map['='] = 11
    char2ind_map['.'] = 12
    char2ind_map['#'] = 13

    expressions = []
    for a, b, p in zip(operands[:,0], operands[:,1], products):
        expr = f'{a}*{b}={p}.'
        # pad with '#' so all have length expr_len
        expr = expr.ljust(expr_len, '#')
        expressions.append(list(expr))

    return expressions, char2ind_map


def get_multiplication_dataset(N, 
                               max_operand_digits=2, 
                               seed=1, 
                               val_prop=0.1):
    '''Generates and preprocesses a multiplication dataset for next-char prediction.

    Returns:
    --------
    x_train: tf.Tensor (int32) shape=(train_size, seq_len)
    y_train: tf.Tensor (int32) shape=(train_size, seq_len)
    x_val:   tf.Tensor (int32) shape=(val_size,   seq_len)
    y_val:   tf.Tensor (int32) shape=(val_size,   seq_len)
    char2ind_map: dict mapping each char to its integer code.
    '''
    # 1) build the raw multiplication expressions & vocab
    expressions, char2ind_map = make_multiplication_expressions(
        N, max_operand_digits, seed
    )

    # 2) reuse the addition-sample helper to turn chars → ints & split samples/labels
    x_int, y_int = make_addition_samples_and_labels(expressions, char2ind_map)

    # 3) split into train/validation sets
    x_train, y_train, x_val, y_val = make_train_val_split(
        x_int, y_int, val_prop
    )

    return x_train, y_train, x_val, y_val, char2ind_map

def make_blank_expressions(N, max_operand_digits=2, seed=1, operation = '+'):
    '''Generates arithmetic expressions using the multiplication operator (*) and two operands.'''
    import numpy as np

    # Largest operand (e.g. 99 if max_operand_digits=2)
    max_operand = int('9' * max_operand_digits)

    # Compute maximum length of any product string
    if operation == '+':
        max_result = max_operand + max_operand
    elif operation == '*':
        max_result = max_operand * max_operand
    elif operation == '-':
        max_result = max_operand - 0
    elif operation == '/':
        max_result = max_operand  # max_operand // 1
    elif operation == '%':
        max_result = max_operand  # max_operand % 1
    else:
        print(f'UNKNOWN operation! {operation}')
    max_result_len = len(str(max_result))

    # Total expression length: operand1 + operator + operand2 + '=' + product + '.' 
    expr_len = max_operand_digits + 1 + max_operand_digits + 1 + max_result_len + 1

    # RNG for reproducibility
    rng = np.random.default_rng(seed)
    operands = rng.integers(0, max_operand + 1, size=(N, 2))
    if operation == '+':
        results  = operands[:, 0] + operands[:, 1]
    elif operation == '*':
        results  = operands[:, 0] * operands[:, 1]
    elif operation == '-':
        results  = operands[:, 0] - operands[:, 1]
    elif operation == '/':
        # prevent divide-by-zero
        b_vals = operands[:, 1].copy()
        b_vals[b_vals == 0] = 1  # replace 0s with 1
        results = operands[:, 0] // b_vals
    elif operation == '%':
        results  = operands[:, 0] % operands[:, 1]


    # Build vocab mapping
    char2ind_map = {str(d): d for d in range(10)}
    char2ind_map[operation] = 10
    char2ind_map['='] = 11
    char2ind_map['.'] = 12
    char2ind_map['#'] = 13

    expressions = []
    for a, b, p in zip(operands[:,0], operands[:,1], results ):
        expr = f'{a}{operation}{b}={p}.'
        # pad with '#' so all have length expr_len
        expr = expr.ljust(expr_len, '#')
        expressions.append(list(expr))

    return expressions, char2ind_map

def get_blank_dataset(N, operation='+', max_operand_digits=2, seed=1, val_prop=0.1):
    # 1) build the raw operations expressions & vocab
    expressions, char2ind_map = make_blank_expressions(N, max_operand_digits, seed, operation=operation)

    # 2) reuse the addition-sample helper to turn chars → ints & split samples/labels
    x_int, y_int = make_addition_samples_and_labels(expressions, char2ind_map)

    # 3) split into train/validation sets
    x_train, y_train, x_val, y_val = make_train_val_split(
        x_int, y_int, val_prop
    )

    return x_train, y_train, x_val, y_val, char2ind_map


def make_reduction_expressions(N, max_operand_digits=2, seed=1, digit = 0):
    '''Generates arithmetic expressions using the multiplication operator (*) and two operands.'''
    import numpy as np

    # Largest operand (e.g. 99 if max_operand_digits=2)
    max_operand = int('9' * max_operand_digits)

    # Compute maximum length of any product string
    max_result_len = 1

    # Total expression length: operand1 + 'd' + digit + '=' + product + '.'
    expr_len = max_operand_digits + 1 + 1 + 1 + 1 + 1  # = max_operand_digits + 5


    # RNG for reproducibility
    rng = np.random.default_rng(seed)
    operands = rng.integers(0, max_operand + 1, size=(N, 1))

    results = (operands[:, 0] // (10**digit)) % 10

    # Build vocab mapping
    char2ind_map = {str(d): d for d in range(10)}
    char2ind_map['d'] = 10
    char2ind_map['='] = 11
    char2ind_map['.'] = 12
    char2ind_map['#'] = 13

    expressions = []
    for a, p in zip(operands[:,0], results ):
        expr = f'{a}d{str(digit)}={p}.'
        # pad with '#' so all have length expr_len
        expr = expr.ljust(expr_len, '#')
        expressions.append(list(expr))

    return expressions, char2ind_map

def get_reduction_dataset(N, digit=0, max_operand_digits=2, seed=1, val_prop=0.1):
    # 1) build the raw operations expressions & vocab
    expressions, char2ind_map = make_reduction_expressions(N, max_operand_digits, seed, digit=digit)

    # 2) reuse the addition-sample helper to turn chars → ints & split samples/labels
    x_int, y_int = make_addition_samples_and_labels(expressions, char2ind_map)

    # 3) split into train/validation sets
    x_train, y_train, x_val, y_val = make_train_val_split(
        x_int, y_int, val_prop
    )

    return x_train, y_train, x_val, y_val, char2ind_map


def make_reduction_expressions_multi(N_per, max_operand_digits=2, seed=1):
    '''Generates arithmetic expressions using the multiplication operator (*) and two operands.'''
    import numpy as np

    # Largest operand (e.g. 99 if max_operand_digits=2)
    max_operand = int('9' * max_operand_digits)
    # Compute maximum length of any product string
    max_result_len = 1
    # Total expression length: operand1 + 'd' + digit + '=' + product + '.'
    expr_len = max_operand_digits + 1 + 1 + 1 + 1 + 1  # = max_operand_digits + 5
    # RNG for reproducibility
    rng = np.random.default_rng(seed)


    expressions = []

    # Build vocab mapping
    char2ind_map = {str(n): n for n in range(10)}
    char2ind_map['d'] = 10
    char2ind_map['='] = 11
    char2ind_map['.'] = 12
    char2ind_map['#'] = 13

    for d in range(max_operand_digits):
        operands = rng.integers(0, max_operand + 1, size=(N_per, 1))
        results = (operands[:, 0] // (10**d)) % 10

        for a, p in zip(operands[:,0], results ):
            expr = f'{a}d{str(d)}={p}.'
            # pad with '#' so all have length expr_len
            expr = expr.ljust(expr_len, '#')
            expressions.append(list(expr))

    rng.shuffle(expressions)
    return expressions, char2ind_map

def get_reduction_dataset_multi(N_per, max_operand_digits=2, seed=1, val_prop=0.1):
    # 1) build the raw operations expressions & vocab
    expressions, char2ind_map = make_reduction_expressions_multi(N_per, max_operand_digits, seed)

    # 2) reuse the addition-sample helper to turn chars → ints & split samples/labels
    x_int, y_int = make_addition_samples_and_labels(expressions, char2ind_map)

    # 3) split into train/validation sets
    x_train, y_train, x_val, y_val = make_train_val_split(
        x_int, y_int, val_prop
    )

    return x_train, y_train, x_val, y_val, char2ind_map