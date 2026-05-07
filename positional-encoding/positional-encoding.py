import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    """
    PE = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for j in range(0, d_model, 2):
            pair_idx = j // 2  
            div = base ** (2 * pair_idx / d_model)
            PE[pos, j] = np.sin(pos / div)
            if j + 1 < d_model:
                PE[pos, j + 1] = np.cos(pos / div)
    
    return PE

# Тест
