import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    if p == 0.0:
        mask = np.ones_like(x)
        output = x
    elif p == 1.0:
        mask = np.zeros_like(x)
        output = np.zeros_like(x)
    else:
        if rng == None:
            mask_b = np.random.binomial(1, 1 - p, size=x.shape)
        else:
            mask_b = rng.binomial (1, 1 - p, size = x.shape)

        output = x * mask_b / (1 - p)
        mask = mask_b / (1 - p)
    return output, mask
