import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = max(len(i) for i in seqs)
    
    for j in range(len(seqs)):
        while len(seqs[j]) < max_len:
            seqs[j].append(pad_value)
        while len(seqs[j])  > max_len :
            seqs[j].pop()    
    return seqs
    