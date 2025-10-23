
import math
from torch import Tensor
import torch
import torch.nn.functional as functional

def band_exclusion_mask(m: int, n: int, d1: int, d2: int, *, device=None) -> torch.Tensor:
    """
    Returns an m x n boolean tensor where entries above the d2-th diagonal
    or below the d1-th diagonal are True; others (within [d1, d2]) are False.

    Examples:
        d1=0, d2=0  -> ~I (True everywhere except main diagonal)
        d1=-1, d2=1 -> False band of width 3 around diagonal
    """
    
    mask = torch.ones((m, n), dtype=torch.bool, device=device)
    mask.triu_(d2 + 1)                       # True above d2
    lower_band = torch.ones_like(mask).tril_(d1 - 1)  # True below d1
    mask |= lower_band                       # Combine regions
    
    return mask

def attention_chunk(queries: Tensor, keys: Tensor, values: Tensor, causal_buffer: Tensor):
    attn = queries @ keys.transpose(-2, -1) / math.sqrt(keys.size(-1))
    attn = attn.masked_fill(causal_buffer, float("-inf"))
    attn = functional.softmax(attn, dim=-1)
    outputs = attn @ values
    return outputs

def attention(queries: Tensor, keys: Tensor, values: Tensor, window_size: int = None, num_interleaves: int = 2):
    
    """
    Efficient implementation of causal, cache-conditioned Grouped Query Attention (GQA) where each 
    token can "see" the previous window_size tokens (including itself).
    
    Time complexity is O(n) with respect to the sequence length assuming a set
    window_size. 
    
    To explain, I'll illustrate using a toy example of a single batch with
    window_size=3, 10 queries, and 11 key/value pairs (1 extra kv pair
    indicates there is 1 kv pair from the cache). The attention mask would in
    this case be:
    
    [[1 1 0 0 0 0 0 0 0 0 0]
     [1 1 1 0 0 0 0 0 0 0 0]
     [0 1 1 1 0 0 0 0 0 0 0]
     [0 0 1 1 1 0 0 0 0 0 0]
     [0 0 0 1 1 1 0 0 0 0 0]
     [0 0 0 0 1 1 1 0 0 0 0]
     [0 0 0 0 0 1 1 1 0 0 0]
     [0 0 0 0 0 0 1 1 1 0 0]
     [0 0 0 0 0 0 0 1 1 1 0]
     [0 0 0 0 0 0 0 0 1 1 1]]
     
    Where each row corresponds to a query, and each column to a kv pair. Normal
    attention implementations would require O(n^2) space and flops to compute then
    use this mask for large sequences, however that becomes prohibitively expensive.
    We can make a more efficient implementation by noticing that only (q, kv) pairs
    along the window's diagonals need to be computed. This can be done efficiently
    by computing the results along long diagonals through several chunks. For this,
    let's assume the chunks each have a height of 2. To illustrate this, I've marked
    the entries corresponding to each chunk with separate letters (a, b, c, d, and z).
        
    [[z z z 0 0 0 0 0 0 0 0]
     [z z z 0 0 0 0 0 0 0 0]
     [0 a a a a 0 0 0 0 0 0]
     [0 a a a a 0 0 0 0 0 0]
     [0 0 0 b b b b 0 0 0 0]
     [0 0 0 b b b b 0 0 0 0]
     [0 0 0 0 0 c c c c 0 0]
     [0 0 0 0 0 c c c c 0 0]
     [0 0 0 0 0 0 0 d d d d]
     [0 0 0 0 0 0 0 d d d d]]
     
    I used "z" to mark the first chunk since it is different from the others. In the
    following code, this is refered to as the "dense" chunk. "dense" because it will
    usually (but not always) represent a fully connected (lower left triangle matrix)
    set of tokens. The other chunks (marked with a, b, c, and d in this example) are
    refered to as "sparse" since they always have some trivial entries other than those
    above the 0th diagonal.
    
    To reach an O(n) solution, we can simply compute the results from each of these
    chunks separately, then concatenate them at the end. However, a more efficient
    implementation can be reached by packing the computation of different chunks into
    the same set of matrix operations.
    
    To compute the results for any given chunk, we need 4 things:
    - a list of queries
    - a list of keys
    - a list of values
    - an attention mask
    
    The task of selecting the appropriate queries, keys, and values for each chunk is
    not trivial. We can lessen this burden by taking advantage of the way we contructed
    these chunks. Note that chunks a and c do not share any kv pairs. That means the key
    lists for chunks a and c are adjacently located in the full key list. 
    
    Illustration of needed queries and key / value indexes for chunks a and c:
           [  a's  |  c's  ]
        [[z z z 0 0 0 0 0 0 0 0]
         [z z z 0 0 0 0 0 0 0 0]
    ‾‾‾  [0 a a a a 0 0 0 0 0 0]
    _a_  [0 a a a a 0 0 0 0 0 0]
         [0 0 0 b b b b 0 0 0 0]
         [0 0 0 b b b b 0 0 0 0]
    ‾‾‾  [0 0 0 0 0 c c c c 0 0]
    _c_  [0 0 0 0 0 c c c c 0 0]
         [0 0 0 0 0 0 0 d d d d]
         [0 0 0 0 0 0 0 d d d d]]
         
    Keys / Values List: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                           [    a's   ][    c's   ]
    
    Queries List: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                       [ a's ]     [ c's ]
    
    To find the the key / value lists, we can simply select that range of keys and then
    reshape them to create a (2, 4) shaped tensor of keys for chunks a and c. Finding the
    query lists are slightly more complicated, since they are not located adjacent to each 
    other in the queries list, but rather alternate with those of the other chunks. Still,
    pytorch supports a slicing tensors in this way.
    
    Note that in this example our chunks are constructed so that you can divide them into two
    "interleaves" ({a, c} and {b, d}). You can also construct them such that they are divided
    into any number of interleaves, controlled by the parameter "num_interleaves". The chunk
    dimensions are then automatically calculated according to the window_size and num_interleaves
    although not all inputs are compatible (for example it's impossible to create valid chunks
    for window_size=3, num_interleaves=3). By default, num_interleaves is set to two, which is
    compatible with any window_size.
    
    Variables are named according to this 2d graphical understanding. "width" and "height" denote
    the dimensions of the above illustrated matrix. "chunk_height" and "chunk_width" thus respectively
    describe the number of queries and keys associated with a single sparse chunk.
    
    """
    
    device = queries.device
    
    ## Ingest and assert input shapes
    B, H_q, height, D_qk = queries.shape
    B, H_kv, width, D_qk = keys.shape
    B, H_kv, width, D_v = values.shape
    cache_size = width - height

    assert(queries.size(0) == keys.size(0))
    assert(queries.size(3) == keys.size(3))
    assert(keys.shape[:3] == values.shape[:3])
    assert(cache_size >= 0)
    assert(H_q % H_kv == 0)
    
    """ Calculate chunk dimensions """
    # assume unlimited window_size if none is specified
    window_size = window_size or width
    chunk_height = (window_size - 1) // (num_interleaves - 1)
    chunk_width = window_size + chunk_height - 1
    assert(num_interleaves * chunk_height == chunk_width)
    
    # num_chunks does not include the dense chunk
    num_chunks_per_interleave = max(0, (height - window_size) // (num_interleaves * chunk_height))
    num_chunks = num_chunks_per_interleave * num_interleaves
    
    # split the queries into two sections: those which will be computed by
    # the sparse chunks, and others which will be done by the dense chunk
    sparse_height = num_chunks * chunk_height
    dense_height = height - sparse_height
    
    """ Repeat keys and values for GQA """
    if H_kv != H_q:
        rate = H_q // H_kv
        keys = keys.repeat(1, rate, 1, 1)
        values = values.repeat(1, rate, 1, 1)
    
    """ Calculate dense chunk """
    dense_output = None
    if dense_height > 0:
        dense_left = max(0, cache_size - window_size)
        dense_right = cache_size + dense_height
        dense_width = dense_right - dense_left
        
        dense_queries = queries[:, :, :dense_height, :]
        dense_keys = keys[:, :, dense_left:dense_right, :]
        dense_values = values[:, :, dense_left:dense_right, :]
        dense_causal = band_exclusion_mask(dense_height, dense_width, cache_size-window_size-dense_left+1, cache_size-dense_left, device=device)
        
        dense_output = attention_chunk(dense_queries, dense_keys, dense_values, dense_causal)
    
    """ Calculate sparse chunks """
    sparse_output = None
    if sparse_height > 0:
        
        sparse_queries = queries[:, :, dense_height:, :]
        sparse_queries = sparse_queries.reshape(B, H_q, sparse_height // chunk_height, chunk_height, D_qk)
        
        sparse_left = width - (num_chunks - 1) * chunk_height - chunk_width
        sparse_causal = band_exclusion_mask(chunk_height, chunk_width, chunk_width-chunk_height-window_size+1, chunk_width-chunk_height, device=device)
        
        for i in range(num_interleaves):
            
            interleave_left = sparse_left + i * chunk_height
            interleave_width = num_chunks_per_interleave * chunk_width
            interleave_end = interleave_left + interleave_width
            
            interleave_keys = keys[:, :, interleave_left:interleave_end, :]
            interleave_values = values[:, :, interleave_left:interleave_end, :]
            
            interleave_keys = interleave_keys.reshape(B, H_q, num_chunks_per_interleave, chunk_width, D_qk)
            interleave_values = interleave_values.reshape(B, H_q, num_chunks_per_interleave, chunk_width, D_v)
            interleave_queries = sparse_queries[:, :, i::num_interleaves, :, :]

            interleave_output = attention_chunk(interleave_queries, interleave_keys, interleave_values, sparse_causal)

            assert(interleave_output.shape == (B, H_q, num_chunks_per_interleave, chunk_height, D_v))

            if sparse_output == None:
                sparse_output = interleave_output
            else:
                sparse_output = torch.concat((sparse_output, interleave_output), dim=-2)
        
        sparse_output = sparse_output.reshape(B, H_q, sparse_height, D_v)
        
    """ Concatenate dense and sparse results """
    if sparse_output == None:
        return dense_output
    if dense_output == None:
        return sparse_output
    return torch.concat((dense_output, sparse_output), dim=2)
