import struct
import numpy as np

def read_states(path):
    """Read a states_N*.bin dump. Returns (N, X, occ, a_lambda, C)."""
    with open(path, 'rb') as f:
        magic, N, X, dim = struct.unpack('<4Q', f.read(32))
        if magic != 0x4C415547484C4E31:
            raise ValueError('not a Laughlin state dump')
        rec = np.dtype([('k', '<u8'), ('a', '<i8'), ('C', '<f8')])
        arr = np.frombuffer(f.read(), dtype=rec)

    nbits = N + X - 1
    keys = arr['k'].astype(np.uint64)
    shifts = np.arange(nbits, dtype=np.uint64)
    bits = ((keys[:, None] >> shifts) & np.uint64(1)).astype(np.int8)

    occ = np.zeros((len(arr), X), dtype=np.int16)
    orbital = np.cumsum(1 - bits, axis=1) - (1 - bits)
    r, c = np.nonzero(bits == 1)
    np.add.at(occ, (r, orbital[r, c]), 1)
    return N, X, occ, arr['a'], arr['C']

if __name__ == '__main__':
    import sys
    N, X, occ, a, C = read_states(sys.argv[1])
    print(f'N = {N}, orbitals = {X}, states = {len(a)}')
    print(f'particle number conserved : {bool((occ.sum(1) == N).all())}')
    Lz = (occ * np.arange(X)).sum(1)
    print(f'total Lz = {Lz[0]} (expected {N*(N-1)}), uniform: {bool((Lz == Lz[0]).all())}')
    print(f'largest |a_lambda| = {abs(a).max()}')
    print('\nfirst 5 states (occupation | a_lambda | C):')
    for i in range(min(5, len(a))):
        print(' ', ''.join(map(str, occ[i])), a[i], f'{C[i]:.6e}')
