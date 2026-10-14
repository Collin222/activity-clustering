def _check_square(A):
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("matrix must be non-empty and square")
    return n
 
 
def det(A, tol=1e-12):
    """Determinant of square matrix A. O(n^3)."""
    n = _check_square(A)
    M = [list(row) for row in A]  # copy; don't mutate input
    sign = 1
    result = 1
    for col in range(n):
        # partial pivot: largest |value| in this column at/below the diagonal
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) <= tol:
            return 0
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
            sign = -sign
        p = M[col][col]
        result *= p
        for r in range(col + 1, n):
            f = M[r][col] / p
            if f:
                Mr, Mc = M[r], M[col]
                for c in range(col + 1, n):
                    Mr[c] -= f * Mc[c]
    return sign * result
 
 
def inverse(A, tol=1e-12):
    """Inverse of square matrix A via Gauss-Jordan. Raises ValueError if singular."""
    n = _check_square(A)
    # augment [A | I]
    one = A[0][0] ** 0 if not isinstance(A[0][0], float) else 1.0  # keeps Fraction type
    zero = one - one
    M = [list(row) + [one if i == j else zero for j in range(n)] for i, row in enumerate(A)]
    w = 2 * n
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) <= tol:
            raise ValueError("matrix is singular (or nearly so)")
        M[col], M[pivot] = M[pivot], M[col]
        # normalize pivot row
        p = M[col][col]
        Mc = M[col]
        for c in range(col, w):
            Mc[c] /= p
        # eliminate this column from every other row
        for r in range(n):
            if r != col:
                f = M[r][col]
                if f:
                    Mr = M[r]
                    for c in range(col, w):
                        Mr[c] -= f * Mc[c]
    return [row[n:] for row in M]
