class ShapeException(Exception):
    pass


def shape(shape_item):
    """ shape should take a vector or matrix and return a tuple with the
        number of rows (for a vector) or the number of rows and columns
        (for a matrix.)
    """
    if isinstance(shape_item[0], int):
        # this is a vector
        shape_length = (len(shape_item),)
    else:
        # this is a matrix
        shape_length = (len(shape_item), len(shape_item[0]))

    return shape_length


def vector_add(v1, v2):
    """
       [a b]  + [c d]  = [a+c b+d]

       Matrix + Matrix = Matrix
    """
    if shape(v1) != shape(v2):
        raise ShapeException("You cannot add two vectors that are not"
                             " the same length!")

    r_vector = [v1[x]+v2[x] for x in range(len(v1))]

    return r_vector


def vector_sub(v1, v2):
    """
        [a b]  - [c d]  = [a-c b-d]

        Matrix + Matrix = Matrix
    """
    if shape(v1) != shape(v2):
        raise ShapeException("You cannot subtract two vectors that are not"
                             " the same length!")

    r_vector = [v1[x]-v2[x] for x in range(len(v1))]

    return r_vector


def vector_sum(*vectors):
    """
        vector_sum can take any number of vectors and add them together.
    """
    cnt = 1
    # for all the vectors passed in - check their size is equal to the first
    while cnt <= len(vectors)-1:
        if shape(vectors[0]) != shape(vectors[cnt]):
            raise ShapeException("You cannot add two vectors that are not"
                                 " the same length!")
        cnt += 1

    return [
        sum([vector[i] for vector in vectors])
        for i in range(len(vectors[0]))
    ]


def dot(v1, v2):
    """
        dot([a b], [c d])   = a * c + b * d

        dot(Vector, Vector) = Scalar
    """
    if shape(v1) != shape(v2):
        raise ShapeException("You cannot add two vectors that are not"
                             " the same length!")

    dot_sum = sum([v1[x]*v2[x] for x in range(len(v1))])

    return dot_sum


def vector_multiply(v1, multiple):
    """
        [a b]  *  Z     = [a*Z b*Z]

        Vector * Scalar = Vector
    """
    r_vector = [v1[x]*multiple for x in range(len(v1))]

    return r_vector


def vector_mean(*vectors):
    """
        mean([a b], [c d]) = [mean(a, c) mean(b, d)]

        mean(Vector)       = Vector
    """
    length = len(vectors)
    m_scalar = 1/length
    r_vector = vector_sum(*vectors)
    r_vector = vector_multiply(r_vector, m_scalar)

    return r_vector


def magnitude(v1):
    """
        magnitude([a b])  = sqrt(a^2 + b^2)

        magnitude(Vector) = Scalar
    """
    mag_num = sum([v1[x]**2 for x in range(len(v1))])
    mag_num **= 0.5

    return mag_num


def matrix_row(matrix, row):
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    return matrix[row]


def matrix_col(matrix, col):
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    # first transpose the matrix
    matrix = [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    return matrix[col]


def matrix_scalar_multiply(matrix, scalar):
    """
        [[a b]   *  Z   =   [[a*Z b*Z]
         [c d]]              [c*Z d*Z]]

        Matrix * Scalar = Matrix
    """
    return [[each*scalar for each in matrix[i]]
            for i in range(len(matrix))]


def matrix_vector_multiply(matrix, vector):
    """
        [[a b]   *  [x   =   [a*x+b*y
         [c d]       y]       c*x+d*y
         [e f]                e*x+f*y]

        Matrix * Vector = Vector
    """
    if shape(matrix_row(matrix, 0)) != shape(vector):
        raise ShapeException("You cannot multiply these two shapes!")

    return [sum(matrix[i][j]*vector[j] for j in range(len(vector)))
            for i in range(len(matrix))]


def matrix_matrix_multiply(m1, m2):
    """
        [[a b]   *  [[w x]   =   [[a*w+b*y a*x+b*z]
        [c d]       [y z]]       [c*w+d*y c*x+d*z]
        [e f]                    [e*w+f*y e*x+f*z]]

        Matrix * Matrix = Matrix
    """
    if shape(matrix_row(m1, 0)) != shape(matrix_row(m2, 0)) or \
       shape(matrix_col(m1, 0)) != shape(matrix_col(m2, 0)):
        raise ShapeException("You cannot multiply these two shapes!")

    return [[sum(m1[i][k]*m2[k][j] for k in range(len(m2)))
            for j in range(len(m2[0]))]
            for i in range(len(m1))]


if __name__ == '__main__':

    a = [1, 4, 3, 9]
    b = [2, 2, 2, 5]
    c = [3, 1, 5, 9]
    d = [3, 2, 8, 7]

    A = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    B = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    C = [[1, 2],
         [2, 1],
         [1, 2]]
    D = [[1, 2, 3],
         [3, 2, 1],
         [1, 1, 1]]

    print(matrix_vector_multiply(A, [2, 5, 4]))
    print(matrix_scalar_multiply(B, 1))
    print(matrix_scalar_multiply(C, 3))
    print(matrix_matrix_multiply(A, B))
    print(matrix_matrix_multiply(A, D))
    print(shape(a))
    print(shape(C))
    print(vector_add(a, b))
    print(vector_sub(a, b))
    print(vector_sum(a, b, c))
    print(dot(a, b))
    print(vector_multiply(a, 10))
    print(vector_mean(a, b, c))
    print(magnitude(a))
    print(matrix_row(A, 0))
    print(matrix_col(B, 2))
