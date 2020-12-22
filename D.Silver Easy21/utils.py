import random

def zeros_3d(x_len, y_len, z_len):
    arr = []

    for _ in range(x_len):
        x_ls = []
        for _ in range(y_len):
            x_ls.append([0] * z_len)

        arr.append(x_ls)
        
    return arr

def zeros_2d(x_len, y_len):
    arr = []

    for _ in range(x_len):
        arr.append([0] * y_len)

    return arr
    
def mse_3d(solution_1, solution_2):
    mse = 0.0

    for i in range(10):
        for j in range(21):
            for k in range(2):
                error = solution_1[i][j][k] - solution_2[i][j][k]
                mse  += error * error

    return mse / (10 * 21 * 2)

def dot_product(vec1, vec2):
    val = 0

    assert len(vec1) == len(vec2)

    for i in range(len(vec1)):
        val += vec1[i] * vec2[i]

    return val

def random_vec(n):
    vec = []

    for _ in range(n):
        vec.append(random.random())
    
    return vec