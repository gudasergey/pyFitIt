import numpy as np
import math
from numba import jit
import time


def ihs (dim_num, n, d=5, seed=None):
    if seed is None: seed = int(time.time()*1e7)
    if seed == 0: seed = 74836149
    res = ihs_helper (dim_num, n, d, seed)
    return res.reshape(n,dim_num)


@jit(nopython=True)
def ihs_helper (dim_num, n, d, seed ):
    r8_huge = 1.0E+30
    seedArray = [seed]

    avail = np.empty((dim_num * n), dtype=np.int32)
    list = np.empty((d * n), dtype=np.int32)
    point = np.empty((dim_num * d * n), dtype=np.int32)
    x = np.empty((dim_num*n), dtype=np.int32)

    opt = np.float64(n) / pow(np.float64(n), (1.0 / np.float64(dim_num)))

    #
    #  Pick the first point.
    #
    for i in range(dim_num):
        x[i+(n-1)*dim_num] = i4_uniform_ab ( 1, n, seedArray )

    # for i in range(dim_num):
    #     x[i + (n - 1) * dim_num] = i4_uniform_ab(1, n, seedArray)

    #
    #  Initialize AVAIL,
    #  and set an entry in a random row of each column of AVAIL to N.
    #
    for i in range(n):
        avail[i * dim_num:(i + 1) * dim_num] = i + 1

    for i in range(dim_num):
        avail[i+(x[i+(n-1)*dim_num]-1)*dim_num] = n

    # for j in range(n):
    #     for i in range(dim_num):
    #         avail[i + j * dim_num] = j + 1
    #
    # for i in range(dim_num):
    #     avail[i + (x[i + (n - 1) * dim_num] - 1) * dim_num] = n

    #
    #  Main loop:
    #  Assign a value to X(1:M,COUNT) for COUNT = N-1 down to 2.
    #
    # dt1=0; dt2=0; dt3=0
    for count in range(n-1, 1, -1):
        #
        #  Generate valid points.
        #
        for i in range(dim_num):
            for k in range(d):
                for j in range(count):
                    list[j+k*count] = avail[i+j*dim_num]

            for k in range(count*d - 1, -1, -1):
                point_index = i4_uniform_ab(0, k, seedArray)
                point[i+k*dim_num] = list[point_index]
                list[point_index] = list[k]
        #
        #  For each candidate, determine the distance to all the
        #  points that have already been selected, and save the minimum value.
        #
        # t2 = time.time()
        min_all = r8_huge
        best = 0

        for k in range(d * count):
            min_can = r8_huge

            for j in range(count, n):
                dist = np.float64(0.0)
                for i in range(dim_num):
                    dist = dist + np.float64(point[i+k*dim_num] - x[i+j*dim_num]) * np.float64(point[i+k*dim_num] - x[i + j*dim_num])
                dist = math.sqrt(dist)

                if dist < min_can:
                    min_can = dist

            if abs(min_can - opt) < min_all:
                min_all = abs(min_can - opt)
                best = k

        for i in range(dim_num):
            x[i+(count-1)*dim_num] = point[i+best*dim_num]

        #
        #  Having chosen X(*,COUNT), update AVAIL.
        #
        # t3 = time.time()
        for i in range(dim_num):
            for j in range(n):
                if avail[i+j*dim_num] == x[i+(count-1)*dim_num]:
                    avail[i+j*dim_num] = avail[i+(count-1)*dim_num]
        # t4 = time.time()
        # dt1 += t2-t1; dt2 += t3-t2; dt3 = t4-t3;
    #
    #  For the last point, there's only one choice.
    #
    for i in range(dim_num):
        x[i+0*dim_num] = avail[i+0*dim_num]
    # print(dt1,dt2,dt3)
    return x


@jit(nopython=True)
def i4_uniform_ab ( a, b, seedArray ):
    i4_huge = 2147483647

    #assert seedArray[0] != 0, "I4_UNIFORM_AB - Fatal error! Input value of SEED = 0."
    #
    #  Guarantee A <= B.
    #
    if b < a:
        c = a
        a = b
        b = c

    k = int(seedArray[0] / 127773)

    seedArray[0] = 16807 * ( seedArray[0] - k * 127773 ) - k * 2836

    if seedArray[0] < 0:
        seedArray[0] = seedArray[0] + i4_huge

    r = seedArray[0] * 4.656612875E-10
    #
    #  Scale R to lie between A-0.5 and B+0.5.
    #
    r = ( 1.0 - r ) * (a - 0.5 ) + r * (b + 0.5)
    #
    #  Use rounding to convert R to an integer between A and B.
    #
    value = int(round(r))
    #
    #  Guarantee A <= VALUE <= B.
    #
    if value < a:
        value = a
    if b < value:
        value = b

    return value
