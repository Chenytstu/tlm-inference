#ifndef LINEAR_BEAVER_H
#define LINEAR_BEAVER_H
#include "LinearBeaver/beaver.h"
#include "utils/io_pack.h"
#include <numeric>
#include <vector>

inline uint64_t sum(uint64_t *v, size_t size) {
    return std::accumulate(v, v + size, 0);
}

inline uint64_t *T(uint64_t *M, size_t row, size_t column) {
    size_t i, j;
    uint64_t *result = new uint64_t[row * column];
    for (i = 0; i < row; i++) {
        for (j = 0; j < column; j++) {
            result[j * row + i] = M[i * column + j];
        }
    }
    return result;
}

uint64_t *add(uint64_t *v1, uint64_t *v2, size_t size) {
    auto result = new uint64_t[size];
    for (size_t i = 0; i < size; i++) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

uint64_t *mul(uint64_t *v1, uint64_t *v2, size_t size) {
    auto result = new uint64_t[size];
    for (size_t i = 0; i < size; i++) {
        result[i] = v1[i] * v2[i];
    }
    return result;
}

uint64_t vec_dot(uint64_t *v1, uint64_t *v2, size_t size) {
    uint64_t result = 0;
    for (size_t i = 0; i < size; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

class LinearBeaver {
public:
    int party;
    sci::IOPack *iopack;

    LinearBeaver(int party_, sci::IOPack *iopack_) : iopack(iopack_) {
        assert(party_ == sci::PUBLIC || party_ == sci::ALICE || party_ == sci::BOB);
        this->party = party_;
    };

    ~LinearBeaver(){};

    uint64_t *hadamard_product(uint64_t *X, uint64_t *Y, uint64_t *Z, size_t size,
                               uint64_t *a, uint64_t *b, uint64_t *c) {
        uint64_t *s = new uint64_t[size];
        uint64_t *t = new uint64_t[size];
        size_t i;
        for (i = 0; i < size; i++) {
            s[i] = X[i] - a[i];
            t[i] = Y[i] - b[i];
        }
        iopack->io->send_data(s, sizeof(uint64_t) * size);
        iopack->io->send_data(t, sizeof(uint64_t) * size);

        uint64_t *s1 = new uint64_t[size];
        uint64_t *t1 = new uint64_t[size];
        iopack->io->recv_data(s1, sizeof(uint64_t) * size);
        iopack->io->recv_data(t1, sizeof(uint64_t) * size);

        for (i = 0; i < size; i++) {
            s[i] += s1[i];
            t[i] += t1[i];
            s1[i] = s[i];
            t1[i] = t[i];
        }

        for (i = 0; i < size; i++) {
            s[i] = s[i] * b[i]; // bs
            t[i] = t[i] * a[i]; // at
        }
        auto st = mul(s1, t1, size); // st
        for (i = 0; i < size; i++) {
            Z[i] = s[i] + t[i] + c[i]; // at + bs + c;
        }
        delete[] t;
        delete[] s1;
        delete[] t1;
        if (party == sci::ALICE) {
            for (i = 0; i < size; i++) {
                Z[i] += st[i]; // at + bs + c + st;
            }
        }
        delete[] st;
        delete[] s;
    };

    void matrix_multiplication(uint64_t *X, uint64_t *Y, uint64_t *Z,
                               size_t dim1, size_t dim2, size_t dim3,
                               uint64_t *a, uint64_t *b, uint64_t c) {
        /*
        X: dim1 * dim2
        Y: dim2 * dim3
        Z: dim1 * dim3
        a: dim2 * 1
        b: dim2 * 1
        c: 1 * 1
        c1 + c2 = (a1 + a2) \cdot (b1 + b2)
        */
        size_t i, j;
        uint64_t *s = new uint64_t[dim1 * dim2];
        for (i = 0; i < dim1 * dim2; i++) {
            s[i] = X[i];
        }
        uint64_t *t = T(Y, dim2, dim3);
        for (i = 0; i < dim2; i++) {
            for (j = 0; j < dim1; j++) {
                s[j * dim2 + i] -= a[i];
            }
            for (j = 0; j < dim3; j++) {
                t[j * dim2 + i] -= b[i];
            }
        }

        iopack->io->send_data(s, sizeof(uint64_t) * dim1 * dim2);
        uint64_t *s1 = new uint64_t[dim1 * dim2];
        iopack->io->recv_data(s1, sizeof(uint64_t) * dim1 * dim2);

        iopack->io->send_data(t, sizeof(uint64_t) * dim3 * dim2);
        uint64_t *t1 = new uint64_t[dim3 * dim2];
        iopack->io->recv_data(t1, sizeof(uint64_t) * dim3 * dim2);

        for (i = 0; i < dim1 * dim2; i++) {
            s[i] += s1[i];
        }
        for (i = 0; i < dim3 * dim2; i++) {
            t[i] += t1[i];
        }

        for (i = 0; i < dim1; i++) {
            for (j = 0; j < dim3; j++) {
                Z[i * dim3 + j] = c * dim2 + vec_dot(s + i * dim2, b, dim2) + vec_dot(t + j * dim2, a, dim2);
                if (party == sci::ALICE) {
                    Z[i * dim3 + j] += vec_dot(s + i * dim2, t + j * dim2, dim2);
                }
            }
        }
        delete[] s;
        delete[] t;
        delete[] s1;
        delete[] t1;
    }
};
#endif
