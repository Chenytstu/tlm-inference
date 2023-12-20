#include "LinearBeaver/linear-beaver.hpp"
#include "utils/emp-tool.h"
#include "utils/utils.h"
#include <iostream>
#include <random>
using namespace std;
using namespace sci;

size_t dim1 = 512, dim2 = 512, dim3 = 512;

uint64_t *matmul(uint64_t *X, uint64_t *Y, size_t dim1, size_t dim2, size_t dim3) {
    size_t i, j, k;
    uint64_t *Z = new uint64_t[dim1 * dim3];
    for (i = 0; i < dim1; i++) {
        for (j = 0; j < dim3; j++) {
            Z[i * dim3 + j] = 0;
            for (k = 0; k < dim2; k++) {
                Z[i * dim3 + j] += (X[i * dim2 + k] * Y[k * dim3 + j]);
            }
        }
    }
    return Z;
}

void test_matmul(uint64_t *X, uint64_t *Y, size_t dim1, size_t dim2, size_t dim3) {
    INIT_TIMER;
    START_TIMER;
    auto Z = matmul(X, Y, dim1, dim2, dim3);
    STOP_TIMER("matmul (plaintext)");
}

int main(int argc, char **argv) {
    ArgMapping amap;
    amap.arg("dim1", dim1, "dim1");
    amap.arg("dim2", dim2, "dim2");
    amap.arg("dim3", dim3, "dim3");
    amap.parse(argc, argv);

    // prepare data
    uint64_t *rand_X = new uint64_t[dim1 * dim2];
    uint64_t *rand_Y = new uint64_t[dim2 * dim3];

    // generate test data
    PRG128 prg;
    prg.random_data(rand_X, sizeof(uint64_t) * dim1 * dim2);
    prg.random_data(rand_Y, sizeof(uint64_t) * dim2 * dim3);
    for (size_t i = 0; i < dim1 * dim2; i++) {
        rand_X[i] %= 256;
    }
    for (size_t i = 0; i < dim2 * dim3; i++) {
        rand_Y[i] %= 256;
    }

    cout << "**********************************\n";
    cout << "Test matrix mul\n";
    cout << "**********************************\n";
    printf("matrix size: x(%ld*%ld), y(%ld*%ld)\n", dim1, dim2, dim2, dim3);
    test_matmul(rand_X, rand_Y, dim1, dim2, dim3);
    delete[] rand_X;
    delete[] rand_Y;
    cout << "Test matrix mul end, data cleaned";
    cout << "\n**********************************\n";
    return 0;
}