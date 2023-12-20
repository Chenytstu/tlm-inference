#ifndef BEAVER_H
#define BEAVER_H
#include "utils/utils.h"
#include <cassert>
#include <cstdint>

inline void add(uint64_t *v, uint64_t size, uint64_t &num) {
    for (size_t i = 0; i < size; i++) {
        v[i] = v[i] + num;
    }
}

inline void add(uint64_t &num, uint64_t *v, uint64_t size) {
    add(v, size, num);
}

inline void sub(uint64_t *v, uint64_t size, uint64_t num) {
    for (size_t i = 0; i < size; i++) {
        v[i] = v[i] - num;
    }
}

inline void sub(uint64_t &num, uint64_t *v, uint64_t size) {
    for (size_t i = 0; i < size; i++) {
        v[i] = num - v[i];
    }
}

inline void mul(uint64_t *v, uint64_t size, uint64_t &num) {
    for (size_t i = 0; i < size; i++) {
        v[i] = v[i] * num;
    }
}

inline void mul(uint64_t &num, uint64_t *v, uint64_t size) {
    mul(v, size, num);
}

uint64_t *beaver(int32_t bw) {
    PRG128 prg;
    uint64_t *bv = new uint64_t[6]; // a1, a2, b1, b2, c1, c2, (a1 + a2) * (b1 + b2) = (c1 + c2);
    size_t j;
    uint64_t mask1 = (bw >= 64 ? -1 : ((1ULL << bw * 2) - 1));
    uint64_t mask2 = (bw * 2 >= 64 ? -1 : ((1ULL << bw * 2) - 1));
    for (char i = 0; i < 4; i++) {
        prg.random_data(&bv[i], sizeof(uint64_t));
        bv[i] &= mask1;
    }
    auto result = (bv[0] + bv[1]) * (bv[2] + bv[3]);
    prg.random_data(&bv[4], sizeof(uint64_t));;
    bv[4] &= mask2;
    bv[5] = result - bv[4];
    return bv;
}
#endif