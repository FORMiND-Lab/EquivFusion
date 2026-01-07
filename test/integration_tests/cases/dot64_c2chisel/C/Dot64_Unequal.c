#include <stdint.h>

#define N 64

int64_t Dot64(int16_t arg_0[N], int16_t arg_1[N]) {
    int64_t sum = 0;
    for (int i = 0; i < N; i++) {
        int16_t product = arg_0[i] * arg_1[i];      // Result is truncated to 16 bits
        sum += (int64_t)product;
    }

    return sum;
}
