#include <cstdint>
extern "C" int64_t Dot64(const int16_t (&arg_0)[64], const int16_t (&arg_1)[64]) {
    int64_t sum = 0;
    for (int i = 0; i < 64; ++i) {
        int16_t product = arg_0[i] * arg_1[i];      // Product is truncated to 16 bits
        sum += product;
    }
    return sum;
}
