#include <cstdint>

constexpr int  N = 64;

extern "C" int64_t Dot64(const int16_t (&arg_0)[N], const int16_t (&arg_1)[N]) {
    int64_t sum = 0;
    for (int i = 0; i < N; ++i) {
        int32_t product = arg_0[i] * arg_1[i];
        sum += product;
    }
    return sum;
}
