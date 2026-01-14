#include <cstdint>

#define N 8

extern "C" void Sort(unsigned char input[N], unsigned char output[N]) {
    unsigned char temp[N];

    for (unsigned int i = 0; i < N; i++) {
        temp[i] = input[i];
    } 

    for (unsigned int i = N - 1; i > 0; i--) {
        unsigned char high = temp[0];
        unsigned char low = 0;

        for (unsigned int j = 1; j < N; j++) {
            if (j <= i) {
                if (temp[j] > high) {
                    low = high;
                    high = temp[j];
                } else {
                    low = temp[j];
                }
            } else {
	        low = temp[j - 1];
	    }

            temp[j - 1] = low;
        }

        temp[i] = high;
    }

    for (unsigned int i = 0; i < N; i++) {
        output[i] = temp[i];
    }
}
