#include <stdint.h>
#include <stdbool.h>

/*
void LZC(uint8_t data_in, uint8_t *lzc) {
    uint8_t count = 0;
    for (int j = 6; j >= 0; --j) { // 从bit6到bit0
        if ((data_in >> j) & 1) break;
        count++;
    }
    *lzc = count & 0x07;
}
*/

uint8_t LZC(uint8_t data_in) {
    uint8_t count = 0;
    bool skip = false;
    for (int i = 7; i >= 1; i--) {
    	if (!skip) {
	    if ((data_in >> i) & 1) {
	        skip = true;
	    } else {
	        count++;
	    }
	}
    }

    return count;
}

