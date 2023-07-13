#include <stdint.h>

extern "C" {
	__global__ void kernel(float* out) {
		uint32_t n = threadIdx.x + blockIdx.x*blockDim.x;

		out[n] = atan2(static_cast<float>(n), 2.0f);
	}
}