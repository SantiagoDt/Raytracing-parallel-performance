#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "implementations/RayTracingImplementations.hpp"
#include "implementations/utils.h"

int main() {

	int w = 1200;
	int h = 800;
	int ns = 10;
	int num_frames = 6;


	printf("Resolución: %dx%d, Samples: %d, Frames: %d\n", w, h, ns, num_frames);
	auto start = std::chrono::high_resolution_clock::now();
	rayTracingCPU_Frames(w, h, ns, num_frames);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	double elapsed = duration.count() / 1000.0;

	printf("Tiempo total de cómputo: %.3f segundos\n", elapsed);
	printf("Frames generados: %d imágenes\n", num_frames);


	return (0);
}
