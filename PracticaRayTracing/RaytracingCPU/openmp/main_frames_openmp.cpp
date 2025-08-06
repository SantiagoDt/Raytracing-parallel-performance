
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "../implementations/RayTracingImplementations.hpp"
#include "../implementations/utils.h"

int main() {

	int w = 1200;
	int h = 800;
	int ns = 10;
	int num_frames = 3; 

	int patch_x_size = w;
	int patch_y_size = h;
	int patch_x_idx = 1;
	int patch_y_idx = 1;

	int patch_x_start = (patch_x_idx - 1) * patch_x_size;
	int patch_x_end = patch_x_idx * patch_x_size;
	int patch_y_start = (patch_y_idx - 1) * patch_y_size;
	int patch_y_end = patch_y_idx * patch_y_size;

	printf("Resolución: %dx%d, Samples: %d, Frames: %d\n", w, h, ns, num_frames);

	double t0, t1;
	t0 = omp_get_wtime();
	
	rayTracingCPU_Frames_OpenMP(w, h, ns, num_frames, patch_x_start, patch_y_start, patch_x_end, patch_y_end);
	
	t1 = omp_get_wtime();
	double elapsed = t1 - t0;
	
	printf("Tiempo total de cómputo: %.3f segundos\n", elapsed);
	printf("Tiempo promedio por frame: %.3f segundos\n", elapsed / num_frames);
	printf("Frames generados: %d imágenes\n", num_frames);


	return (0);
}
