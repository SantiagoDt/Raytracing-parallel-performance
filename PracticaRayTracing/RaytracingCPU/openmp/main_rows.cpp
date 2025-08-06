#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "../implementations/RayTracingImplementations.hpp"
#include "../implementations/utils.h"

int main() {


	int w = 1200;
	int h = 800;
	int ns = 50;

	int patch_x_size = w;
	int patch_y_size = h;
	int patch_x_idx = 1;
	int patch_y_idx = 1;

	int size = sizeof(unsigned char) * patch_x_size * patch_y_size * 3;
	unsigned char* data = (unsigned char*)calloc(size, 1);
	int patch_x_start = (patch_x_idx - 1) * patch_x_size;
	int patch_x_end = patch_x_idx * patch_x_size;
	int patch_y_start = (patch_y_idx - 1) * patch_y_size;
	int patch_y_end = patch_y_idx * patch_y_size;


	printf("Resolución: %dx%d, Samples: %d\n", w, h, ns);


	double t0, t1;
	t0 = omp_get_wtime();
	
	rayTracingCPU_OpenMP_Rows(data, w, h, ns, patch_x_start, patch_y_start, patch_x_end, patch_y_end);
	
	t1 = omp_get_wtime();
	double elapsed = t1 - t0;
	
	writeBMP("output/imgCPU_rows.bmp", data, patch_x_size, patch_y_size);

	printf("Tiempo de cómputo: %.3f segundos\n", elapsed);
	printf("Imagen guardada como: output/imgCPU_rows.bmp\n");


	free(data);
	return (0);
}
