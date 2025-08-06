#include "Vec3.cuh"

void rayTracingGPU(Vec3 *img, int w, int h, int ns = 10);
void rayTracingGPU_ByRows(Vec3 *img, int w, int h, int ns = 10);
void rayTracingGPU_ByColumns(Vec3 *img, int w, int h, int ns = 10);
void rayTracingGPU_MultiImage(Vec3 *img, int w, int h, int numImages, int ns = 10);
