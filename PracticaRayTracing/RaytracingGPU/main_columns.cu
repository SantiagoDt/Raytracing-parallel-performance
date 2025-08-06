#include <cstdio>
#include <cstdlib>

#include "raytracing.cuh"

#include "Camera.cuh"
#include "Crystalline.cuh"
#include "Diffuse.cuh"
#include "Metallic.cuh"
#include "Object.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"
#include "Vec3.cuh"

#include "random.cuh"
#include "utils.cuh"

int main() {
    int w = 1920;
    int h = 1080;
    int ns = 100;
    clock_t start, stop;
    double timer_seconds;

    printf("Resolución: %dx%d, muestras: %d\n", w, h, ns);

    size_t size = sizeof(unsigned char) * w * h * 3;
    unsigned char *data = (unsigned char *)malloc(size);

    Vec3 *img;
    size_t isize = w * h * sizeof(Vec3);
    cudaMallocManaged((void **)&img, isize);

    start = clock();
    rayTracingGPU_ByColumns(img, w, h, ns);
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Tiempo total: %.3f segundos\n", timer_seconds);

    for (int i = h - 1; i >= 0; i--) {
        for (int j = 0; j < w; j++) {
            size_t idx = i * w + j;
            data[idx * 3 + 0] = char(255.99 * img[idx].b());
            data[idx * 3 + 1] = char(255.99 * img[idx].g());
            data[idx * 3 + 2] = char(255.99 * img[idx].r());
        }
    }
    writeBMP("imgGPU-columns.bmp", data, w, h);
    printf("Imagen guardada como imgGPU-columns.bmp\n");

    cudaFree(img);
    free(data);
    cudaDeviceReset();
    return 0;
}
