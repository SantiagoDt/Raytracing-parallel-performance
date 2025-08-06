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

    int w = 1200;
    int h = 800;
    int ns = 10; 
    int numImages = 3; 
    clock_t start, stop;
    double timer_seconds;

    printf("Resolución: %dx%d, imágenes: %d, muestras: %d\n", w, h, numImages, ns);

    size_t size = sizeof(unsigned char) * w * h * 3;
    unsigned char *data = (unsigned char *)malloc(size);

    Vec3 *multiImg;
    size_t multiSize = w * h * numImages * sizeof(Vec3);
    cudaMallocManaged((void **)&multiImg, multiSize);

    start = clock();
    rayTracingGPU_MultiImage(multiImg, w, h, numImages, ns);
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Tiempo total: %.3f segundos\n", timer_seconds);

    for (int imgIdx = 0; imgIdx < numImages; imgIdx++) {
        for (int i = h - 1; i >= 0; i--) {
            for (int j = 0; j < w; j++) {
                size_t src_idx = imgIdx * w * h + i * w + j;
                size_t dst_idx = i * w + j;
                data[dst_idx * 3 + 0] = char(255.99 * multiImg[src_idx].b());
                data[dst_idx * 3 + 1] = char(255.99 * multiImg[src_idx].g());
                data[dst_idx * 3 + 2] = char(255.99 * multiImg[src_idx].r());
            }
        }
        char filename[50];
        sprintf(filename, "imgGPU-multi-%02d.bmp", imgIdx);
        writeBMP(filename, data, w, h);
        printf("Imagen guardada como %s\n", filename);
    }

    cudaFree(multiImg);
    free(data);
    cudaDeviceReset();
    return 0;
}
