#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

#include "../implementations/RayTracingImplementations.hpp"
#include "../implementations/utils.h"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);  
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

    int w = 256;
    int h = 256;
    int ns = 10;
    int total_frames = size; 

    if (rank == 0) {
        printf("MPI Ray Tracing con %d procesos (POR FRAMES)\n", size);
        printf("Tamaño de imagen: %dx%d, muestras: %d\n", w, h, ns);
        printf("Generando %d frames (1 por proceso)\n", total_frames);
    }

    // Reservar memoria para la imagen completa de este frame
    int image_size = sizeof(unsigned char) * w * h * 3;
    unsigned char* frame_data = (unsigned char*)calloc(image_size, 1);

    double start_time_total = 0.0, end_time_total = 0.0;
    if (rank == 0) {
        start_time_total = MPI_Wtime();
    }

    rayTracingCPU(frame_data, w, h, ns, 0, 0, w, h);

    if (rank == 0) {
        end_time_total = MPI_Wtime();
        printf("Tiempo total: %.3f segundos\n", end_time_total - start_time_total);
    }

    char filename[256];
    snprintf(filename, sizeof(filename), "output/mpi_frame_%02d.bmp", rank);
    writeBMP(filename, frame_data, w, h);
    printf("Proceso %d: Frame guardado como %s\n", rank, filename);

    free(frame_data);
    MPI_Finalize();
    return 0;
}
