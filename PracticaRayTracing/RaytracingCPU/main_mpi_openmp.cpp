#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "RayTracingImplementations.hpp"
#include "utils.h"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);  
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

    int w = 1200;
    int h = 800;
    int ns = 10;
    int total_frames = size; 
    
    if (rank == 0) {
        printf("MPI + OpenMP Ray Tracing con %d procesos (POR FRAMES HÍBRIDO)\n", size);
        printf("Tamaño de imagen: %dx%d, muestras: %d\n", w, h, ns);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int image_size = sizeof(unsigned char) * w * h * 3;
    unsigned char* frame_data = (unsigned char*)calloc(image_size, 1);
    double start_time = MPI_Wtime();
    rayTracingCPU_OpenMP_Blocks_Fixed(frame_data, w, h, ns, 0, 0, w, h);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    char filename[256];
    snprintf(filename, sizeof(filename), "hybrid_frame_%02d.bmp", rank);
    writeBMP(filename, frame_data, w, h);
    printf("Proceso %d: Frame guardado como %s (tiempo: %.3f s)\n", rank, filename, local_time);
    if (rank == 0) {
        printf("Tiempo máximo entre todos los procesos: %.3f s\n", local_time);
    }
    free(frame_data);
    MPI_Finalize();
    return 0;
}
