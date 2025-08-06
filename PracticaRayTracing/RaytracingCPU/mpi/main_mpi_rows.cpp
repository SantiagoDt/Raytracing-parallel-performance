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

    int w = 600;
    int h = 400;  
    int ns = 10;
    
    // División equitativa de filas entre procesos
    int base_rows = h / size;       
    int extra_rows = h % size;       
    int my_rows = base_rows + (rank < extra_rows ? 1 : 0); 
    int my_start_row = rank * base_rows + (rank < extra_rows ? rank : extra_rows); 
    int my_end_row = my_start_row + my_rows;
    
    int local_size = sizeof(unsigned char) * w * my_rows * 3;
    unsigned char* local_data = (unsigned char*)calloc(local_size, 1);
    

    double start_time_total = 0.0, end_time_total = 0.0;
    if (rank == 0) {
        start_time_total = MPI_Wtime();
        printf("MPI Ray Tracing con %d procesos\n", size);
        printf("Tamaño de imagen: %dx%d, muestras: %d\n", w, h, ns);
        printf("Filas base: %d, Filas extra: %d\n", base_rows, extra_rows);
    }
    
    double start_time = MPI_Wtime();
    
    rayTracingCPU(local_data, w, h, ns, 0, my_start_row, w, my_end_row);
    
    double end_time = MPI_Wtime();

    double local_time = end_time - start_time;
    
    // Encontrar el tiempo máximo entre todos los procesos
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {

        unsigned char* full_image = (unsigned char*)malloc(sizeof(unsigned char) * w * h * 3);
        memcpy(full_image, local_data, local_size);
        MPI_Request* requests = (MPI_Request*)malloc((size-1) * sizeof(MPI_Request));
        MPI_Status* statuses = (MPI_Status*)malloc((size-1) * sizeof(MPI_Status));
        for (int p = 1; p < size; p++) {
            int p_rows = base_rows + (p < extra_rows ? 1 : 0);
            int p_start_row = p * base_rows + (p < extra_rows ? p : extra_rows);
            int p_size = w * p_rows * 3;
            MPI_Irecv(full_image + (p_start_row * w * 3), p_size, MPI_UNSIGNED_CHAR, 
                     p, 0, MPI_COMM_WORLD, &requests[p-1]);
        }
        MPI_Waitall(size-1, requests, statuses);
        writeBMP("output/mpi_result.bmp", full_image, w, h);
        end_time_total = MPI_Wtime();
        printf("Tiempo total: %.3f segundos\n", end_time_total - start_time_total);
        printf("Imagen guardada como output/mpi_result.bmp\n");
        free(requests);
        free(statuses);
        free(full_image);
    } else {
        MPI_Send(local_data, local_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    free(local_data);
    MPI_Finalize();
    return 0;
}
