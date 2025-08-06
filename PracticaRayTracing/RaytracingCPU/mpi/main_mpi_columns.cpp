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
    

    int base_cols = w / size;     
    int extra_cols = w % size;  
    int my_cols = base_cols + (rank < extra_cols ? 1 : 0); 
    int my_start_col = rank * base_cols + (rank < extra_cols ? rank : extra_cols);
    int my_end_col = my_start_col + my_cols; 
    
    int local_size = sizeof(unsigned char) * my_cols * h * 3;
    unsigned char* local_data = (unsigned char*)calloc(local_size, 1);
    
    double start_time_total = 0.0, end_time_total = 0.0;
    if (rank == 0) {
        start_time_total = MPI_Wtime();
        printf("MPI Ray Tracing con %d procesos (POR COLUMNAS)\n", size);
        printf("Tamaño de imagen: %dx%d, muestras: %d\n", w, h, ns);
        printf("Columnas base: %d, Columnas extra: %d\n", base_cols, extra_cols);
    }
    
    double start_time = MPI_Wtime();
    
    rayTracingCPU(local_data, w, h, ns, my_start_col, 0, my_end_col, h);
    
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        unsigned char* full_image = (unsigned char*)malloc(sizeof(unsigned char) * w * h * 3);
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < my_cols; col++) {
                int local_idx = (row * my_cols + col) * 3;
                int global_idx = (row * w + col) * 3;
                full_image[global_idx + 0] = local_data[local_idx + 0];
                full_image[global_idx + 1] = local_data[local_idx + 1];
                full_image[global_idx + 2] = local_data[local_idx + 2];
            }
        }
        MPI_Request* requests = (MPI_Request*)malloc((size-1) * sizeof(MPI_Request));
        MPI_Status* statuses = (MPI_Status*)malloc((size-1) * sizeof(MPI_Status));
        unsigned char** temp_buffers = (unsigned char**)malloc((size-1) * sizeof(unsigned char*));
        for (int p = 1; p < size; p++) {
            int p_cols = base_cols + (p < extra_cols ? 1 : 0);
            int p_size = p_cols * h * 3;
            temp_buffers[p-1] = (unsigned char*)malloc(p_size);
            MPI_Irecv(temp_buffers[p-1], p_size, MPI_UNSIGNED_CHAR, 
                     p, 0, MPI_COMM_WORLD, &requests[p-1]);
        }
        MPI_Waitall(size-1, requests, statuses);
        for (int p = 1; p < size; p++) {
            int p_cols = base_cols + (p < extra_cols ? 1 : 0);
            int p_start_col = p * base_cols + (p < extra_cols ? p : extra_cols);
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < p_cols; col++) {
                    int local_idx = (row * p_cols + col) * 3;
                    int global_idx = (row * w + (p_start_col + col)) * 3;
                    full_image[global_idx + 0] = temp_buffers[p-1][local_idx + 0];
                    full_image[global_idx + 1] = temp_buffers[p-1][local_idx + 1];
                    full_image[global_idx + 2] = temp_buffers[p-1][local_idx + 2];
                }
            }
            free(temp_buffers[p-1]);
        }
        writeBMP("output/mpi_result_columns.bmp", full_image, w, h);
        end_time_total = MPI_Wtime();
        printf("Tiempo total: %.3f segundos\n", end_time_total - start_time_total);
        printf("Imagen guardada como output/mpi_result_columns.bmp\n");
        free(requests);
        free(statuses);
        free(temp_buffers);
        free(full_image);
    } else {
        MPI_Send(local_data, local_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    free(local_data);
    MPI_Finalize();
    return 0;
}
