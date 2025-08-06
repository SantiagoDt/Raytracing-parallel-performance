#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <cmath>
#include <vector>

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

    // División equitativa de bloques rectangulares entre procesos
    int grid_cols = (int)sqrt(size);
    int grid_rows = (size + grid_cols - 1) / grid_cols;
    if (grid_cols * grid_rows > size) {
        grid_rows = size / grid_cols;
        if (grid_cols * grid_rows < size) grid_rows++;
    }
    int my_col = rank % grid_cols;
    int my_row = rank / grid_cols;
    if (rank >= grid_cols * grid_rows || my_row >= grid_rows) {
        MPI_Finalize();
        return 0;
    }
    int block_w = w / grid_cols;
    int block_h = h / grid_rows;
    int my_start_x = my_col * block_w;
    int my_end_x = (my_col == grid_cols - 1) ? w : (my_col + 1) * block_w;
    int my_start_y = my_row * block_h;
    int my_end_y = (my_row == grid_rows - 1) ? h : (my_row + 1) * block_h;
    int my_w = my_end_x - my_start_x;
    int my_h = my_end_y - my_start_y;
    int local_size = sizeof(unsigned char) * my_w * my_h * 3;
    unsigned char* local_data = (unsigned char*)calloc(local_size, 1);

    double start_time_total = 0.0, end_time_total = 0.0;
    if (rank == 0) {
        start_time_total = MPI_Wtime();
        printf("MPI Ray Tracing BLOQUES RECTANGULARES con %d procesos\n", size);
        printf("Tamaño de imagen: %dx%d, muestras: %d\n", w, h, ns);
        printf("Grid de procesos: %dx%d\n", grid_cols, grid_rows);
        printf("Tamaño base de bloque: %dx%d píxeles\n", block_w, block_h);
    }

    rayTracingCPU(local_data, w, h, ns, my_start_x, my_start_y, my_end_x, my_end_y);

    double max_time = 0.0;

    if (rank == 0) {
        unsigned char* full_image = (unsigned char*)calloc(w * h * 3, 1);
        for (int y = 0; y < my_h; y++) {
            int src_offset = y * my_w * 3;
            int dst_offset = ((my_start_y + y) * w + my_start_x) * 3;
            memcpy(full_image + dst_offset, local_data + src_offset, my_w * 3);
        }
        for (int p = 1; p < size && p < grid_cols * grid_rows; p++) {
            int p_col = p % grid_cols;
            int p_row = p / grid_cols;
            if (p_row >= grid_rows) continue;
            int p_start_x = p_col * block_w;
            int p_end_x = (p_col == grid_cols - 1) ? w : (p_col + 1) * block_w;
            int p_start_y = p_row * block_h;
            int p_end_y = (p_row == grid_rows - 1) ? h : (p_row + 1) * block_h;
            int p_w = p_end_x - p_start_x;
            int p_h = p_end_y - p_start_y;
            int p_size = p_w * p_h * 3;
            unsigned char* p_data = (unsigned char*)malloc(p_size);
            MPI_Recv(p_data, p_size, MPI_UNSIGNED_CHAR, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int y = 0; y < p_h; y++) {
                int src_offset = y * p_w * 3;
                int dst_offset = ((p_start_y + y) * w + p_start_x) * 3;
                memcpy(full_image + dst_offset, p_data + src_offset, p_w * 3);
            }
            free(p_data);
        }
        end_time_total = MPI_Wtime();
        writeBMP("output/mpi_static_chunk.bmp", full_image, w, h);
        printf("Tiempo total: %.3f segundos\n", end_time_total - start_time_total);
        printf("Imagen guardada como output/mpi_static_chunk.bmp\n");
        free(full_image);
    } else {
        MPI_Send(local_data, local_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    free(local_data);
    MPI_Finalize();
    return 0;
}
