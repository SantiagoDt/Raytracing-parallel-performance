#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
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
    int block_size = 64;

    int blocks_x = (w + block_size - 1) / block_size;
    int blocks_y = (h + block_size - 1) / block_size;
    int total_blocks = blocks_x * blocks_y;
    std::vector<int> my_blocks;
    for (int block_id = rank; block_id < total_blocks; block_id += size) {
        my_blocks.push_back(block_id);
    }
    if (rank == 0) {
        printf("MPI Ray Tracing BLOQUES con %d procesos\n", size);
        printf("Tamaño de imagen: %dx%d, muestras: %d\n", w, h, ns);
    }
    int total_pixels = 0;
    for (int block_id : my_blocks) {
        int bx = block_id % blocks_x;
        int by = block_id / blocks_x;
        int block_start_x = bx * block_size;
        int block_start_y = by * block_size;
        int block_end_x = (block_start_x + block_size < w) ? block_start_x + block_size : w;
        int block_end_y = (block_start_y + block_size < h) ? block_start_y + block_size : h;
        int pixels_in_block = (block_end_x - block_start_x) * (block_end_y - block_start_y);
        total_pixels += pixels_in_block;
    }
    int local_size = sizeof(unsigned char) * total_pixels * 3;
    unsigned char* local_data = (unsigned char*)calloc(local_size, 1);
    double start_time_total = 0.0, end_time_total = 0.0;
    if (rank == 0) {
        start_time_total = MPI_Wtime();
    }
    int current_offset = 0;
    for (int block_id : my_blocks) {
        int bx = block_id % blocks_x;
        int by = block_id / blocks_x;
        int block_start_x = bx * block_size;
        int block_start_y = by * block_size;
        int block_end_x = (block_start_x + block_size < w) ? block_start_x + block_size : w;
        int block_end_y = (block_start_y + block_size < h) ? block_start_y + block_size : h;
        int block_w = block_end_x - block_start_x;
        int block_h = block_end_y - block_start_y;
        rayTracingCPU(local_data + current_offset, w, h, ns, block_start_x, block_start_y, block_end_x, block_end_y);
        current_offset += block_w * block_h * 3;
    }
    if (rank == 0) {
        unsigned char* full_image = (unsigned char*)calloc(w * h * 3, 1);
        current_offset = 0;
        for (int block_id : my_blocks) {
            int bx = block_id % blocks_x;
            int by = block_id / blocks_x;
            int block_start_x = bx * block_size;
            int block_start_y = by * block_size;
            int block_end_x = (block_start_x + block_size < w) ? block_start_x + block_size : w;
            int block_end_y = (block_start_y + block_size < h) ? block_start_y + block_size : h;
            int block_w = block_end_x - block_start_x;
            int block_h = block_end_y - block_start_y;
            for (int y = 0; y < block_h; y++) {
                int src_offset = current_offset + y * block_w * 3;
                int dst_offset = ((block_start_y + y) * w + block_start_x) * 3;
                memcpy(full_image + dst_offset, local_data + src_offset, block_w * 3);
            }
            current_offset += block_w * block_h * 3;
        }
        std::vector<MPI_Request> requests;
        std::vector<unsigned char*> recv_buffers;
        for (int p = 1; p < size; p++) {
            std::vector<int> p_blocks;
            for (int block_id = p; block_id < total_blocks; block_id += size) {
                p_blocks.push_back(block_id);
            }
            for (int block_id : p_blocks) {
                int bx = block_id % blocks_x;
                int by = block_id / blocks_x;
                int block_start_x = bx * block_size;
                int block_start_y = by * block_size;
                int block_end_x = (block_start_x + block_size < w) ? block_start_x + block_size : w;
                int block_end_y = (block_start_y + block_size < h) ? block_start_y + block_size : h;
                int block_w = block_end_x - block_start_x;
                int block_h = block_end_y - block_start_y;
                int block_pixels = block_w * block_h * 3;
                unsigned char* block_buffer = (unsigned char*)malloc(block_pixels);
                recv_buffers.push_back(block_buffer);
                MPI_Request req;
                MPI_Irecv(block_buffer, block_pixels, MPI_UNSIGNED_CHAR, p, block_id, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        int buffer_idx = 0;
        for (int p = 1; p < size; p++) {
            std::vector<int> p_blocks;
            for (int block_id = p; block_id < total_blocks; block_id += size) {
                p_blocks.push_back(block_id);
            }
            for (int block_id : p_blocks) {
                int bx = block_id % blocks_x;
                int by = block_id / blocks_x;
                int block_start_x = bx * block_size;
                int block_start_y = by * block_size;
                int block_end_x = (block_start_x + block_size < w) ? block_start_x + block_size : w;
                int block_end_y = (block_start_y + block_size < h) ? block_start_y + block_size : h;
                int block_w = block_end_x - block_start_x;
                int block_h = block_end_y - block_start_y;
                for (int y = 0; y < block_h; y++) {
                    int src_offset = y * block_w * 3;
                    int dst_offset = ((block_start_y + y) * w + block_start_x) * 3;
                    memcpy(full_image + dst_offset, recv_buffers[buffer_idx] + src_offset, block_w * 3);
                }
                buffer_idx++;
            }
        }
        end_time_total = MPI_Wtime();
        writeBMP("output/mpi_chunk_64.bmp", full_image, w, h);
        printf("Tiempo total: %.3f segundos\n", end_time_total - start_time_total);
        printf("Imagen guardada como output/mpi_chunk_64.bmp\n");
        for (auto buffer : recv_buffers) {
            free(buffer);
        }
        free(full_image);
    } else {
        current_offset = 0;
        for (int block_id : my_blocks) {
            int bx = block_id % blocks_x;
            int by = block_id / blocks_x;
            int block_start_x = bx * block_size;
            int block_start_y = by * block_size;
            int block_end_x = (block_start_x + block_size < w) ? block_start_x + block_size : w;
            int block_end_y = (block_start_y + block_size < h) ? block_start_y + block_size : h;
            int block_w = block_end_x - block_start_x;
            int block_h = block_end_y - block_start_y;
            int block_pixels = block_w * block_h * 3;
            MPI_Send(local_data + current_offset, block_pixels, MPI_UNSIGNED_CHAR, 0, block_id, MPI_COMM_WORLD);
            current_offset += block_pixels;
        }
    }
    free(local_data);
    MPI_Finalize();
    return 0;
}
