#include <cstdio>
#include <ctime>

#include "Camera.cuh"
#include "Ray.cuh"
#include "Vec3.cuh"

#include "Crystalline.cuh"
#include "Diffuse.cuh"
#include "Metallic.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"

#include "random.cuh"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '"
                  << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int w, int h, curandState *rand_state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if ((idx >= w) || (idy >= h))
        return;

    int pixel_index = idy * w + idx;
    curand_init(42, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render_init_rows(int w, int h, curandState *rand_state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    int idy = blockIdx.y;                            
    if (idx >= w || idy >= h)
        return;
    int pixel_index = idy * w + idx;
    curand_init(42, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Vec3 *fb, int w, int h, int ns, Camera **cam, SceneGPU *world, curandState *rand_state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if ((idx >= w) || (idy >= h))
        return;

    int pixel_index = idy * w + idx;
    curandState local_rand_state = rand_state[pixel_index];
    Vec3 col(0, 0, 0);

    for (int s = 0; s < ns; s++) {
        float u = float(idx + curand_uniform(&local_rand_state)) / float(w);
        float v = float(idy + curand_uniform(&local_rand_state)) / float(h);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += world->getSceneColor(r, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void render_rows(Vec3 *fb, int w, int h, int ns, Camera **cam, SceneGPU *world, curandState *rand_state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    int idy = blockIdx.y;                           

    if (idx >= w || idy >= h)
        return;

    int pixel_index = idy * w + idx;
    curandState local_rand_state = rand_state[pixel_index];
    Vec3 col(0, 0, 0);

    for (int s = 0; s < ns; s++) {
        float u = float(idx + curand_uniform(&local_rand_state)) / float(w);
        float v = float(idy + curand_uniform(&local_rand_state)) / float(h);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += world->getSceneColor(r, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Object **aux, int numobjects, SceneGPU *d_world, Camera **d_camera, int nx, int ny,
                             curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_world->setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
        d_world->setInfColor(Vec3(1.0f, 1.0f, 1.0f));
        d_world->setList(aux, numobjects);
        d_world->add(new Object(new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f), new Diffuse(Vec3(0.5f, 0.5f, 0.5f))));
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                Vec3 center(a + RND, 0.2f, b + RND);
                if (choose_mat < 0.8f) {
                    d_world->add(
                        new Object(new Sphere(center, 0.2f), new Diffuse(Vec3(RND * RND, RND * RND, RND * RND))));
                } else if (choose_mat < 0.95f) {
                    d_world->add(new Object(
                        new Sphere(center, 0.2f),
                        new Metallic(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND)));
                } else {
                    d_world->add(new Object(new Sphere(center, 0.2f), new Crystalline(1.5f)));
                }
            }
        }
        d_world->add(new Object(new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f), new Crystalline(1.5f)));
        d_world->add(new Object(new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f), new Diffuse(Vec3(0.4f, 0.2f, 0.1f))));
        d_world->add(new Object(new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f), new Metallic(Vec3(0.7f, 0.6f, 0.5f), 0.0f)));

        *rand_state = local_rand_state;

        Vec3 lookfrom(13.0f, 2.0f, 3.0f);
        Vec3 lookat(0.0f, 0.0f, 0.0f);
        float dist_to_focus = 10.0f; //(lookfrom - lookat).length();
        float aperture = 0.1f;
        *d_camera =
            new Camera(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 20.0, float(nx) / float(ny), aperture, dist_to_focus);
    }
}

void rayTracingGPU(Vec3 *img, int w, int h, int ns = 1) {
    int tx = 32;
    int ty = 8;
    clock_t start, stop;
    double timer_seconds;

    std::cerr << "Rendering a " << w << "x" << h << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = w * h;
    size_t fb_size = num_pixels * sizeof(Vec3);

    start = clock();
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Object **aux;
    int numobjects = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&aux, numobjects * sizeof(Object *)));
    SceneGPU *d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Scene)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    create_world<<<1, 1>>>(aux, numobjects, d_world, d_camera, w, h, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Loading took " << timer_seconds << " seconds.\n";

    start = clock();
    dim3 blocks(w / tx + 1, h / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(w, h, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, w, h, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Rendering took " << timer_seconds << " seconds.\n";

    start = clock();
    for (int i = h - 1; i >= 0; i--) {
        for (int j = 0; j < w; j++) {
            size_t pixel_index = i * w + j;
            img[pixel_index] = fb[pixel_index];
        }
    }
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Copy took " << timer_seconds << " seconds.\n";

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(aux));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
}

void rayTracingGPU_ByRows(Vec3 *img, int w, int h, int ns = 1) {
    clock_t start, stop;
    double timer_seconds;    std::cerr << "Rendering BY ROWS a " << w << "x" << h << " image with " << ns << " samples per pixel.\n";
    const int THREADS_PER_BLOCK = 256;
    int blocks_per_row = (w + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::cerr << "Configuration: " << blocks_per_row << " x " << h << " blocks with up to " << THREADS_PER_BLOCK << " threads per block.\n";

    int num_pixels = w * h;
    size_t fb_size = num_pixels * sizeof(Vec3);

    start = clock();
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Object **aux;
    int numobjects = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&aux, numobjects * sizeof(Object *)));
    SceneGPU *d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Scene)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    create_world<<<1, 1>>>(aux, numobjects, d_world, d_camera, w, h, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Loading took " << timer_seconds << " seconds.\n";    start = clock();
    dim3 blocks((w + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, h);  
    dim3 threads(THREADS_PER_BLOCK, 1);                              
    
    render_init_rows<<<blocks, threads>>>(w, h, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render_rows<<<blocks, threads>>>(fb, w, h, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Rendering BY ROWS took " << timer_seconds << " seconds.\n";

    start = clock();
    for (int i = h - 1; i >= 0; i--) {
        for (int j = 0; j < w; j++) {
            size_t pixel_index = i * w + j;
            img[pixel_index] = fb[pixel_index];
        }
    }
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Copy took " << timer_seconds << " seconds.\n";

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(aux));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
}

__global__ void render_init_columns(int w, int h, curandState *rand_state) {
    int idy = threadIdx.x + blockIdx.y * blockDim.x;  
    int idx = blockIdx.x;                        

    if (idy >= h || idx >= w)
        return;

    int pixel_index = idy * w + idx;
    curand_init(42, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render_columns(Vec3 *fb, int w, int h, int ns, Camera **cam, SceneGPU *world, curandState *rand_state) {
    int idy = threadIdx.x + blockIdx.y * blockDim.x;
    int idx = blockIdx.x;                            

    if (idy >= h || idx >= w)
        return;

    int pixel_index = idy * w + idx;
    curandState local_rand_state = rand_state[pixel_index];
    Vec3 col(0, 0, 0);

    for (int s = 0; s < ns; s++) {
        float u = float(idx + curand_uniform(&local_rand_state)) / float(w);
        float v = float(idy + curand_uniform(&local_rand_state)) / float(h);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += world->getSceneColor(r, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

void rayTracingGPU_ByColumns(Vec3 *img, int w, int h, int ns = 1) {
    clock_t start, stop;
    double timer_seconds;

    std::cerr << "Rendering BY COLUMNS a " << w << "x" << h << " image with " << ns << " samples per pixel.\n";
    const int THREADS_PER_BLOCK = 256;
    int blocks_per_column = (h + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::cerr << "Configuration: " << w << " x " << blocks_per_column << " blocks with up to " << THREADS_PER_BLOCK << " threads per block.\n";

    int num_pixels = w * h;
    size_t fb_size = num_pixels * sizeof(Vec3);

    start = clock();
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Object **aux;
    int numobjects = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&aux, numobjects * sizeof(Object *)));
    SceneGPU *d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Scene)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    create_world<<<1, 1>>>(aux, numobjects, d_world, d_camera, w, h, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Loading took " << timer_seconds << " seconds.\n";

    start = clock();
    dim3 blocks(w, (h + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);  // w columnas × múltiples bloques por columna
    dim3 threads(THREADS_PER_BLOCK, 1);                               // 256 threads por bloque
    
    render_init_columns<<<blocks, threads>>>(w, h, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render_columns<<<blocks, threads>>>(fb, w, h, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Rendering BY COLUMNS took " << timer_seconds << " seconds.\n";

    start = clock();
    for (int i = h - 1; i >= 0; i--) {
        for (int j = 0; j < w; j++) {
            size_t pixel_index = i * w + j;
            img[pixel_index] = fb[pixel_index];
        }
    }
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Copy took " << timer_seconds << " seconds.\n";

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(aux));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
}

__global__ void render_init_3d(int w, int h, int numImages, curandState *rand_state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = blockIdx.z;

    if ((idx >= w) || (idy >= h) || (idz >= numImages))
        return;

    int pixel_index = idz * w * h + idy * w + idx;
    curand_init(42 + idz * 1000, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render_3d(Vec3 *fb, int w, int h, int numImages, int ns, Camera **cam, SceneGPU *world, curandState *rand_state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = blockIdx.z;

    if ((idx >= w) || (idy >= h) || (idz >= numImages))
        return;

    int pixel_index = idz * w * h + idy * w + idx;
    curandState local_rand_state = rand_state[pixel_index];
    Vec3 col(0, 0, 0);

    for (int s = 0; s < ns; s++) {
        float time_offset = idz * 0.1f; // Cada imagen es 0.1 segundos después
        float u = float(idx + curand_uniform(&local_rand_state)) / float(w);
        float v = float(idy + curand_uniform(&local_rand_state)) / float(h);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += world->getSceneColor(r, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}


void rayTracingGPU_MultiImage(Vec3 *img, int w, int h, int numImages, int ns = 1) {
    clock_t start, stop;
    double timer_seconds;

    std::cerr << "Rendering MULTI-IMAGE " << numImages << " images of " << w << "x" << h << " with " << ns << " samples per pixel.\n";
    std::cerr << "Configuration: 3D grid with " << numImages << " images in parallel.\n";

    int num_pixels_per_image = w * h;
    int total_pixels = num_pixels_per_image * numImages;
    size_t fb_size = total_pixels * sizeof(Vec3);

    start = clock();
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, total_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Object **aux;
    int numobjects = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&aux, numobjects * sizeof(Object *)));
    SceneGPU *d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Scene)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    create_world<<<1, 1>>>(aux, numobjects, d_world, d_camera, w, h, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Loading took " << timer_seconds << " seconds.\n";

    start = clock();
    dim3 threads(16, 16, 1);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y, numImages);
    
    render_init_3d<<<blocks, threads>>>(w, h, numImages, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render_3d<<<blocks, threads>>>(fb, w, h, numImages, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Rendering MULTI-IMAGE took " << timer_seconds << " seconds.\n";    start = clock();
    for (int imgIdx = 0; imgIdx < numImages; imgIdx++) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                size_t src_pixel_index = imgIdx * w * h + i * w + j;
                size_t dst_pixel_index = imgIdx * w * h + i * w + j;
                img[dst_pixel_index] = fb[src_pixel_index];
            }
        }
    }
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Copy took " << timer_seconds << " seconds.\n";

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(aux));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
}
