#ifndef RAYTRACING_IMPLEMENTATIONS_HPP
#define RAYTRACING_IMPLEMENTATIONS_HPP

#include <string>
#include "Scene.h"

Scene loadObjectsFromFile(const std::string& filename);
Scene randomScene();
void rayTracingCPU(unsigned char* img, int w, int h, int ns = 10, int px = 0, int py = 0, int pw = -1, int ph = -1);

void rayTracingCPU_OpenMP_Rows(unsigned char* img, int w, int h, int ns = 10, int px = 0, int py = 0, int pw = -1, int ph = -1);
void rayTracingCPU_OpenMP_Columns(unsigned char* img, int w, int h, int ns = 10, int px = 0, int py = 0, int pw = -1, int ph = -1);
void rayTracingCPU_OpenMP_Blocks(unsigned char* img, int w, int h, int ns = 10, int px = 0, int py = 0, int pw = -1, int ph = -1);
void rayTracingCPU_OpenMP_Blocks_Manual(unsigned char* img, int w, int h, int ns, int px, int py, int pw, int ph);
void rayTracingCPU_OpenMP_Blocks_Fixed(unsigned char* img, int w, int h, int ns, int px, int py, int pw, int ph, int block_size = 32);

void rayTracingCPU_Frames(int w, int h, int ns, int num_frames);
void rayTracingCPU_Frames_OpenMP(int w, int h, int ns, int num_frames, int px = 0, int py = 0, int pw = -1, int ph = -1);

#endif