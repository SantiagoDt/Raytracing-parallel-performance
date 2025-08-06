#include "RayTracingImplementations.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <omp.h>
#include "Camera.h"
#include "Object.h"
#include "Scene.h"
#include "Sphere.h"
#include "Diffuse.h"
#include "Metallic.h"
#include "Crystalline.h"
#include "random.h"
#include "utils.h"

Scene loadObjectsFromFile(const std::string& filename) {
	std::ifstream file(filename);
	std::string line;

	Scene list;

	if (file.is_open()) {
		while (std::getline(file, line)) {
			std::stringstream ss(line);
			std::string token;
			std::vector<std::string> tokens;

			while (ss >> token) {
				tokens.push_back(token);
			}

			if (tokens.empty()) continue; // L�nea vac�a

			// Esperamos al menos la palabra clave "Object"
			if (tokens[0] == "Object" && tokens.size() >= 12) { // M�nimo para Sphere y un material con 1 float
				// Parsear la esfera
				if (tokens[1] == "Sphere" && tokens[2] == "(" && tokens[7] == ")") {
					try {
						float sx = std::stof(tokens[3].substr(tokens[3].find('(') + 1, tokens[3].find(',') - tokens[3].find('(') - 1));
						float sy = std::stof(tokens[4].substr(0, tokens[4].find(',')));
						float sz = std::stof(tokens[5].substr(0, tokens[5].find(',')));
						float sr = std::stof(tokens[6]);

						// Parsear el material del �ltimo objeto creado

						if (tokens[8] == "Crystalline" && tokens[9] == "(" && tokens[11].back() == ')') {
							float ma = std::stof(tokens[10]);
							list.add(new Object(
								new Sphere(Vec3(sx, sy, sz), sr),
								new Crystalline(ma)
							));
							std::cout << "Crystaline" << sx << " " << sy << " " << sz << " " << sr << " " << ma << "\n";
						}
						else if (tokens[8] == "Metallic" && tokens.size() == 15 && tokens[9] == "(" && tokens[14] == ")") {
							float ma = std::stof(tokens[10].substr(tokens[10].find('(') + 1, tokens[10].find(',') - tokens[10].find('(') - 1));
							float mb = std::stof(tokens[11].substr(0, tokens[11].find(',')));
							float mc = std::stof(tokens[12].substr(0, tokens[12].find(',')));
							float mf = std::stof(tokens[13].substr(0, tokens[13].length() - 1));
							list.add(new Object(
								new Sphere(Vec3(sx, sy, sz), sr),
								new Metallic(Vec3(ma, mb, mc), mf)
							));
							std::cout << "Metallic" << sx << " " << sy << " " << sz << " " << sr << " " << ma << " " << mb << " " << mc << " " << mf << "\n";
						}
						else if (tokens[8] == "Diffuse" && tokens.size() == 14 && tokens[9] == "(" && tokens[13].back() == ')') {
							float ma = std::stof(tokens[10].substr(tokens[10].find('(') + 1, tokens[10].find(',') - tokens[10].find('(') - 1));
							float mb = std::stof(tokens[11].substr(0, tokens[11].find(',')));
							float mc = std::stof(tokens[12].substr(0, tokens[12].find(',')));
							list.add(new Object(
								new Sphere(Vec3(sx, sy, sz), sr),
								new Diffuse(Vec3(ma, mb, mc))
							));
							std::cout << "Diffuse" << sx << " " << sy << " " << sz << " " << sr << " " << ma << " " << mb << " " << mc << "\n";
						}
						else {
							std::cerr << "Error: Material desconocido o formato incorrecto en la l�nea: " << line << std::endl;
						}
					}
					catch (const std::invalid_argument& e) {
						std::cerr << "Error: Conversi�n inv�lida en la l�nea: " << line << " - " << e.what() << std::endl;
					}
					catch (const std::out_of_range& e) {
						std::cerr << "Error: Valor fuera de rango en la l�nea: " << line << " - " << e.what() << std::endl;
					}
				}
				else {
					std::cerr << "Error: Formato de esfera incorrecto en la l�nea: " << line << std::endl;
				}
			}
			else {
				std::cerr << "Error: Formato de objeto incorrecto en la l�nea: " << line << std::endl;
			}
		}
		file.close();
	}
	else {
		std::cerr << "Error: No se pudo abrir el archivo: " << filename << std::endl;
	}
	return list;
}

Scene randomScene() {
	int n = 500;
	Scene list;
	list.add(new Object(
		new Sphere(Vec3(0, -1000, 0), 1000),
		new Diffuse(Vec3(0.5, 0.5, 0.5))
	));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = Mirandom();
			Vec3 center(a + 0.9f * Mirandom(), 0.2f, b + 0.9f * Mirandom());
			if ((center - Vec3(4, 0.2f, 0)).length() > 0.9f) {
				if (choose_mat < 0.8f) {  // diffuse
					list.add(new Object(
						new Sphere(center, 0.2f),
						new Diffuse(Vec3(Mirandom() * Mirandom(),
							Mirandom() * Mirandom(),
							Mirandom() * Mirandom()))
					));
				}
				else if (choose_mat < 0.95f) { // metal
					list.add(new Object(
						new Sphere(center, 0.2f),
						new Metallic(Vec3(0.5f * (1 + Mirandom()),
							0.5f * (1 + Mirandom()),
							0.5f * (1 + Mirandom())),
							0.5f * Mirandom())
					));
				}
				else {  // glass
					list.add(new Object(
						new Sphere(center, 0.2f),
						new Crystalline(1.5f)
					));
				}
			}
		}
	}

	list.add(new Object(
		new Sphere(Vec3(0, 1, 0), 1.0),
		new Crystalline(1.5f)
	));
	list.add(new Object(
		new Sphere(Vec3(-4, 1, 0), 1.0f),
		new Diffuse(Vec3(0.4f, 0.2f, 0.1f))
	));
	list.add(new Object(
		new Sphere(Vec3(4, 1, 0), 1.0f),
		new Metallic(Vec3(0.7f, 0.6f, 0.5f), 0.0f)
	));

	return list;
}



void rayTracingCPU(unsigned char* img, int w, int h, int ns, int px, int py, int pw, int ph) {
	if (pw == -1) pw = w;
	if (ph == -1) ph = h;
	int patch_w = pw - px;
    // Semilla fija para que randomScene() sea igual en todos los procesos que llaman a rayTracingCPU
    //caso contrario cada bloque genera un bloque distinto al tener semillas distintas
    srand(42); 
	Scene world = randomScene();
	world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
	world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

	Vec3 lookfrom(13, 2, 3);
	Vec3 lookat(0, 0, 0);
	float dist_to_focus = 10.0;
	float aperture = 0.1f;

	Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);

	for (int j = 0; j < (ph - py); j++) {
		for (int i = 0; i < (pw - px); i++) {

			Vec3 col(0, 0, 0);
			for (int s = 0; s < ns; s++) {
				float u = float(i + px + Mirandom()) / float(w);
				float v = float(j + py + Mirandom()) / float(h);
				Ray r = cam.get_ray(u, v);
				col += world.getSceneColor(r);
			}
			col /= float(ns);
			col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

			img[(j * patch_w + i) * 3 + 2] = char(255.99 * col[0]);
			img[(j * patch_w + i) * 3 + 1] = char(255.99 * col[1]);
			img[(j * patch_w + i) * 3 + 0] = char(255.99 * col[2]);
		}
	}
}

void rayTracingCPU_OpenMP_Rows(unsigned char* img, int w, int h, int ns, int px, int py, int pw, int ph){
    if (pw == -1) pw = w;
    if (ph == -1) ph = h;
    int patch_w = pw - px;
    

    Scene world = randomScene();

    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);
    
    #pragma omp parallel for
    for (int j = 0; j < (ph - py); j++) {
        for (int i = 0; i < (pw - px); i++) {

            Vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++) {
                float u = float(i + px + Mirandom()) / float(w);
                float v = float(j + py + Mirandom()) / float(h);
                Ray r = cam.get_ray(u, v);
                col += world.getSceneColor(r);
            }
            col /= float(ns);
            col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            img[(j * patch_w + i) * 3 + 2] = char(255.99 * col[0]);
            img[(j * patch_w + i) * 3 + 1] = char(255.99 * col[1]);
            img[(j * patch_w + i) * 3 + 0] = char(255.99 * col[2]);
        }
    }
}

void rayTracingCPU_OpenMP_Columns(unsigned char* img, int w, int h, int ns, int px, int py, int pw, int ph){
    if (pw == -1) pw = w;
    if (ph == -1) ph = h;
    int patch_w = pw - px;
    Scene world = randomScene();

    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);
    
    for (int j = 0; j < (ph - py); j++) {
        #pragma omp parallel for 
        for (int i = 0; i < (pw - px); i++) {

            Vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++) {
                float u = float(i + px + Mirandom()) / float(w);
                float v = float(j + py + Mirandom()) / float(h);
                Ray r = cam.get_ray(u, v);
                col += world.getSceneColor(r);
            }
            col /= float(ns);
            col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            img[(j * patch_w + i) * 3 + 2] = char(255.99 * col[0]);
            img[(j * patch_w + i) * 3 + 1] = char(255.99 * col[1]);
            img[(j * patch_w + i) * 3 + 0] = char(255.99 * col[2]);
        }
    }
}



void rayTracingCPU_Frames(int w, int h, int ns, int num_frames) {
    printf("Iniciando ray tracing secuencial para %d frames (%dx%d, %d samples)...\n", 
           num_frames, w, h, ns);
    
    Scene world = randomScene();

    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));
    
    Vec3 lookFrom(13, 2, 3);
    Vec3 lookAt(0, 0, 0);
    float distToFocus = 10.0;
    float aperture = 0.1;
    
    Camera cam(lookFrom, lookAt, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, distToFocus);
    
    // Reservar memoria para la imagen
    int size = sizeof(unsigned char) * w * h * 3;
    unsigned char* data = (unsigned char*)calloc(size, 1);
    
    for (int frame = 0; frame < num_frames; frame++) {
        printf("Renderizando frame %d/%d...\n", frame + 1, num_frames);
        
        for (int j = h - 1; j >= 0; j--) {
            for (int i = 0; i < w; i++) {
                Vec3 col(0, 0, 0);
                for (int s = 0; s < ns; s++) {
                    float u = float(i + Mirandom()) / float(w);
                    float v = float(j + Mirandom()) / float(h);
                    Ray r = cam.get_ray(u, v);
                    col += world.getSceneColor(r);
                }
                col /= float(ns);
                col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

                data[(j * w + i) * 3 + 2] = char(255.99 * col[0]);
                data[(j * w + i) * 3 + 1] = char(255.99 * col[1]);
                data[(j * w + i) * 3 + 0] = char(255.99 * col[2]);
            }
        }
        char filename[50];
        sprintf(filename, "frame_%03d.bmp", frame + 1);
        writeBMP(filename, data, w, h);
        printf("Frame %d guardado como %s\n", frame + 1, filename);
    }
    free(data);
}


void rayTracingCPU_Frames_OpenMP(int w, int h, int ns, int num_frames, int px, int py, int pw, int ph) {
    if (pw == -1) pw = w;
    if (ph == -1) ph = h;
    int patch_w = pw - px;
    int patch_h = ph - py;

    Scene world = randomScene();
    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));
    
    Vec3 lookFrom(13, 2, 3);
    Vec3 lookAt(0, 0, 0);
    float distToFocus = 10.0;
    float aperture = 0.1;
    
    Camera cam(lookFrom, lookAt, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, distToFocus);
    
    #pragma omp parallel for
    for (int frame = 0; frame < num_frames; frame++) {
        printf("Renderizando frame %d/%d...\n", frame + 1, num_frames);
        
        // Cada thread necesita su propia memoria para la imagen
        int size = sizeof(unsigned char) * patch_w * patch_h * 3;
        unsigned char* data = (unsigned char*)calloc(size, 1);
        
        for (int j = 0; j < (ph - py); j++) {
            for (int i = 0; i < (pw - px); i++) {
                Vec3 col(0, 0, 0);
                for (int s = 0; s < ns; s++) {
                    float u = float(i + px + Mirandom()) / float(w);
                    float v = float(j + py + Mirandom()) / float(h);
                    Ray r = cam.get_ray(u, v);
                    col += world.getSceneColor(r);
                }
                col /= float(ns);
                col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

                data[(j * patch_w + i) * 3 + 2] = char(255.99 * col[0]);
                data[(j * patch_w + i) * 3 + 1] = char(255.99 * col[1]);
                data[(j * patch_w + i) * 3 + 0] = char(255.99 * col[2]);
            }
        }
        

        char filename[50];
        sprintf(filename, "frame_%03d.bmp", frame + 1);
        writeBMP(filename, data, patch_w, patch_h);
        printf("Frame %d guardado como %s\n", frame + 1, filename);
        
        free(data);
    }
}


void rayTracingCPU_OpenMP_Blocks(unsigned char* img, int w, int h, int ns, int px, int py, int pw, int ph) {
    if (pw == -1) pw = w;
    if (ph == -1) ph = h;
    int patch_w = pw - px;
    int patch_h = ph - py;
    

    Scene world = randomScene();
    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);

    int num_threads = omp_get_max_threads();
    int blocks_x, blocks_y;
    // Para los casos más comunes de pruebas (1, 2, 4, 6 hilos), se asignan divisiones exactas
    if (num_threads == 1) {
        blocks_x = 1; blocks_y = 1;
    } else if (num_threads == 2) {
        blocks_x = 2; blocks_y = 1;
    } else if (num_threads == 4) {
        blocks_x = 2; blocks_y = 2;
    } else if (num_threads == 6) {
        blocks_x = 3; blocks_y = 2;
    } else {
        // Para cualquier otro número de hilos, se busca un grid lo más cuadrado posible
        blocks_x = (int)sqrt(num_threads);
        blocks_y = (num_threads + blocks_x - 1) / blocks_x;
    }
    int block_w = (patch_w + blocks_x - 1) / blocks_x;
    int block_h = (patch_h + blocks_y - 1) / blocks_y;
    int total_blocks = blocks_x * blocks_y;
    
    #pragma omp parallel for schedule(static)
    for (int block_id = 0; block_id < total_blocks; block_id++) {
        // Calcular la posición del bloque en el grid de bloques
        int bx = block_id % blocks_x;  
        int by = block_id / blocks_x;
        // Calcular el píxel inicial
        int block_start_x = bx * block_w;
        int block_start_y = by * block_h;
        // Calcular el píxel final  del bloque en la imagen,si sobrepasa se ajusta al tamaño de la imagen
        int block_end_x = (block_start_x + block_w < patch_w) ? block_start_x + block_w : patch_w;
        int block_end_y = (block_start_y + block_h < patch_h) ? block_start_y + block_h : patch_h;
        
        
        for (int j = block_start_y; j < block_end_y; j++) {
            for (int i = block_start_x; i < block_end_x; i++) {
                Vec3 col(0, 0, 0);
                for (int s = 0; s < ns; s++) {
                    float u = float(i + px + Mirandom()) / float(w);
                    float v = float(j + py + Mirandom()) / float(h);
                    Ray r = cam.get_ray(u, v);
                    col += world.getSceneColor(r);
                }
                col /= float(ns);
                col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

                
                img[(j * patch_w + i) * 3 + 2] = char(255.99 * col[0]);
                img[(j * patch_w + i) * 3 + 1] = char(255.99 * col[1]);
                img[(j * patch_w + i) * 3 + 0] = char(255.99 * col[2]);
            }
        }
    }
}

void rayTracingCPU_OpenMP_Blocks_Manual(unsigned char* img, int w, int h, int ns, int px, int py, int pw, int ph) {
    if (pw == -1) pw = w;
    if (ph == -1) ph = h;
    int patch_w = pw - px;
    int patch_h = ph - py;
    
    Scene world = randomScene();
    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);

    int num_threads = omp_get_max_threads();
    int blocks_x, blocks_y;
    // Para los casos más comunes de pruebas (1, 2, 4, 6 hilos), se asignan divisiones exactas
    if (num_threads == 1) {
        blocks_x = 1; blocks_y = 1;
    } else if (num_threads == 2) {
        blocks_x = 2; blocks_y = 1;
    } else if (num_threads == 4) {
        blocks_x = 2; blocks_y = 2;
    } else if (num_threads == 6) {
        blocks_x = 3; blocks_y = 2;
    } else {
        // Para cualquier otro número de hilos, se busca un grid lo más cuadrado posible
        blocks_x = (int)sqrt(num_threads);
        blocks_y = (num_threads + blocks_x - 1) / blocks_x;
    }
    int block_w = (patch_w + blocks_x - 1) / blocks_x;
    int block_h = (patch_h + blocks_y - 1) / blocks_y;
    int total_blocks = blocks_x * blocks_y;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num(); 
        int np = omp_get_num_threads(); 
        
        
    
        for (int block_id = tid; block_id < total_blocks; block_id += np) {
           // Calcular la posición del bloque en el grid
            int bx = block_id % blocks_x;  
            int by = block_id / blocks_x;
            
            //Calcula el pixel inicial
            int block_start_x = bx * block_w;
            int block_start_y = by * block_h;
            int block_end_x = (block_start_x + block_w < patch_w) ? block_start_x + block_w : patch_w;
            int block_end_y = (block_start_y + block_h < patch_h) ? block_start_y + block_h : patch_h;
            //Calcula el pixel final si sobrepasa entonces se ajusta al tamaño de la imagen.
            for (int j = block_start_y; j < block_end_y; j++) {
                for (int i = block_start_x; i < block_end_x; i++) {
                    Vec3 col(0, 0, 0);
                    for (int s = 0; s < ns; s++) {
                        float u = float(i + px + Mirandom()) / float(w);
                        float v = float(j + py + Mirandom()) / float(h);
                        Ray r = cam.get_ray(u, v);
                        col += world.getSceneColor(r);
                    }
                    col /= float(ns);
                    col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

                    img[(j * patch_w + i) * 3 + 2] = char(255.99 * col[0]);
                    img[(j * patch_w + i) * 3 + 1] = char(255.99 * col[1]);
                    img[(j * patch_w + i) * 3 + 0] = char(255.99 * col[2]);
                }
            }
        }
    }
}

void rayTracingCPU_OpenMP_Blocks_Fixed(unsigned char* img, int w, int h, int ns, int px, int py, int pw, int ph, int block_size) {
    if (pw == -1) pw = w;
    if (ph == -1) ph = h;
    int patch_w = pw - px;
    int patch_h = ph - py;
    

    Scene world = randomScene();
    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);


    int blocks_x = (patch_w + block_size - 1) / block_size; 
    int blocks_y = (patch_h + block_size - 1) / block_size;
    int total_blocks = blocks_x * blocks_y;
    
    int num_threads = omp_get_max_threads();
    
    #pragma omp parallel for
    for (int block_id = 0; block_id < total_blocks; block_id++) {
        // Calcular la posición del bloque en el grid
        int bx = block_id % blocks_x; 
        int by = block_id / blocks_x;
        //Calcula el pixel inicial
        int block_start_x = bx * block_size;
        int block_start_y = by * block_size;
        //Calcula el pixel final si sobrepasa entonces se ajusta al tamaño de la imagen.
        int block_end_x = (block_start_x + block_size < patch_w) ? block_start_x + block_size : patch_w;
        int block_end_y = (block_start_y + block_size < patch_h) ? block_start_y + block_size : patch_h;
        

        for (int j = block_start_y; j < block_end_y; j++) {
            for (int i = block_start_x; i < block_end_x; i++) {
                Vec3 col(0, 0, 0);
                for (int s = 0; s < ns; s++) {
                    
                    float u = float(i + px + Mirandom()) / float(w);
                    float v = float(j + py + Mirandom()) / float(h);
                    Ray r = cam.get_ray(u, v);
                    col += world.getSceneColor(r);
                }
                col /= float(ns);
                col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

                img[(j * patch_w + i) * 3 + 2] = char(255.99 * col[0]);
                img[(j * patch_w + i) * 3 + 1] = char(255.99 * col[1]);
                img[(j * patch_w + i) * 3 + 0] = char(255.99 * col[2]);
            }
        }
    }
}