# Especificar versión mínima de CMake
# CMake 3.20 es la versión más reciente que viene incluida con
# Visual Studio 2019
cmake_minimum_required(VERSION "3.20")

# Declaramos el proyecto de CMake
# En caso de que no haga falta alguno de los lenguajes listados,
# se puede deshabilitar simplemente eliminándolo de la lista
# https://cmake.org/cmake/help/latest/command/project.html
project(RayTracingCPU LANGUAGES "CXX")

# Configurar tipo de build por defecto a Release si no se especifica
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Configurar flags de optimización para Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")
endif()

# Configurar estándar de C++ (recomendado C++17 o superior para ray tracing)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#################################################
# Importación de dependencias
#################################################

# Buscar soporte de OpenMP para C++ como opcional
# Si no hace falta usar OpenMP, esta línea se puede comentar
# https://cmake.org/cmake/help/latest/command/find_package.html
# https://cmake.org/cmake/help/latest/module/FindOpenMP.html
find_package("OpenMP" COMPONENTS "CXX")

# Buscar una implementación de la Message Parsing Interface como opcional
# Si no hace falta usar MPI, esta línea se puede comentar
# https://cmake.org/cmake/help/latest/command/find_package.html
# https://cmake.org/cmake/help/latest/module/FindMPI.html
find_package("MPI" COMPONENTS "CXX")

#################################################
# Archivos fuente comunes
#################################################

set(COMMON_SOURCES
    implementations/Camera.h
    implementations/CollisionData.h
    implementations/Crystalline.cpp
    implementations/Crystalline.h
    implementations/Diffuse.h
    implementations/Material.h
    implementations/Metallic.cpp
    implementations/Metallic.h
    implementations/Object.h
    implementations/Ray.h
    implementations/RayTracingImplementations.cpp
    implementations/RayTracingImplementations.hpp
    implementations/Scene.cpp
    implementations/Scene.h
    implementations/Shape.h
    implementations/Sphere.cpp
    implementations/Sphere.h
    implementations/Vec3.h
    implementations/random.cpp
    implementations/random.h
    implementations/utils.cpp
    implementations/utils.h
)

#################################################
# Ejecutables principales
#################################################

# Ejecutable OpenMP chunk estático manual
add_executable(omp_raytracing_chunk_estatico_manual
    ${COMMON_SOURCES}
    openmp/main_omp_chunk_estatico_manual.cpp
)
set_target_properties(omp_raytracing_chunk_estatico_manual PROPERTIES OUTPUT_NAME "omp_raytracing_chunk_estatico_manual")
if(OpenMP_CXX_FOUND)
    target_link_libraries(omp_raytracing_chunk_estatico_manual PRIVATE OpenMP::OpenMP_CXX)
endif()

# Ejecutable OpenMP chunk dinámico
add_executable(omp_raytracing_chunk_dinamico
    ${COMMON_SOURCES}
    openmp/main_omp_chunk_dinamico.cpp
)
set_target_properties(omp_raytracing_chunk_dinamico PROPERTIES OUTPUT_NAME "omp_raytracing_chunk_dinamico")
if(OpenMP_CXX_FOUND)
    target_link_libraries(omp_raytracing_chunk_dinamico PRIVATE OpenMP::OpenMP_CXX)
endif()

# Archivos fuente para el ejecutable secuencial (sin RayTracingImplementations.cpp)
set(SEQUENCIAL_SOURCES
    implementations/Camera.h
    implementations/CollisionData.h
    implementations/Crystalline.cpp
    implementations/Crystalline.h
    implementations/Diffuse.h
    implementations/Material.h
    implementations/Metallic.cpp
    implementations/Metallic.h
    implementations/Object.h
    implementations/Ray.h
    implementations/Scene.cpp
    implementations/Scene.h
    implementations/Shape.h
    implementations/Sphere.cpp
    implementations/Sphere.h
    implementations/Vec3.h
    implementations/random.cpp
    implementations/random.h
    implementations/utils.cpp
    implementations/utils.h
)

# Ejecutable secuencial (sin MPI ni OpenMP)
add_executable(raytracing_sequencial
    ${SEQUENCIAL_SOURCES}
    main_sequencial.cpp
)
set_target_properties(raytracing_sequencial PROPERTIES OUTPUT_NAME "raytracing_sequencial")

# Ejecutable OpenMP por filas
add_executable(omp_raytracing_rows
    ${COMMON_SOURCES}
    openmp/main_rows.cpp
)
set_target_properties(omp_raytracing_rows PROPERTIES OUTPUT_NAME "omp_raytracing_rows")
if(OpenMP_CXX_FOUND)
    target_link_libraries(omp_raytracing_rows PRIVATE OpenMP::OpenMP_CXX)
endif()

# Ejecutable OpenMP por columnas
add_executable(omp_raytracing_columns
    ${COMMON_SOURCES}
    openmp/main_columns.cpp
)
set_target_properties(omp_raytracing_columns PROPERTIES OUTPUT_NAME "omp_raytracing_columns")
if(OpenMP_CXX_FOUND)
    target_link_libraries(omp_raytracing_columns PRIVATE OpenMP::OpenMP_CXX)
endif()

# Ejecutable OpenMP por bloques automáticos
add_executable(omp_raytracing_chunk_estatico_auto
    ${COMMON_SOURCES}
    openmp/main_omp_chunk_estatico_auto.cpp
)
set_target_properties(omp_raytracing_chunk_estatico_auto PROPERTIES OUTPUT_NAME "omp_raytracing_chunk_estatico_auto")
if(OpenMP_CXX_FOUND)
    target_link_libraries(omp_raytracing_chunk_estatico_auto PRIVATE OpenMP::OpenMP_CXX)
endif()

# Ejecutable secuencial por frames
add_executable(raytracing_frames
    ${COMMON_SOURCES}
    main_frames.cpp
)
set_target_properties(raytracing_frames PROPERTIES OUTPUT_NAME "raytracing_frames")
if(OpenMP_CXX_FOUND)
    target_link_libraries(raytracing_frames PRIVATE OpenMP::OpenMP_CXX)
endif()

# Ejecutable OpenMP por frames
add_executable(omp_raytracing_frames
    ${COMMON_SOURCES}
    openmp/main_frames_openmp.cpp
)
set_target_properties(omp_raytracing_frames PROPERTIES OUTPUT_NAME "omp_raytracing_frames")
if(OpenMP_CXX_FOUND)
    target_link_libraries(omp_raytracing_frames PRIVATE OpenMP::OpenMP_CXX)
endif()

#################################################
# Ejecutables MPI
#################################################
#################################################

if(MPI_CXX_FOUND)
    # Ejecutable MPI por filas
    add_executable(mpi_raytracing_rows
        ${COMMON_SOURCES}
        mpi/main_mpi_rows.cpp
    )
    set_target_properties(mpi_raytracing_rows PROPERTIES OUTPUT_NAME "mpi_raytracing_rows")
    target_link_libraries(mpi_raytracing_rows PRIVATE MPI::MPI_CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mpi_raytracing_rows PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Ejecutable MPI chunk dinámico 64x64
    add_executable(mpi_raytracing_chunk_dinamico_64
        ${COMMON_SOURCES}
        mpi/main_mpi_chunk_dinamico_64.cpp
    )
    set_target_properties(mpi_raytracing_chunk_dinamico_64 PROPERTIES OUTPUT_NAME "mpi_raytracing_chunk_dinamico_64")
    target_link_libraries(mpi_raytracing_chunk_dinamico_64 PRIVATE MPI::MPI_CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mpi_raytracing_chunk_dinamico_64 PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Ejecutable MPI por columnas
    add_executable(mpi_raytracing_columns
        ${COMMON_SOURCES}
        mpi/main_mpi_columns.cpp
    )
    set_target_properties(mpi_raytracing_columns PROPERTIES OUTPUT_NAME "mpi_raytracing_columns")
    target_link_libraries(mpi_raytracing_columns PRIVATE MPI::MPI_CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mpi_raytracing_columns PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Ejecutable MPI por frames
    add_executable(mpi_raytracing_frames
        ${COMMON_SOURCES}
        mpi/main_mpi_frames.cpp
    )
    set_target_properties(mpi_raytracing_frames PROPERTIES OUTPUT_NAME "mpi_raytracing_frames")
    target_link_libraries(mpi_raytracing_frames PRIVATE MPI::MPI_CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mpi_raytracing_frames PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Ejecutable MPI chunk estático (rectángulos adaptativos)
    add_executable(mpi_raytracing_chunk_estatico
        ${COMMON_SOURCES}
        mpi/main_mpi_chunk_estatico.cpp
    )
    set_target_properties(mpi_raytracing_chunk_estatico PROPERTIES OUTPUT_NAME "mpi_raytracing_chunk_estatico")
    target_link_libraries(mpi_raytracing_chunk_estatico PRIVATE MPI::MPI_CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mpi_raytracing_chunk_estatico PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Ejecutable MPI + OpenMP híbrido por frames
    add_executable(mpi_openmp_hybrid
        ${COMMON_SOURCES}
        main_mpi_openmp.cpp
    )
    set_target_properties(mpi_openmp_hybrid PROPERTIES OUTPUT_NAME "mpi_openmp_hybrid")
    target_include_directories(mpi_openmp_hybrid PRIVATE implementations)
    target_link_libraries(mpi_openmp_hybrid PRIVATE MPI::MPI_CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mpi_openmp_hybrid PRIVATE OpenMP::OpenMP_CXX)
    endif()

else()
    message(WARNING "MPI no encontrado. Los ejecutables MPI no serán compilados.")
endif()

#################################################
# Información de compilación
#################################################

# Mostrar información sobre las dependencias encontradas
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP encontrado. Versión: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP no encontrado. Algunas funcionalidades pueden estar limitadas.")
endif()

if(MPI_CXX_FOUND)
    message(STATUS "MPI encontrado. Versión: ${MPI_CXX_VERSION}")
else()
    message(WARNING "MPI no encontrado. Los ejecutables MPI no serán compilados.")
endif()

#################################################
# Lista de objetivos a instalar
#################################################

set(INSTALL_TARGETS omp_raytracing_chunk_estatico_manual omp_raytracing_chunk_dinamico raytracing_sequencial omp_raytracing_rows omp_raytracing_columns omp_raytracing_chunk_estatico_auto raytracing_frames omp_raytracing_frames)

if(MPI_CXX_FOUND)
    list(APPEND INSTALL_TARGETS 
        mpi_raytracing_rows 
        mpi_raytracing_chunk_dinamico_64 
        mpi_raytracing_columns 
        mpi_raytracing_frames 
        mpi_raytracing_chunk_estatico
    )
endif()

# Declarar reglas de instalación
# Esto sirve para decirle a CMake cómo debe exportar los archivos compilados
# en el caso de que así se desee
# https://cmake.org/cmake/help/latest/command/install.html
# https://cmake.org/cmake/help/latest/manual/cmake.1.html#install-a-project
install(TARGETS ${INSTALL_TARGETS})

#################################################
# Información de ayuda
#################################################

message(STATUS "")
message(STATUS "=== Configuración del proyecto RayTracingCPU ===")
message(STATUS "Tipo de build: ${CMAKE_BUILD_TYPE}")
message(STATUS "Estándar C++: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Ejecutables que serán compilados:")
message(STATUS "  - omp_raytracing_chunk_estatico_manual (openmp/main_omp_chunk_estatico_manual.cpp) - OpenMP chunks estáticos manuales")
message(STATUS "  - omp_raytracing_chunk_dinamico (openmp/main_omp_chunk_dinamico.cpp) - OpenMP chunks dinámicos 128x128")
message(STATUS "  - raytracing_sequencial (main_sequencial.cpp) - SIN paralelización")
message(STATUS "  - omp_raytracing_rows (openmp/main_rows.cpp) - OpenMP por filas")
message(STATUS "  - omp_raytracing_columns (openmp/main_columns.cpp) - OpenMP por columnas")
message(STATUS "  - omp_raytracing_chunk_estatico_auto (openmp/main_omp_chunk_estatico_auto.cpp) - OpenMP chunks estáticos automáticos")
message(STATUS "  - raytracing_frames (main_frames.cpp) - Secuencial multi-frame")
message(STATUS "  - omp_raytracing_frames (openmp/main_frames_openmp.cpp) - OpenMP multi-frame")
if(MPI_CXX_FOUND)
    message(STATUS "  - mpi_raytracing_rows (mpi/main_mpi_rows.cpp) - MPI por filas")
    message(STATUS "  - mpi_raytracing_chunk_dinamico_64 (mpi/main_mpi_chunk_dinamico_64.cpp) - MPI chunks dinámicos 64x64")
    message(STATUS "  - mpi_raytracing_columns (mpi/main_mpi_columns.cpp) - MPI por columnas")
    message(STATUS "  - mpi_raytracing_frames (mpi/main_mpi_frames.cpp) - MPI por frames")
    message(STATUS "  - mpi_raytracing_chunk_estatico (mpi/main_mpi_chunk_estatico.cpp) - MPI chunks estáticos adaptativos")
    message(STATUS "  - mpi_openmp_hybrid (main_mpi_openmp.cpp) - MPI + OpenMP híbrido por frames")
endif()
message(STATUS "")
message(STATUS "Nota: raytracingcpu_sequencial incluido como referencia (sin paralelización).")
message(STATUS "===============================================")
