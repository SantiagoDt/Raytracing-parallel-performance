#pragma once

#include "CollisionData.cuh"
#include "Ray.cuh"

class Shape {
  public:
    __host__ __device__ virtual bool collide(const Ray &ray, float t_min, float t_max, CollisionData &cd) const = 0;
};
