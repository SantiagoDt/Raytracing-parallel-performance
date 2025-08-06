#pragma once

#include "Vec3.cuh"

struct CollisionData {
    float time;
    Vec3 p;
    Vec3 normal;
};
