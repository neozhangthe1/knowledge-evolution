#pragma once
#include <vector>

std::vector<int> affinityPropagation(
    FILE* input,
    int prefType = 1,
    double damping = 0.5,
    int maxit = 1000,
    int convit = 30
);
