//
// Created by Noah Kubli on 11.03.2024.
//

#pragma once

template<typename T1, typename Ta, typename Tm, typename T2>
void computeCentralForceGPU(size_t first, size_t last, const T1* x, const T1* y, const T1* z, Ta* ax, Ta* ay, Ta* az,
                            const Tm* m, const T2* spos, T2 sm, T2* sf, T2* spot, T1 G);
