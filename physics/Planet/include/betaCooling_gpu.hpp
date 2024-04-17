//
// Created by Noah Kubli on 17.04.2024.
//

#pragma once

template<typename Tpos, typename Ts, typename Tdu>
void betaCoolingGPU(size_t first, size_t last, const Tpos* x, const Tpos* y, const Tpos* z, Tdu* du, Ts star_mass,
                     const Ts* star_pos, Ts beta, Tpos g);
