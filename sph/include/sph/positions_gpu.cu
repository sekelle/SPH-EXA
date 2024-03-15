/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief 2nd order time-step integrator on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "eos.hpp"
#include "positions.hpp"
#include "sph_gpu.hpp"

namespace sph
{

//! Drift back by dt_back, then advance by dt
template<class Tc, class Thydro, class Tm1>
__global__ void driftKernel(GroupView grp, float dt, float dt_back, float dt_m1, Tc* x, Tc* y, Tc* z, Thydro* vx,
                            Thydro* vy, Thydro* vz, const Tm1* x_m1, const Tm1* y_m1, const Tm1* z_m1, const Thydro* ax,
                            const Thydro* ay, const Thydro* az, const uint8_t* rung)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= grp.numGroups) { return; }

    // drifting must not adjust for PBC because that would require octrees to be rebuilt
    cstone::Box<Tc> noPbc(0, 1, cstone::BoundaryType::open);
    auto            bodyBegin = grp.groupStart[tid];
    auto            bodyEnd   = grp.groupEnd[tid];

    for (auto i = bodyBegin; i < bodyEnd; ++i)
    {
        float            dt_m1_rung = (rung != nullptr) ? dt_m1 * (1 << rung[i]) : dt_m1;
        cstone::Vec3<Tc> An{ax[i], ay[i], az[i]};
        cstone::Vec3<Tc> Xnback{x[i], y[i], z[i]};
        cstone::Vec3<Tc> dXn{x_m1[i], y_m1[i], z_m1[i]};

        // recover Xn, at which point An was calculated
        cstone::Vec3<Tc> Xn_recov, ignore;
        util::tie(Xn_recov, ignore, ignore) = positionUpdate(-dt_back, dt_m1_rung, Xnback, An, dXn, noPbc);

        // drift to new point in time starting from (Xn, An, dXn)
        cstone::Vec3<Tc> Xnp1, Vnp1;
        util::tie(Xnp1, Vnp1, ignore) = positionUpdate(dt, dt_m1_rung, Xn_recov, An, dXn, noPbc);

        util::tie(x[i], y[i], z[i])    = util::tie(Xnp1[0], Xnp1[1], Xnp1[2]);
        util::tie(vx[i], vy[i], vz[i]) = util::tie(Vnp1[0], Vnp1[1], Vnp1[2]);
    }
}

template<class Tc, class Thydro, class Tm1>
void driftPositionsGpu(const GroupView& grp, float dt, float dt_back, float dt_m1, Tc* x, Tc* y, Tc* z, Thydro* vx,
                       Thydro* vy, Thydro* vz, const Tm1* x_m1, const Tm1* y_m1, const Tm1* z_m1, const Thydro* ax,
                       const Thydro* ay, const Thydro* az, const uint8_t* rung)
{
    unsigned           numThreads = 256;
    cstone::LocalIndex numBlocks  = (grp.numGroups + numThreads - 1) / numThreads;

    driftKernel<<<numBlocks, numThreads>>>(grp, dt, dt_back, dt_m1, x, y, z, vx, vy, vz, x_m1, y_m1, z_m1, ax, ay, az,
                                           rung);
}

#define DRIFT_GPU(Tc, Thydro, Tm1)                                                                                     \
    template void driftPositionsGpu(const GroupView& grp, float dt, float dt_back, float dt_m1, Tc* x, Tc* y, Tc* z,   \
                                    Thydro* vx, Thydro* vy, Thydro* vz, const Tm1* x_m1, const Tm1* y_m1,              \
                                    const Tm1* z_m1, const Thydro* ax, const Thydro* ay, const Thydro* az,             \
                                    const uint8_t* rung)

DRIFT_GPU(double, double, double);
DRIFT_GPU(double, double, float);
DRIFT_GPU(double, float, float);
DRIFT_GPU(float, float, float);

template<class Tc, class Tv, class Ta, class Tdu, class Tm1, class Tt, class Thydro>
__global__ void computePositionsKernel(GroupView grp, float dt, float dt_m1, Tc* x, Tc* y, Tc* z, Tv* vx, Tv* vy,
                                       Tv* vz, Tm1* x_m1, Tm1* y_m1, Tm1* z_m1, Ta* ax, Ta* ay, Ta* az,
                                       const uint8_t* rung, Tt* temp, Tt* u, Tdu* du, Tm1* du_m1, Thydro* h,
                                       Thydro* mui, Tc gamma, Tc constCv, const cstone::Box<Tc> box)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= grp.numGroups) { return; }

    auto bodyBegin = grp.groupStart[tid];
    auto bodyEnd   = grp.groupEnd[tid];
    for (auto i = bodyBegin; i < bodyEnd; ++i)
    {

        bool fbcX   = (box.boundaryX() == cstone::BoundaryType::fixed);
        bool fbcY   = (box.boundaryY() == cstone::BoundaryType::fixed);
        bool fbcZ   = (box.boundaryZ() == cstone::BoundaryType::fixed);
        bool anyFBC = fbcX || fbcY || fbcZ;

        if (anyFBC && vx[i] == Tv(0) && vy[i] == Tv(0) && vz[i] == Tv(0))
        {
            if (fbcCheck(x[i], h[i], box.xmax(), box.xmin(), fbcX) ||
                fbcCheck(y[i], h[i], box.ymax(), box.ymin(), fbcY) ||
                fbcCheck(z[i], h[i], box.zmax(), box.zmin(), fbcZ))
            {
                return;
            }
        }

        float            dt_m1_rung = (rung != nullptr) ? dt_m1 * (1 << rung[i]) : dt_m1;
        cstone::Vec3<Tc> A{ax[i], ay[i], az[i]};
        cstone::Vec3<Tc> X{x[i], y[i], z[i]};
        cstone::Vec3<Tc> X_m1{x_m1[i], y_m1[i], z_m1[i]};
        cstone::Vec3<Tc> V;
        util::tie(X, V, X_m1) = positionUpdate(dt, dt_m1_rung, X, A, X_m1, box);

        util::tie(x[i], y[i], z[i])          = util::tie(X[0], X[1], X[2]);
        util::tie(x_m1[i], y_m1[i], z_m1[i]) = util::tie(X_m1[0], X_m1[1], X_m1[2]);
        util::tie(vx[i], vy[i], vz[i])       = util::tie(V[0], V[1], V[2]);

        if (temp != nullptr)
        {
            Thydro cv    = (constCv < 0) ? idealGasCv(mui[i], gamma) : constCv;
            auto   u_old = temp[i] * cv;
            temp[i]      = energyUpdate(u_old, dt, dt_m1_rung, du[i], du_m1[i]) / cv;
            du_m1[i]     = du[i];
        }
        else if (u != nullptr)
        {
            u[i]     = energyUpdate(u[i], dt, dt_m1_rung, du[i], du_m1[i]);
            du_m1[i] = du[i];
        }
    }
}

template<class Tc, class Tv, class Ta, class Tdu, class Tm1, class Tt, class Thydro>
void computePositionsGpu(const GroupView& grp, float dt, float dt_m1, Tc* x, Tc* y, Tc* z, Tv* vx, Tv* vy, Tv* vz,
                         Tm1* x_m1, Tm1* y_m1, Tm1* z_m1, Ta* ax, Ta* ay, Ta* az, const uint8_t* rung, Tt* temp, Tt* u,
                         Tdu* du, Tm1* du_m1, Thydro* h, Thydro* mui, Tc gamma, Tc constCv, const cstone::Box<Tc>& box)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = (grp.numGroups + numThreads - 1) / numThreads;

    computePositionsKernel<<<numBlocks, numThreads>>>(grp, dt, dt_m1, x, y, z, vx, vy, vz, x_m1, y_m1, z_m1, ax, ay, az,
                                                      rung, temp, u, du, du_m1, h, mui, gamma, constCv, box);
}

#define POS_GPU(Tc, Tv, Ta, Tdu, Tm1, Tt, Thydro)                                                                      \
    template void computePositionsGpu(const GroupView& grp, float dt, float dt_m1, Tc* x, Tc* y, Tc* z, Tv* vx,        \
                                      Tv* vy, Tv* vz, Tm1* x_m1, Tm1* y_m1, Tm1* z_m1, Ta* ax, Ta* ay, Ta* az,         \
                                      const uint8_t* rung, Tt* temp, Tt* u, Tdu* du, Tm1* du_m1, Thydro* h,            \
                                      Thydro* mui, Tc gamma, Tc constCv, const cstone::Box<Tc>& box)

//        Tc      Tv     Ta      Tdu     Tm1     Tt      Thydro
POS_GPU(double, double, double, double, double, double, double);
POS_GPU(float, float, float, float, float, float, float);
POS_GPU(double, double, double, float, float, double, double);
POS_GPU(double, float, float, double, float, double, float);

} // namespace sph
