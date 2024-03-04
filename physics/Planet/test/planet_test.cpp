//
// Created by Noah Kubli on 05.03.2024.
//
#include "gtest/gtest.h"
#include "cstone/domain/domain.hpp"
#include <mpi.h>
#include "computeCentralForce.hpp"
#include "sph/particles_data.hpp"
#include "cstone/fields/field_get.hpp"

struct PlanetTest : public ::testing::Test
{
    int rank = 0, numRanks = 0;

    PlanetTest()
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized == 0) { MPI_Init(0, NULL); }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    }
};

TEST_F(PlanetTest, testStarPosition)
{

    if (numRanks != 2) throw std::runtime_error("Must be excuted with two ranks");

    std::array<double, 3> force{1., 0., 0.};
    std::array<double, 3> star_pos{0., 1., 0.};
    std::array<double, 3> star_pos_m1{0., 0., 0.};
    planet::computeAndExchangeStarPosition(force, star_pos, star_pos_m1, 1., 1.);
    if (rank == 0) { printf("rank: %d star pos: %lf\t%lf\t%lf\n", rank, star_pos[0], star_pos[1], star_pos[2]); }
    if (rank == 1) { printf("rank: %d star pos: %lf\t%lf\t%lf\n", rank, star_pos[0], star_pos[1], star_pos[2]); }
    EXPECT_TRUE(star_pos[0] == 2.);
    EXPECT_TRUE(star_pos[1] == 1.);
    EXPECT_TRUE(star_pos[2] == 0.);
}

static void fill_grid(auto& x, auto& y, auto& z, double start, double end, size_t n_elements)
{
    assert(x.size() == y.size() == z.size());

    for (size_t i = 0; i < x.size(); i++)
    {
        const size_t iz = i / (n_elements * n_elements);
        const size_t iy = i / n_elements % n_elements;
        const size_t ix = i % n_elements;
        x[i]            = start + ix * (end - start) / n_elements;
        y[i]            = start + iy * (end - start) / n_elements;
        z[i]            = start + iz * (end - start) / n_elements;
    }
};

TEST_F(PlanetTest, testAccretion)
{
    const double inner_limit = 0.2;
    // int rank = 0, numRanks = 0;
    if (numRanks != 2) throw std::runtime_error("Must be excuted with two ranks");

    using KeyType = uint64_t;
    using T       = double;
    // MPI_Init(0, NULL);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    int   bucketSize = 1;
    float theta      = 1.0;

    cstone::Domain<KeyType, T> domain(rank, numRanks, bucketSize, bucketSize, theta);

    constexpr size_t n_elements  = 20;
    constexpr size_t n_elements3 = n_elements * n_elements * n_elements;

    sphexa::ParticlesData<cstone::CpuTag> data;
    data.setConserved("x", "y", "z", "h");
    data.setDependent("keys");
    data.resize(n_elements3);

    fill_grid(data.x, data.y, data.z, -1., 1., n_elements);
    std::fill(data.h.begin(), data.h.end(), 0.05);
    auto countInside = [&domain, inner_limit](const auto& d, size_t start, size_t end)
    {
        size_t count = 0;
        for (size_t i = start; i < end; i++)
        {
            double x    = d.x[i];
            double y    = d.y[i];
            double z    = d.z[i];
            double dist = std::sqrt(x * x + y * y + z * z);
            if (dist < inner_limit) count++;
        }
        return count;
    };
    std::vector<double>   s1, s2;
    std::vector<uint64_t> s3;
    std::vector<float>    s7, s8;
    std::cout << "inside init: " << countInside(data, 0, data.x.size()) << std::endl;

    domain.sync(data.keys, data.x, data.y, data.z, data.h, std::tuple{}, std::tie(s1, s2, s3, s7, s8));

    // std::cout << "inside sync: " << countInside(data, domain.startIndex(), domain.endIndex()) << std::endl;

    using Fields = util::FieldList<"x", "y", "z", "h", "keys">;
    planet::accreteParticles<Fields>(data, domain, {0., 0., 0.}, inner_limit, 100.);
    // std::cout << "inside accreteParticles: " << countInside(data, domain.startIndex(), domain.endIndex()) <<
    // std::endl;

    domain.sync(data.keys, data.x, data.y, data.z, data.h, std::tuple{}, std::tie(s1, s2, s3, s7, s8));

    size_t n_inside            = countInside(data, domain.startIndex(), domain.endIndex());
    size_t n_inside_with_halos = countInside(data, 0, data.x.size());

    EXPECT_TRUE(n_inside == 0);
    EXPECT_TRUE(n_inside_with_halos == 0);
}