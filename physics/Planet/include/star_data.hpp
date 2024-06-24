//
// Created by Noah Kubli on 07.03.2024.
//

#pragma once

#include <array>
#include <iostream>

struct StarData
{
    std::array<double, 3> position{};
    std::array<double, 3> position_m1{};
    double                m{1.};
    double                inner_size{5.};
    double                beta{6.28};
    double                removal_limit_h{5.};
    float                 cooling_rho_limit{1.683e-3};
    double                u_floor{9.3e-6}; // 9.3e-6 approx. 1 K; 2.73 K: u = 2.5e-5;
    double                K_u{0.25};
    double du_adjust{std::numeric_limits<double>::infinity()}; //  ~ 0.25 * u_typical / t_resolve; 0.25 * 5e-5
                                                               // / 0.125 = 1e-4

    template<typename Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                ar->stepAttribute(attribute, location, attrSize);
            }
            catch (std::out_of_range&)
            {
                if (ar->rank() == 0)
                {
                    std::cout << "Attribute " << attribute
                              << " not set in file or initializer, setting to default value " << *location << std::endl;
                }
            }
        };

        optionalIO("star::x", &position[0], 1);
        optionalIO("star::y", &position[1], 1);
        optionalIO("star::z", &position[2], 1);
        optionalIO("star::x_m1", &position_m1[0], 1);
        optionalIO("star::y_m1", &position_m1[1], 1);
        optionalIO("star::z_m1", &position_m1[2], 1);
        optionalIO("star::m", &m, 1);
        optionalIO("star::inner_size", &inner_size, 1);
        optionalIO("star::beta", &beta, 1);
        optionalIO("star::removal_limit_h", &removal_limit_h, 1);
        optionalIO("star::cooling_rho_limit", &cooling_rho_limit, 1);
        optionalIO("star::u_floor", &u_floor, 1);
        optionalIO("star::K_u", &K_u, 1);
        optionalIO("star::du_adjust", &du_adjust, 1);
    };

    //! @brief Potential from interaction between star and particles
    double potential{};

    // Local to Rank

    std::array<double, 3> force_local{};
    double                potential_local{};
    size_t                n_accreted_local{};
    size_t                n_removed_local{};
    double                m_accreted_local{};
    std::array<double, 3> p_accreted_local{};
};