//
// Created by Noah Kubli on 31.03.2024.
//

#pragma once

#include "cstone/fields/field_get.hpp"
#include <type_traits>

template<util::StructuralString exclude, typename Fields>
struct FieldListExclude
{
};

template<util::StructuralString exclude, util::StructuralString field, util::StructuralString... fields>
struct FieldListExclude<exclude, util::FieldList<field, fields...>>
{
    using rest  = FieldListExclude<exclude, util::FieldList<fields...>>::value;
    using eq    = std::bool_constant<exclude == field>;
    using value = std::conditional_t<eq::value, rest, decltype(util::FieldList<field>{} + rest{})>;
};

template<util::StructuralString exclude>
struct FieldListExclude<exclude, util::FieldList<>>
{
    using value = util::FieldList<>;
};
