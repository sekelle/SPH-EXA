//
// Created by Noah Kubli on 31.03.2024.
//

#pragma once

#include "cstone/fields/field_get.hpp"
#include <type_traits>

//! @brief Generate FieldList by excluding a certain field if included.
template<util::StructuralString exclude, typename Fields>
struct FieldListExclude
{
};

template<util::StructuralString exclude, util::StructuralString field, util::StructuralString... fields>
struct FieldListExclude<exclude, util::FieldList<field, fields...>>
{
    using rest  = typename FieldListExclude<exclude, util::FieldList<fields...>>::type;
    using eq    = std::bool_constant<exclude == field>;
    using type = std::conditional_t<eq::value, rest, decltype(util::FieldList<field>{} + rest{})>;
};

template<util::StructuralString exclude>
struct FieldListExclude<exclude, util::FieldList<>>
{
    using type = util::FieldList<>;
};

template<util::StructuralString exclude, typename Fields>
using FieldListExclude_t = typename FieldListExclude<exclude, Fields>::type;
