// Copyright (c) 2018 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/throw_exception.hpp>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "opencv2_imread_gray.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace blaze;

///////////////////////////////////////////////////////////////////////////////
namespace phylanx_plugin
{
    constexpr char const* const help_string = R"(
        opencv2_imread_gray(name)
        Args:

            name (string) : string to a file path

        Returns:

            blaze::DynamicTensor<std::uint8>
        )";

    ///////////////////////////////////////////////////////////////////////////
    phylanx::execution_tree::match_pattern_type const
        opencv2_imread_gray::match_data =
        {
            hpx::util::make_tuple("opencv2_imread_gray",
                std::vector<std::string>{"opencv2_imread_gray(_1)"},
                &create_opencv2_imread_gray,
                &phylanx::execution_tree::create_primitive<opencv2_imread_gray>,
                help_string
            )
        };

    namespace detail
    {
        std::string extract_function_name(std::string const& name)
        {
            using namespace phylanx::execution_tree::compiler;

            primitive_name_parts name_parts;
            if (!parse_primitive_name(name, name_parts))
            {
                return name;
            }
            return name_parts.primitive;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    opencv2_imread_gray::opencv2_imread_gray(
            primitive_arguments_type&& operands, std::string const& name,
            std::string const& codename)
      : phylanx::execution_tree::primitives::primitive_component_base(
            std::move(operands), name, codename)
    {}

    ///////////////////////////////////////////////////////////////////////////
    blaze::DynamicTensor<std::uint8_t> opencv2_imread_gray::calculate(std::string const& name) const
    {
        Mat const img = imread(name.c_str(), IMREAD_GRAYSCALE);
        if(img.data == nullptr) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "opencv2_imread_gray::eval",
                generate_error_message(
                    "image file not loaded successfully: " + name));
        }

        DynamicTensor<std::uint8_t> bimg(img.rows, img.cols, img.channels(), img.data);
        return bimg;
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<phylanx::execution_tree::primitive_argument_type>
    opencv2_imread_gray::eval(primitive_arguments_type const& operands,
        primitive_arguments_type const& args, eval_context ctx) const
    {
        if (operands.size() > 1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "opencv2_imread_gray::eval",
                generate_error_message(
                    "opencv2_imread_gray accepts either none or exactly one "
                    "argument"));
        }

        if (operands.empty())
        {
            // no arguments, derive functionality from primitive name
            return hpx::make_ready_future(primitive_argument_type{
                calculate(detail::extract_function_name(name_))});
        }

        auto this_ = this->shared_from_this();
        return string_operand(
                operands[0], args, name_, codename_, std::move(ctx))
            .then(
                [this_](hpx::future<std::string> val)
                ->  primitive_argument_type
                {
                    return primitive_argument_type{
                        this_->calculate(val.get())};
                });
    }
}
