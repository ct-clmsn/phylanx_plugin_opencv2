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
#include <functional>

#include "opencv2_imread.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace blaze;

///////////////////////////////////////////////////////////////////////////////
namespace phylanx_plugin
{
    constexpr char const* const help_string = R"(
        cv.imread(name, image_type)
        Args:

            name (string) : string to a file path
            name (int) : int cv.IMREAD_COLOR/cv.IMREAD_GRAYSCALE (optional)

        Returns:

            blaze::DynamicTensor<std::uint8>
        )";

    ///////////////////////////////////////////////////////////////////////////
    phylanx::execution_tree::match_pattern_type const
        opencv2_imread::match_data =
        {

            hpx::util::make_tuple("imread",
                std::vector<std::string>{"imread(_1)", "imread(_1, _2)"},
                &create_opencv2_imread,
                &phylanx::execution_tree::create_primitive<opencv2_imread>,
                help_string
            )
        };

    ///////////////////////////////////////////////////////////////////////////
    opencv2_imread::opencv2_imread(
            primitive_arguments_type&& operands, std::string const& name,
            std::string const& codename)
      : phylanx::execution_tree::primitives::primitive_component_base(
            std::move(operands), name, codename)
    {}

    ///////////////////////////////////////////////////////////////////////////
    phylanx::execution_tree::primitive_argument_type opencv2_imread::calculate(
        primitive_arguments_type && args) const
    {
        // extract arguments
        auto const fname = extract_string_value(args[0]);
        auto const imread_type = args.size() == 1 ? IMREAD_COLOR : extract_scalar_integer_value(args[1]);

        Mat const img = imread(fname.c_str(), imread_type);
        if(img.data == nullptr) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "opencv2_imread::eval",
                generate_error_message(
                    "image file not loaded successfully: " + fname));
        }

        DynamicTensor<std::uint8_t> bimg(img.rows, img.cols, img.channels(), img.data);
        return phylanx::execution_tree::primitive_argument_type{std::move(bimg)};
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<phylanx::execution_tree::primitive_argument_type>
    opencv2_imread::eval(primitive_arguments_type const& operands,
        primitive_arguments_type const& args, eval_context ctx) const
    {
        auto const operands_size = operands.size();
        if (operands_size != 2 || operands_size != 1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "opencv2_imread::eval",
                generate_error_message(
                    "opencv2_imread accepts exactly two "
                    "arguments"));
        }

        bool arguments_valid = true;
        for (std::size_t i = 0; i != operands.size(); ++i)
        {
            if (!valid(operands[i]))
            {
                arguments_valid = false;
            }
        }

        if (!arguments_valid)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "stack_operation::eval",
                generate_error_message(
                    "the stack_operation primitive requires that "
                        "the arguments given by the operands array "
                        "are valid"));
        }

        auto this_ = this->shared_from_this();
        return hpx::dataflow(hpx::launch::sync, hpx::util::unwrapping(
            [this_ = std::move(this_)](primitive_arguments_type && args)
            ->  primitive_argument_type
            {
                return this_->calculate(std::move(args));
            }),
            phylanx::execution_tree::primitives::detail::map_operands(
                operands, phylanx::execution_tree::functional::value_operand{}, args,
                name_, codename_));
    }
}
