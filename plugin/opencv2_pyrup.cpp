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

#include "opencv2_pyrup.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace blaze;

///////////////////////////////////////////////////////////////////////////////
namespace phylanx_plugin
{
    constexpr char const* const help_string = R"(
        opencv2_pyrup(name)
        Args:

            name (string) : string to a file path

        Returns:

            blaze::DynamicTensor<std::uint8>
        )";

    ///////////////////////////////////////////////////////////////////////////
    phylanx::execution_tree::match_pattern_type const
        opencv2_pyrup::match_data =
        {
            hpx::util::make_tuple("opencv2_pyrup",
                std::vector<std::string>{"opencv2_pyrup(_1)"},
                &create_opencv2_pyrup,
                &phylanx::execution_tree::create_primitive<opencv2_pyrup>,
                help_string
            )
        };

    ///////////////////////////////////////////////////////////////////////////
    opencv2_pyrup::opencv2_pyrup(
            primitive_arguments_type&& operands, std::string const& name,
            std::string const& codename)
      : phylanx::execution_tree::primitives::primitive_component_base(
            std::move(operands), name, codename)
    {}

    ///////////////////////////////////////////////////////////////////////////
    phylanx::execution_tree::primitive_argument_type opencv2_pyrup::calculate(
        primitive_arguments_type && args) const
    {
        auto arg1 = extract_numeric_value(args[0], name_, codename_);
        auto img = arg1.tensor();

        Mat cvimgin(img.rows(), img.columns(), CV_8UC3, img.data());
        if(cvimgin.data == nullptr) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "opencv2_pyrup::eval",
                generate_error_message(
                    "image not accessed successfully"));
        }

        Mat cvimgout;
        pyrUp(cvimgin, cvimgout, cv::Size(cvimgin.cols*2, cvimgin.rows*2) );
        DynamicTensor<std::uint8_t> bimg(cvimgout.rows, cvimgout.cols, cvimgout.channels(), cvimgout.data);
        return phylanx::execution_tree::primitive_argument_type{std::move(bimg)};
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<phylanx::execution_tree::primitive_argument_type>
    opencv2_pyrup::eval(primitive_arguments_type const& operands,
        primitive_arguments_type const& args, eval_context ctx) const
    {
        if (operands.size() != 1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "opencv2_pyrup::eval",
                generate_error_message(
                    "opencv2_pyrup accepts either none or exactly one "
                    "argument"));
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
