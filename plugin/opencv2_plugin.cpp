// Copyright (c) 2018 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/plugins/plugin_factory.hpp>

#include "opencv2_plugin.hpp"

PHYLANX_REGISTER_PLUGIN_MODULE();

PHYLANX_REGISTER_PLUGIN_FACTORY(opencv2_imread_color_plugin,
    phylanx_plugin::opencv2_imread_color::match_data);
PHYLANX_REGISTER_PLUGIN_FACTORY(opencv2_imread_gray_plugin,
    phylanx_plugin::opencv2_imread_gray::match_data);
