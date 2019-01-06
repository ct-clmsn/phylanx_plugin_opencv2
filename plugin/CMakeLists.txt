# Copyright (c) 2018 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(plugin_headers
    ${CMAKE_CURRENT_LIST_DIR}/opencv2_plugin.hpp
    ${CMAKE_CURRENT_LIST_DIR}/opencv2_imread.hpp
    ${CMAKE_CURRENT_LIST_DIR}/opencv2_pyrup.hpp
    ${CMAKE_CURRENT_LIST_DIR}/opencv2_pyrdown.hpp)

set(plugin_sources
    ${CMAKE_CURRENT_LIST_DIR}/opencv2_plugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/opencv2_imread.cpp
    ${CMAKE_CURRENT_LIST_DIR}/opencv2_pyrup.cpp
    ${CMAKE_CURRENT_LIST_DIR}/opencv2_pyrdown.cpp)

add_phylanx_primitive_plugin(
    ocv2
    HEADERS ${plugin_headers}
    SOURCES ${plugin_sources})
