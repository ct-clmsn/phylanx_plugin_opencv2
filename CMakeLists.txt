# Copyright (c) 2018 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V3.3.2
cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

# explicitly set certain policies
cmake_policy(VERSION 3.3.2)
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
set_property(GLOBAL PROPERTY HPX_IGNORE_COMPILER_COMPATIBILITY TRUE)

################################################################################
# Build type (needs to be handled before project command below)
################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)")
endif()

  if(NOT HPX_DIR AND HPX_ROOT)
    set(HPX_DIR ${HPX_ROOT}/lib/cmake/HPX)
  endif()

  if(NOT HPX_DIR AND EXISTS "$ENV{HPX_DIR}")
    set(HPX_DIR $ENV{HPX_DIR})
  endif()

  if(EXISTS "${HPX_DIR}")
    find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)

    if(NOT HPX_FOUND)
      phylanx_error("HPX could not be found, please set HPX_DIR to help locating it.")
    endif()

    # make sure that configured build type for Phylanx matches the one used for HPX
    #if(NOT (${HPX_BUILD_TYPE} STREQUAL ${CMAKE_BUILD_TYPE}))
    #  list(FIND ${CMAKE_BUILD_TYPE} ${HPX_BUILD_TYPE} __pos)
    #  if(${__pos} EQUAL -1)
    #    phylanx_warn(
    #      "The configured CMAKE_BUILD_TYPE (${CMAKE_BUILD_TYPE}) is "
    #      "different from the build type used for the found HPX "
    #      "(HPX_BUILD_TYPE: ${HPX_BUILD_TYPE})")
    #  endif()
    #endif()

    include_directories(${HPX_INCLUDE_DIRS})
    link_directories(${HPX_LIBRARY_DIR})

    #if (HPX_GIT_COMMIT)
    #  string(SUBSTRING ${HPX_GIT_COMMIT} 0 10 __hpx_git_commit)
    #  phylanx_info("HPX version: " ${HPX_VERSION_STRING} "(${__hpx_git_commit})")
    #else()
    #  phylanx_info("HPX version: " ${HPX_VERSION_STRING})
    #endif()

    # make sure that HPX is not configured with jemalloc
    if(NOT MSVC AND ("${HPX_WITH_MALLOC}" STREQUAL "jemalloc"))
        phylanx_warn(
          "HPX is configured with: ${HPX_WITH_MALLOC}. Due to incompatibilities "
          "between the Python runtime and jemalloc, application execution will "
          "fail unless the jemalloc library is preloaded with LD_PRELOAD. For "
          "more reliable execution, we recommend reconfiguring HPX and Phylanx "
          "with TCMalloc")
    endif()
  endif()

#find_package(LAPACK REQUIRED)
#include_directories(${BLAS_INCLUDE_DIR})

find_package(blaze REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
if(NOT blaze_FOUND)
  phylanx_error("Blaze could not be found. Please specify blaze_DIR to assist locating it.")
endif()
include("${blaze_DIR}/blaze-config-version.cmake")
include_directories(${blaze_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11
#phylanx_info("Blaze version: " "${PACKAGE_VERSION}")

# Make sure HPX is used as the parallelization target for Blaze
#phylanx_add_config_define(BLAZE_USE_HPX_THREADS 1)
#phylanx_add_config_define(BLAZE_USE_SHARED_MEMORY_PARALLELIZATION 1)
#if(MSVC)
#  phylanx_add_config_cond_define(NOMINMAX 1)
#endif()

# Add tensors from BlazeTensors
find_package(BlazeTensor REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
include_directories(${BlazeTensor_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11
#if(NOT BlazeTensor_FOUND)
#  phylanx_error("BlazeTensor could not be found. Please specify BlazeTensor_DIR to assist locating it.")
#endif()
#phylanx_add_config_define(PHYLANX_HAVE_BLAZE_TENSOR)

#set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
#set_property(GLOBAL OpenCV_SHARED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11
link_libraries(${OpenCV_LIBS})

################################################################################
# project metadata
################################################################################
project(PhylanxOpenCV2Plugin C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

find_package(Phylanx)

add_subdirectory(plugin)
