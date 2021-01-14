cmake_minimum_required(VERSION 3.5)
project(particletracing)


add_subdirectory(pybind11)
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

#find_package(NumPy REQUIRED)
#find_package(OpenMP)


set(CMAKE_CXX_FLAGS "-O3 -march=native")

#add_subdirectory(pybind11)

add_executable(${PROJECT_NAME} particletracing/particletracing.cpp particletracing/coordhelpers.cpp particletracing/magneticfield.cpp particletracing/rootfinding.cpp)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
target_include_directories(${PROJECT_NAME} PRIVATE "particletracing" "eigen")
#target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen})


pybind11_add_module(pyparticle particletracing/python.cpp particletracing/particletracing.cpp particletracing/coordhelpers.cpp particletracing/magneticfield.cpp particletracing/rootfinding.cpp)
set_target_properties(pyparticle
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
target_include_directories(pyparticle PRIVATE  "particletracing" "eigen")

#if(OpenMP_CXX_FOUND)
#    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
#    target_link_libraries(profiling_biot_savart PRIVATE OpenMP::OpenMP_CXX)
#endif()

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
