cmake_minimum_required(VERSION 3.8)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)

add_library(CorrFunc_evaluator STATIC "external/CorrFunc_evaluator.cpp")
set_property(TARGET CorrFunc_evaluator PROPERTY COMPILE_FLAGS "-fPIC")
set(EXTRA_LIBS ${EXTRA_LIBS} CorrFunc_evaluator)

set(CMAKE_SWIG_FLAGS "")

set_source_files_properties(Reader.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(Reader.i PROPERTIES SWIG_FLAGS "-includeall")
swig_add_library(Reader LANGUAGE python SOURCES Reader.i Reader.cpp)
swig_link_libraries(Reader ${EXTRA_LIBS} ${PYTHON_LIBRARIES})