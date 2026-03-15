cmake_minimum_required(VERSION 3.16)
project(GuidedLinearUpsampling)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
set(OpenCV_ROOT "/Users/guoxintao/Downloads/opencv-4.12.0/opencv-4.12.0/install")
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

# Create executable
add_executable(GLU 
    src/main.cpp
    src/glu.cpp
)

target_link_libraries(GLU ${OpenCV_LIBS})

# Copy test images to build directory
add_custom_command(TARGET GLU POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/images
    $<TARGET_FILE_DIR:GLU>/images
)