cmake_minimum_required(VERSION 3.0)
set(PROJECT_NAME Iviso)
set(PROJECT_TEST_NAME ${PROJECT_NAME}Test)
set(GTEST_LIBS_DIR ext/gTest/lib/)

project(${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
find_package(OpenCV REQUIRED)

include_directories(include ext/gTest/include ext/mockitopp/include)

file(GLOB SOURCE_FILES "src/*.cpp")
file(GLOB TEST_FILES "tests/*.cpp")
file(GLOB GTEST_LIBS "${GTEST_LIBS_DIR}*.a")

#Build Library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

add_executable(Scratchpad scratchpad/main.cpp)
target_link_libraries(Scratchpad
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        m)

add_custom_command(
        TARGET Scratchpad
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/TestData $<TARGET_FILE_DIR:Scratchpad>/TestData)

#Build Tests
add_executable(${PROJECT_TEST_NAME} ${TEST_FILES})
add_dependencies(${PROJECT_TEST_NAME} ${PROJECT_NAME}) #Ensure Library is built before tests
target_link_libraries(${PROJECT_TEST_NAME}
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${GTEST_LIBS}
        pthread)