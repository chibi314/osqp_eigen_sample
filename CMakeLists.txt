cmake_minimum_required(VERSION 2.8.3)
project(osqp_eigen_sample)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED)

find_package(OsqpEigen)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES osqp_eigen_sample
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(sample_qp src/sample_qp.cpp)
target_link_libraries(sample_qp OsqpEigen::OsqpEigen osqp::osqp)

add_executable(sample_slsqp src/sample_slsqp.cpp)
target_link_libraries(sample_slsqp OsqpEigen::OsqpEigen osqp::osqp)
