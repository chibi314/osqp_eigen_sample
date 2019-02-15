// osqp-eigen
#include "OsqpEigen/OsqpEigen.h"

// eigen
#include <Eigen/Dense>

#include <iostream>

/* Problem Definition (same as osqp_demo)*/
/*
  minimize 0.5 x^T P x + q^T x
  subject to l <= A_c x <= u

  Dim of x: 2
  number of constraints: 3

  P:
  |4 1|
  |1 2|

  q:
  |1 1|

  A:
  |1 1|
  |1 0|
  |0 1|

  l:
  |1 0 0|

  u:
  |1 0.6999999999999 0.69999999999999|
  

 */



int main()
{
  const int var_num = 2;
  const int constraint_num = 2;
  OsqpEigen::Solver solver;
  Eigen::SparseMatrix<double> hessian; //P
  Eigen::VectorXd gradient; //q
  Eigen::SparseMatrix<double> linearMatrix; //A
  Eigen::VectorXd lowerBound; //l
  Eigen::VectorXd upperBound; //u

  hessian.resize(2, 2);
  hessian.insert(0, 0) = 4.0;
  hessian.insert(0, 1) = 1.0;
  hessian.insert(1, 0) = 1.0;
  hessian.insert(1, 1) = 2.0;

  gradient.resize(2);
  gradient << 1.0, 1.0;

  // linearMatrix.resize(4, 2);
  // linearMatrix.insert(0, 0) = 1.0;
  // linearMatrix.insert(1, 1) = 1.0;
  // linearMatrix.insert(2, 0) = -1.0;
  // linearMatrix.insert(3, 1) = -1.0;

  // lowerBound.resize(4);
  // lowerBound << 0.1, 0.1, -1.0, -1.0;

  // upperBound.resize(4);
  // upperBound << 1e6, 1e6, 1e6, 1e6;


  linearMatrix.resize(2, 2);
  linearMatrix.insert(0, 0) = 1.0;
  linearMatrix.insert(1, 1) = 1.0;

  lowerBound.resize(2);
  lowerBound << 0.1, 0.1;

  upperBound.resize(2);
  upperBound << 1.0, 1.0;



  std::cout << hessian << std::endl;
  std::cout << gradient << std::endl;
  std::cout << linearMatrix << std::endl;

  solver.settings()->setWarmStart(true);

  solver.data()->setNumberOfVariables(2);
  solver.data()->setNumberOfConstraints(constraint_num);
  if(!solver.data()->setHessianMatrix(hessian)) return 1;
  if(!solver.data()->setGradient(gradient)) return 1;
  if(!solver.data()->setLinearConstraintsMatrix(linearMatrix)) return 1;
  if(!solver.data()->setLowerBound(lowerBound)) return 1;
  if(!solver.data()->setUpperBound(upperBound)) return 1;

  if(!solver.initSolver()) return 1;

  solver.solve();

  auto solution = solver.getSolution();
  auto dual_solution = solver.getDualSolution();

  std::cout << "solution: " << solution << std::endl;
  std::cout << "dual_solution: " << dual_solution << std::endl;

  return 0;
}
