// osqp-eigen
#include "OsqpEigen/OsqpEigen.h"

// eigen
#include <Eigen/Dense>

#include <iostream>

/* Problem Definition (same as osqp_demo)*/
/*
  minimize sqrt(x2)
  subject to x2>=0, x2>=(2x1)^3, x2>=(-x1+1)^3

  Dim of x: 2
  number of constraints: 3


 */

//all the vectors should be vertical
Eigen::MatrixXd updateHessianBFGS(const Eigen::MatrixXd& h_prev, const Eigen::VectorXd& x, const Eigen::VectorXd& x_prev, const Eigen::VectorXd& lambda_prev, const Eigen::VectorXd& grad_f, const Eigen::VectorXd& grad_f_prev, const Eigen::MatrixXd& grad_g, const Eigen::MatrixXd& grad_g_prev)
{
  auto s = x - x_prev;
  auto q = (grad_f - grad_f_prev) + lambda_prev.transpose() * (grad_g - grad_g_prev);

  auto h = h_prev + (q * q.transpose()) / (q.transpose() * s) - (h_prev * s * s.transpose() * h_prev.transpose()) / (s.transpose() * h_prev * s);

  return h;
}

Eigen::SparseMatrix<double> convertSparseMatrix(const Eigen::MatrixXd& mat)
{
  Eigen::SparseMatrix<double> ret;
  ret.resize(mat.rows(), mat.cols());
  for (size_t i = 0; i < ret.rows(); ++i) {
    for (size_t j = 0; j < ret.cols(); ++j) {
      if (mat(i, j) != 0.0) {
        ret.insert(i, j) = mat(i, j);
      }
    }
  }
  return ret;
}

void QP(const Eigen::MatrixXd& hessian_dense)
{
  const int N = 2;
  const int M = 3;
  OsqpEigen::Solver solver;
  Eigen::SparseMatrix<double> hessian = convertSparseMatrix(hessian_dense); //P
  Eigen::VectorXd gradient = getCostFunctionGrad; //q
  Eigen::SparseMatrix<double> linearMatrix; //A
  Eigen::VectorXd lowerBound; //l
  Eigen::VectorXd upperBound; //u

}

int main()
{
  const int var_num = 2;
  const int constraint_num = 3;
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
