// osqp-eigen
#include "OsqpEigen/OsqpEigen.h"

// eigen
#include <Eigen/Dense>

#include <iostream>
#include <cmath>

/* Problem Definition (same as osqp_demo)*/
/*
  minimize sqrt(x2)
  subject to x2>=0, x2>=(2x1)^3, x2>=(-x1+1)^3

  Dim of x: 2
  number of constraints: 3


 */

//all the vectors should be vertical

const int variable_num = 2;
const int constraint_num = 3;

double autoScale(const Eigen::VectorXd& s, const Eigen::VectorXd& q)
{
  double s_norm2 = s.transpose() * s;
  double y_norm2 = q.transpose() * q;
  double ys = std::abs(q.transpose() * s);
  if (ys == 0.0 || y_norm2 == 0 || s_norm2 == 0) {
    return 1.0;
  } else {
    return y_norm2 / ys;
  }
}

Eigen::MatrixXd updateHessianBFGS(Eigen::MatrixXd h_prev, const Eigen::VectorXd& x, const Eigen::VectorXd& x_prev, const Eigen::VectorXd& lambda_prev, const Eigen::VectorXd& grad_f, const Eigen::VectorXd& grad_f_prev, const Eigen::MatrixXd& grad_g, const Eigen::MatrixXd& grad_g_prev)
{
  static bool first_update = true;
  Eigen::VectorXd s = x - x_prev;
  Eigen::VectorXd q = (grad_f - grad_f_prev) + (grad_g - grad_g_prev).transpose() * lambda_prev;

  double qs = q.transpose() * s;
  Eigen::VectorXd Hs = h_prev * s;
  double sHs = s.transpose() * h_prev * s;

  if (sHs < 0.0 || first_update) {
    h_prev = Eigen::MatrixXd::Identity(variable_num, variable_num) * autoScale(s, q);
    Hs = h_prev * s;
    sHs = s.transpose() * h_prev * s;
    first_update = false;
  }

  if (qs < 0.2 * sHs) {
    double update_factor = (1 - 0.2) / (1 - qs / sHs);
    q = update_factor * q + (1 - update_factor) * Hs;
    qs = q.transpose() * s;
  }

  auto h = h_prev + (q * q.transpose()) / qs - (Hs * Hs.transpose()) / sHs;
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

double costFunction(const Eigen::VectorXd& x)
{
  return std::sqrt(x(1));
}

Eigen::VectorXd costFunctionGrad(const Eigen::VectorXd& x)
{
  Eigen::VectorXd grad;
  grad.resize(variable_num);
  grad(0) = 0.0;
  grad(1) = 0.5 / std::sqrt(x(1));
  return grad;
}

//b(x) >= 0
Eigen::VectorXd inequalityConstraint(const Eigen::VectorXd& x)
{
  Eigen::VectorXd b;
  b.resize(constraint_num);
  b(0) = x(1);
  b(1) = x(1) - 8 * x(0) * x(0) * x(0);
  b(2) = x(1) - (-x(0) + 1) * (-x(0) + 1) * (-x(0) + 1);

  return b;
}

Eigen::MatrixXd inequalityConstraintGrad(const Eigen::VectorXd& x)
{
  Eigen::MatrixXd grad;
  grad.resize(constraint_num, variable_num);
  grad(0, 0) = 0.0;
  grad(0, 1) = 1.0;
  grad(1, 0) = -24 * x(0) * x(0);
  grad(1, 1) = 1.0;
  grad(2, 0) = 3 * (-x(0) + 1) * (-x(0) + 1);
  grad(2, 1) = 1.0;

  return grad;
}

void QP(const Eigen::MatrixXd hessian_dense, const Eigen::VectorXd& x, Eigen::VectorXd& x_d, Eigen::VectorXd& dual_solution)
{
  OsqpEigen::Solver solver;
  Eigen::SparseMatrix<double> hessian = convertSparseMatrix(hessian_dense);
  Eigen::VectorXd gradient = costFunctionGrad(x); //q
  Eigen::SparseMatrix<double> linearMatrix = convertSparseMatrix(inequalityConstraintGrad(x)); //A
  Eigen::VectorXd lowerBound = -inequalityConstraint(x); //l
  Eigen::VectorXd upperBound; //u
  upperBound.resize(constraint_num);
  upperBound << 1e6, 1e6, 1e6;

  solver.settings()->setWarmStart(true);

  solver.data()->setNumberOfVariables(2);
  solver.data()->setNumberOfConstraints(constraint_num);
  if(!solver.data()->setHessianMatrix(hessian)) return;
  if(!solver.data()->setGradient(gradient)) return;
  if(!solver.data()->setLinearConstraintsMatrix(linearMatrix)) return;
  if(!solver.data()->setLowerBound(lowerBound)) return;
  if(!solver.data()->setUpperBound(upperBound)) return;

  if(!solver.initSolver()) return;

  solver.solve();

  x_d = solver.getSolution();
  dual_solution = solver.getDualSolution();
}

int main()
{
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity(variable_num, variable_num);
  Eigen::VectorXd x(variable_num);
  x(0) = 10;
  x(1) = 10;
  Eigen::VectorXd x_prev(variable_num);
  Eigen::VectorXd dual_solution(constraint_num);
  Eigen::VectorXd x_d(variable_num);

  std::vector<Eigen::VectorXd> x_log;

  for (size_t i = 0; i < 10000; ++i) {
    x_prev = x;
    QP(hessian, x, x_d, dual_solution);

    if (x_d.norm() > 0.1) {
      x_d = x_d / x_d.norm() * 0.1;
    }
    x = x + x_d;

    std::cout << "iteration: " << i << std::endl;
    std::cout << "x: " << std::endl << x << std::endl;
    std::cout << "cost: " << costFunction(x) << std::endl;
    std::cout << "hessian: " << std::endl << hessian << std::endl;

    hessian = updateHessianBFGS(hessian, x, x_prev, dual_solution, costFunctionGrad(x), costFunctionGrad(x_prev), inequalityConstraintGrad(x), inequalityConstraintGrad(x_prev));

    x_log.push_back(x);
    if (x_d.norm() < 0.001)
     break;
    //    if (costFunction(x) < 0.545) break;
  }
  // for (auto& xx : x_log) {
  //   std::cout << xx[0] << " " << xx[1] << std::endl;
  // }
  return 0;
}
