#include <functional>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <iostream>

/*
 * Check the accuracy of the Green's function
 */

#include "GreensFunction.hpp"



int main( int, char** )
{

	std::cout << std::setprecision( 10 );
	std::cout << GradShafranovGreensFunction1D( 1.0, 1.001 ) << std::endl;
	std::cout << GradShafranovGreensFunction1D( 1.0, 1.001, -1e-3 ) << std::endl;

	std::cout << GradShafranovGreensFunction1D( 1.0, 0.999 ) << std::endl;
	std::cout << GradShafranovGreensFunction1D( 1.0, 0.999, 1e-3 ) << std::endl;

	std::cout << GradShafranovGreensFunction1D( 1.0, 1.00001 ) << std::endl;
	std::cout << GradShafranovGreensFunction1D( 1.0, 1.00001, -1e-5 ) << std::endl;
	
	std::cout << GradShafranovGreensFunction1D( 1.0, 0.99999 ) << std::endl;
	std::cout << GradShafranovGreensFunction1D( 1.0, 0.99999, 1e-5 ) << std::endl;
	
	std::cout << GradShafranovGreensFunction1D( 1.0, 1.0 + 1.0e-7 ) << std::endl;
	std::cout << GradShafranovGreensFunction1D( 1.0, 1.0 + 1.0e-7, -1e-7 ) << std::endl;

	std::cout << GradShafranovGreensFunction1D( 1.0, 1.0 + 1.0e-8, -1e-8 ) << std::endl;
	
	std::cout << GradShafranovGreensFunction1D( 1.0, 1.0 + 1.0e-20, -1e-20 ) << std::endl;
	std::cout << "N.B. 1 + 10^{-20] - 1 = " << ( 1.0 + 1.0e-20 ) - 1.0 << std::endl;

	std::cout << DerivativeGreensFunction1D( 1.0, 1.0002 ) << std::endl;
	std::cout << DerivativeGreensFunction1D_Residue( 1.0002 )/( -0.0002 ) + DerivativeGreensFunction1D_Weak( -0.0002, 1.0002 ) << std::endl;
	double delta = -1e-8;
	std::cout << DerivativeGreensFunction1D_Residue( 1 - delta )/( delta ) + DerivativeGreensFunction1D_Weak( delta, 1 - delta ) << std::endl;

	return 0;
}
