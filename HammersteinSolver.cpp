
#include <kinsol/kinsol.h>             /* access to KINSOL func., consts. */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector       */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */

#include <iostream>
#include <cmath>

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>

#include "HammersteinEquation.hpp"
#include "GeneralizedHammersteinEq.hpp"
#include "SingularHammersteinEq.hpp"

void KinsolErrorWrapper( int errorFlag, std::string&& fName )
{
	if ( errorFlag == KIN_SUCCESS )
		return;
	else
	{
		throw std::runtime_error( "Error " + std::to_string( errorFlag ) + " returned from KINSol function: " + fName );
	}
}

/*
 * Code for solving nonlinear integral equations of the Hammerstein form
 *                    / b
 * y( x ) = f( x ) +  |   ds K( x, s ) g( s, y( s ) )    ( 1 )
 *                    / a
 *
 * by the method of Kumar and Sloan [ mathematics of computation volume 48. number 178 april 1987. pages 585-593 ]
 * using the collocation points of Kumar [ SIAM Journal on Numerical Analysis, Vol. 25, No. 2 (Apr., 1988), pp. 328-341 ]
 *
 */


/*
 * Set up the test problem
 *
 * y = f( t ) + PVInt[ y(s)/( 2*pi*t( t-s ) ),{s,-1,1} ]
 *
 * with f( t ) such that y( s ) = |s| is the exact answer
 * 
 * Namely f( x ) = |x| - ( 1/2pi )*log( ( 1-x^2 )/x^2 )
 *
 * This has a *zero* K_weak ( aka K )
 * and K_residue( t,s ) = 1/( 2*pi*t )
 */


double g_test( double s, double y ) {
	return y;
}

double K_test( double x, double s ) {
	return 0;
}

double K_res( double x, double s ) {
	return ( 1.0 / ( 2.0*M_PI * x ) );
}

double y_star( double t ) {
	return ::fabs( t );
}

double f_test( double t ) {
	return y_star( t ) - ( 1/( 2.0*M_PI ) )*::log( ( 1 - t*t )/( t*t ) );
};


int main( int argc, char** argv )
{

	unsigned int N_Intervals = 16;
	unsigned int PolynomialOrder = 2;

	SingularHammersteinEquation TestProblem( -1, 1, f_test, g_test, K_test, K_res );

	sundials::Context sunctx;

	void *kinMem = KINCreate( sunctx );


	TestProblem.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder );

	sunindextype NDims = TestProblem.getDimension();
	N_Vector zDataInit = N_VNew_Serial( NDims, sunctx );

	// Initial Condition near known answer.

	auto initial_y = [&]( double t ){
		return ::pow( t, 4 );
	};

	TestProblem.computeCoefficients( NV_DATA_S( zDataInit ), initial_y );

	KinsolErrorWrapper( KINInit( kinMem, SingularHammersteinEquation::KINSOL_Hammerstein, zDataInit ), "KINInit" );
	KinsolErrorWrapper( KINSetPrintLevel( kinMem, 0 ), "KINSetPrintLevel" );
	KinsolErrorWrapper( KINSetUserData( kinMem, static_cast<void*>( &TestProblem ) ), "KINSetUserData" );

	SUNMatrix Jac = SUNDenseMatrix( NDims, NDims, sunctx );

	SUNLinearSolver LS = SUNLinSol_Dense( zDataInit, Jac, sunctx );

	KinsolErrorWrapper( KINSetLinearSolver( kinMem, LS, Jac ), "KINSetLinearSolver" );

	double ftol = 1.e-8;
	double scstol = 1.e-8;
	double jtol = 1.e-8;

	KinsolErrorWrapper( KINSetFuncNormTol( kinMem, ftol ), "KINSetFuncNormTol" );
	KinsolErrorWrapper( KINSetScaledStepTol( kinMem, scstol ), "KINSetScaledStepTol" );
	KinsolErrorWrapper( KINSetRelErrFunc( kinMem, jtol ), "KINSetRelErrFunc" );


	N_Vector zero = N_VNew_Serial( NDims, sunctx );
	N_VConst( 0.0, zero );

	N_Vector one = N_VNew_Serial( NDims, sunctx );
	N_VConst( 1.0, one );

	KinsolErrorWrapper( KINSetConstraints( kinMem, zero ), "KINSetConstraints" );
	KinsolErrorWrapper( KINSetMaxSetupCalls( kinMem, 2 ), "KINSetMaxSetupCalls" );

	KinsolErrorWrapper( KINSol( kinMem, zDataInit, KIN_LINESEARCH, one, one ), "KINSol" );

	TestProblem.setzData( zDataInit );

	std::cout << "Test problem 1 (Numerical J): "<< std::endl 
				 << "                    / 1   y(s)       " << std::endl
				 << " y(t) = f(t) + P.V. |   -------------  ds " << std::endl
				 << "                    /-1  2*pi*t*(s-t) " << std::endl
				 << std::endl
				 << " z(t) = (y(t))^3 and f such that the answer is y = t^6 " << std::endl << std::endl;
	std::cout << "Checking z " << std::endl;

	unsigned int N_Samples = 13;
	Eigen::VectorXd samples( N_Samples + 1 ),exactZ( N_Samples + 1 ),exactY( N_Samples + 1 ),interpolated( N_Samples + 1 );

	auto exact_z = [ & ]( double t ){
		return g_test( t, y_star( t ) );
	};

	for ( Eigen::Index i=0; i<=N_Samples; ++i )
	{
		double x = -1 + 2.0*i/static_cast<double>( N_Samples );
		samples( i ) = TestProblem.EvaluateZ( x );
		exactZ( i ) = exact_z( x );
		interpolated( i ) = TestProblem.EvaluateY( x );
		exactY( i ) = y_star( x );
	}

	std::cout << "Answer at t = 0.9 should be " << exact_z( 0.9 ) << " and is numerically " << TestProblem.EvaluateZ( 0.9 ) << std::endl;
	std::cout << "Answer at t = 0.45 should be " << exact_z( 0.45 ) << " and is numerically " << TestProblem.EvaluateZ( 0.45 ) << std::endl;
	std::cout << "Answer at t = -0.45 should be " << exact_z( -0.45 ) << " and is numerically " << TestProblem.EvaluateZ( -0.45 ) << std::endl;
	std::cout << "Answer at t = -0.9 should be " << exact_z( -0.9 ) << " and is numerically " << TestProblem.EvaluateZ( -0.9 ) << std::endl;

	std::cout << std::endl;

	 Eigen::VectorXd err = exactZ - samples;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << "Checking y " << std::endl;

	std::cout << "Answer at t = 0.9 should be " << y_star( 0.9 ) << " and is numerically " << TestProblem.EvaluateY( 0.9 ) << std::endl;
	std::cout << "Answer at t = 0.45 should be " << y_star( 0.45 ) << " and is numerically " << TestProblem.EvaluateY( 0.45 ) << std::endl;
	std::cout << "Answer at t = -0.45 should be " << y_star( -0.45 ) << " and is numerically " << TestProblem.EvaluateY( -0.45 ) << std::endl;
	std::cout << "Answer at t = -0.9 should be " << y_star( -0.9 ) << " and is numerically " << TestProblem.EvaluateY( -0.9 ) << std::endl;
	std::cout << std::endl;

	err = exactY - interpolated;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << f_test( 0.2 ) << std::endl;
	return 0;
}

