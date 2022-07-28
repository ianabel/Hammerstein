
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
 * y = t^2 + Int[ sin( t )exp( -2s )*[ y( s ) ]^2,{s,-1,1} ]
 */

double f_test1( double t ) {
	return t*t;
};

double g_test1( double s, double y ) {
	return y*y;
};

double K_test1( double x, double s ) {
	return ::sin( x )*::exp( -2.0*s );
};

// g differentiated with respect to its second argument
double g_prime_test1( double s, double y ) {
	return 2.0*y;
}

/*
 * Set up the test problem
 *
 * y = f( t ) + Int[ |t-s|^-1/4*[ y( s ) ]^2,{s,0,1} ]
 *
 * with f( t ) such that y( s ) = s^3/4
 */

double f_test2( double t ) {
	boost::math::quadrature::tanh_sinh<double> integrator( 5, 1e-15 );
	auto integrand = [t]( double s ){
		return ::pow( s, 1.5 )/::pow( ::fabs( t-s ), 0.25 );
	};
	return ::pow( t, 0.75 ) - integrator.integrate( integrand, 0, t ) - integrator.integrate( integrand, t, 1 );

	/* needed for better than 1e-8 precision
	boost::math::quadrature::tanh_sinh<double> integrator; // ( 5, 1e-15 );
	// for s < t - but you can't check, because s-t < 1e-16
	auto tanh_sinh_integrand_1 = [ t ]( double s, double sc ) {
		double retval;
		if ( sc < 0 ) { // sc = 0 - s
			retval = ::pow( s, 1.5 )/::pow( t - s, 0.25 );
		} else { // sc = t - s
			retval = ::pow( s, 1.5 )/::pow( sc, 0.25 );
		}
		return retval;
	};
	// for s > t
	auto tanh_sinh_integrand_2 = [ t ]( double s, double sc ) {
		double retval;
		if ( sc > 0 ) { // sc = 1 - s
			retval = ::pow( s, 1.5 ) / ::pow( s - t, 0.25 );
		} else { // sc = t - s && sc < 0
			retval = ::pow( s, 1.5 )/::pow( -sc, 0.25 );
		}
		return retval;
	};

	if ( t == 0 )
		return -integrator.integrate( tanh_sinh_integrand_2, 0, 1 );
	else if ( t == 1.0 )
		return 1 - integrator.integrate( tanh_sinh_integrand_1, 0, 1 );
	else
		return ::pow( t, 0.75 ) - integrator.integrate( tanh_sinh_integrand_1, 0, t ) - integrator.integrate( tanh_sinh_integrand_2, t, 1 );
	*/
};

double g_test2( double s, double y ) {
	return y*y;
};

double K_test2( double x, double s ) {
	return 1.0/::pow( ::fabs( x - s ), 0.25 );
};

double g_test3( double s, double y ) {
	return y*y;
};

double K_test3( double x, double s ) {
	return ::pow( ::fabs( x - s ), 0.25 );
};

double f_test3( double t ) {
	boost::math::quadrature::tanh_sinh<double> integrator( 5, 1e-15 );
	auto integrand = [t]( double s ){
		return ::pow( s, 3.0/2.0 )*K_test3( t, s );
	};
	return ::pow( t, 0.75 ) - ( integrator.integrate( integrand, 0, t ) + integrator.integrate( integrand, t, 1 ) );
};


double g_gtest_a( double s, double y, double yPrime ) {
	return y*y;
}

double K_gtest_a( double x, double s ) {
	return ::pow( ::fabs( x - s ), 0.25 );
}

double K_prime_gtest_a( double x, double s ) {
	if ( s < x )
		return 0.25*::pow( x - s, -0.75 );
	else if ( s >= x )
		return -0.25*::pow( s - x, -0.75 );
	else
		throw std::runtime_error( "NaN passed to function K_prime_gtest" );
}

double f_gtest_a( double t ) {
	boost::math::quadrature::tanh_sinh<double> integrator( 5, 1e-15 );
	auto integrand = [t]( double s ){
		return ::pow( s, 3.0/2.0 )*K_gtest_a( t, s );
	};
	return ::pow( t, 0.75 ) - ( integrator.integrate( integrand, 0, t ) + integrator.integrate( integrand, t, 1 ) );
};

double f_prime_gtest_a( double t ) {
	boost::math::quadrature::tanh_sinh<double> integrator( 5, 1e-15 );
	auto integrand = [t]( double s ){
		return ::pow( s, 3.0/2.0 )*K_prime_gtest_a( t, s );
	};
	return ( 0.75 )*::pow( t, -0.25 ) - ( integrator.integrate( integrand, 0, t ) + integrator.integrate( integrand, t, 1 ) );
};

double g_gtest( double s, double y, double yPrime ) {
	return y*y/yPrime;
}

double K_gtest( double x, double s ) {
	return ::pow( ::fabs( x - s ), 0.25 );
}

double K_prime_gtest( double x, double s ) {
	if ( s < x )
		return 0.25*::pow( x - s, -0.75 );
	else if ( s >= x )
		return -0.25*::pow( s - x, -0.75 );
	else
		throw std::runtime_error( "NaN passed to function K_prime_gtest" );
}

double f_gtest( double t ) {
	boost::math::quadrature::tanh_sinh<double> integrator( 5, 1e-15 );
	auto integrand = [t]( double s ){
		return ( 4.0/3.0 )*::pow( s, 7.0/4.0 )*K_gtest( t, s );
	};
	return ::pow( t, 0.75 ) - ( integrator.integrate( integrand, 0, t ) + integrator.integrate( integrand, t, 1 ) );
};

double f_prime_gtest( double t ) {
	boost::math::quadrature::tanh_sinh<double> integrator( 5, 1e-15 );
	auto integrand = [t]( double s ){
		return ( 4.0/3.0 )*::pow( s, 7.0/4.0 )*K_prime_gtest( t, s );
	};
	return ( 0.75 )*::pow( t, -0.25 ) - ( integrator.integrate( integrand, 0, t ) + integrator.integrate( integrand, t, 1 ) );
};


int main( int argc, char** argv )
{

	unsigned int N_Intervals = 32;
	unsigned int PolynomialOrder = 3;

	HammersteinEquation Problem1( -1, 1, f_test1, g_test1, K_test1 );

	sundials::Context sunctx;

	void *kinMem = KINCreate( sunctx );


	Problem1.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder );

	sunindextype NDims = Problem1.getDimension();
	N_Vector zDataInit = N_VNew_Serial( NDims, sunctx );


	double C = 1.957783986;
	auto exact_y = []( double C_fac,double t ) {
		return t*t + C_fac*::sin( t );
	};
	auto exact_z = [ & ]( double t ) {
		return g_test1( t, exact_y( C, t ) );
	};

	// Initial Condition near known answer.

	auto initial_y = [&]( double t ){
		return exact_y( 1.8, t );
	};

	Problem1.computeCoefficients( NV_DATA_S( zDataInit ), initial_y );

	KinsolErrorWrapper( KINInit( kinMem, HammersteinEquation::KINSOL_Hammerstein, zDataInit ), "KINInit" );
	KinsolErrorWrapper( KINSetPrintLevel( kinMem, 0 ), "KINSetPrintLevel" );
	KinsolErrorWrapper( KINSetUserData( kinMem, static_cast<void*>( &Problem1 ) ), "KINSetUserData" );

	SUNMatrix Jac = SUNDenseMatrix( NDims, NDims, sunctx );

	SUNLinearSolver LS = SUNLinSol_Dense( zDataInit, Jac, sunctx );

	KinsolErrorWrapper( KINSetLinearSolver( kinMem, LS, Jac ), "KINSetLinearSolver" );

	double ftol = 1.e-6;
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

	Problem1.setzData( zDataInit );

	std::cout << "Test problem 1 (Numerical J): "<< std::endl 
				 << "                       / 1                      " << std::endl
				 << " y(t) = t^2 + sin(t) * |   exp(-2s)*[y(s)]^2 ds " << std::endl
				 << "                       /-1                      " << std::endl
				 << std::endl
				 << " z(t) = (y(t))^2 " << std::endl << std::endl;
	std::cout << "Checking z " << std::endl;

	unsigned int N_Samples = 256;
	Eigen::VectorXd samples( N_Samples + 1 ),exactZ( N_Samples + 1 ),exactY( N_Samples + 1 ),interpolated( N_Samples + 1 );

	for ( Eigen::Index i=0; i<=N_Samples; ++i )
	{
		double x = -1 + 2.0*i/static_cast<double>( N_Samples );
		samples( i ) = Problem1.EvaluateZ( x );
		exactZ( i ) = exact_z( x );
		interpolated( i ) = Problem1.EvaluateY( x );
		exactY( i ) = exact_y( C, x );
	}

	std::cout << "Answer at t = 1.0 should be " << exact_z( 1.0 ) << " and is numerically " << Problem1.EvaluateZ( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << exact_z( 0.0 ) << " and is numerically " << Problem1.EvaluateZ( 0.0 ) << std::endl;
	std::cout << "Answer at t = -1.0 should be " << exact_z( -1.0 ) << " and is numerically " << Problem1.EvaluateZ( -1.0 ) << std::endl;

	std::cout << std::endl;

	 Eigen::VectorXd err = exactZ - samples;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << "Checking y " << std::endl;

	std::cout << "Answer at t = 1.0 should be " << exact_y( C, 1.0 ) << " and is numerically " << Problem1.EvaluateY( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << exact_y( C, 0.0 ) << " and is numerically " << Problem1.EvaluateY( 0.0 ) << std::endl;
	std::cout << "Answer at t = -1.0 should be " << exact_y( C, -1.0 ) << " and is numerically " << Problem1.EvaluateY( -1.0 ) << std::endl;

	std::cout << std::endl;

	err = exactY - interpolated;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	// Reset initial condition
	Problem1.computeCoefficients( NV_DATA_S( zDataInit ), initial_y );

	KinsolErrorWrapper( KINSetJacFn( kinMem, HammersteinEquation::KINSOL_HammersteinJacobian ), "Set Jacobian" );
	Problem1.setgPrime( g_prime_test1 );
	KinsolErrorWrapper( KINSol( kinMem, zDataInit, KIN_LINESEARCH, one, one ), "KINSol" );

	Problem1.setzData( zDataInit );

	std::cout << "Test problem 1 (analytic J): "<< std::endl 
				 << "                       / 1                      " << std::endl
				 << " y(t) = t^2 + sin(t) * |   exp(-2s)*[y(s)]^2 ds " << std::endl
				 << "                       /-1                      " << std::endl
				 << std::endl
				 << " z(t) = (y(t))^2 " << std::endl << std::endl;
	std::cout << "Checking z " << std::endl;

	for ( Eigen::Index i=0; i<=N_Samples; ++i )
	{
		double x = -1 + 2.0*i/static_cast<double>( N_Samples );
		samples( i ) = Problem1.EvaluateZ( x );
		exactZ( i ) = exact_z( x );
		interpolated( i ) = Problem1.EvaluateY( x );
		exactY( i ) = exact_y( C, x );
	}

	std::cout << "Answer at t = 1.0 should be " << exact_z( 1.0 ) << " and is numerically " << Problem1.EvaluateZ( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << exact_z( 0.0 ) << " and is numerically " << Problem1.EvaluateZ( 0.0 ) << std::endl;
	std::cout << "Answer at t = -1.0 should be " << exact_z( -1.0 ) << " and is numerically " << Problem1.EvaluateZ( -1.0 ) << std::endl;

	std::cout << std::endl;

	err = exactZ - samples;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << "Checking y " << std::endl;

	std::cout << "Answer at t = 1.0 should be " << exact_y( C, 1.0 ) << " and is numerically " << Problem1.EvaluateY( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << exact_y( C, 0.0 ) << " and is numerically " << Problem1.EvaluateY( 0.0 ) << std::endl;
	std::cout << "Answer at t = -1.0 should be " << exact_y( C, -1.0 ) << " and is numerically " << Problem1.EvaluateY( -1.0 ) << std::endl;

	std::cout << std::endl;

	err = exactY - interpolated;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;


	std::cout << std::endl << " -------------------------- " << std::endl << std::endl;

	std::cout << "Test problem 2 (numerical J / uniform mesh ): "<< std::endl 
				 << "               / 1                        " << std::endl
				 << " y(t) = f(t) + |   |t-s|^-1/4 [y(s)]^2 ds " << std::endl
				 << "               / 0                        " << std::endl
				 << std::endl
				 << " z(t) = (y(t))^2 " << std::endl;

	HammersteinEquation Problem2( 0, 1, f_test2, g_test2, K_test2 );
	Problem2.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder );
	
	std::cout << "Precomputation done." << std::endl;

	KinsolErrorWrapper( KINSetUserData( kinMem, static_cast<void*>( &Problem2 ) ), "KINSetUserData" );
	KinsolErrorWrapper( KINSetJacFn( kinMem, nullptr ), "Set Jacobian" );
	Problem2.computeCoefficients( NV_DATA_S( zDataInit ), []( double t ){ return ::sqrt( t );} );
	KinsolErrorWrapper( KINSol( kinMem, zDataInit, KIN_LINESEARCH, one, one ), "KINSol" );
	Problem2.setzData( zDataInit );
	
	std::cout << "Checking z " << std::endl;

	for ( Eigen::Index i=0; i<=N_Samples; ++i )
	{
		double x = ( 1.0 * i )/static_cast<double>( N_Samples );
		samples( i ) = Problem2.EvaluateZ( x );
		exactZ( i ) = ::pow( x, 1.5 );
		interpolated( i ) = Problem2.EvaluateY( x );
		exactY( i ) = ::pow( x, 0.75 );
	}

	std::cout << "Answer at t = 1.0 should be " << 1.0  << " and is numerically " << Problem2.EvaluateZ( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.5 should be " << ::pow( 0.5, 1.5 ) << " and is numerically " << Problem2.EvaluateZ( 0.5 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << 0.0  << " and is numerically " << Problem2.EvaluateZ( 0.0 ) << std::endl;

	std::cout << std::endl;

	err = exactZ - samples;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << "Checking y " << std::endl;

	std::cout << "Answer at t = 1.0 should be " << 1.0 << " and is numerically " << Problem2.EvaluateY( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.5 should be " << ::pow( 0.5, 0.75 ) << " and is numerically " << Problem2.EvaluateY( 0.5 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << 0.0 << " and is numerically " << Problem2.EvaluateY( 0.0 ) << std::endl;

	std::cout << std::endl;

	err = exactY - interpolated;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;


	std::cout << "Test problem 2 (numerical J / graded mesh ): "<< std::endl 
				 << "               / 1                        " << std::endl
				 << " y(t) = f(t) + |   |t-s|^-1/4 [y(s)]^2 ds " << std::endl
				 << "               / 0                        " << std::endl
				 << std::endl
				 << " z(t) = (y(t))^2 " << std::endl;

	Problem2.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder, HammersteinEquation::BasisType::DGLegendre, true, 0.75 );
	
	std::cout << "Precomputation done." << std::endl;

	KinsolErrorWrapper( KINSetUserData( kinMem, static_cast<void*>( &Problem2 ) ), "KINSetUserData" );
	KinsolErrorWrapper( KINSetJacFn( kinMem, nullptr ), "Set Jacobian" );
	Problem2.computeCoefficients( NV_DATA_S( zDataInit ), []( double t ){ return ::sqrt( t );} );
	KinsolErrorWrapper( KINSol( kinMem, zDataInit, KIN_LINESEARCH, one, one ), "KINSol" );
	Problem2.setzData( zDataInit );
	
	std::cout << "Checking z " << std::endl;

	for ( Eigen::Index i=0; i<=N_Samples; ++i )
	{
		double x = ( 1.0 * i )/static_cast<double>( N_Samples );
		samples( i ) = Problem2.EvaluateZ( x );
		exactZ( i ) = ::pow( x, 1.5 );
		interpolated( i ) = Problem2.EvaluateY( x );
		exactY( i ) = ::pow( x, 0.75 );
	}

	std::cout << "Answer at t = 1.0 should be " << 1.0  << " and is numerically " << Problem2.EvaluateZ( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.5 should be " << ::pow( 0.5, 1.5 ) << " and is numerically " << Problem2.EvaluateZ( 0.5 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << 0.0  << " and is numerically " << Problem2.EvaluateZ( 0.0 ) << std::endl;

	std::cout << std::endl;

	err = exactZ - samples;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << "Checking y " << std::endl;

	std::cout << "Answer at t = 1.0 should be " << 1.0 << " and is numerically " << Problem2.EvaluateY( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.5 should be " << ::pow( 0.5, 0.75 ) << " and is numerically " << Problem2.EvaluateY( 0.5 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << 0.0 << " and is numerically " << Problem2.EvaluateY( 0.0 ) << std::endl;

	std::cout << std::endl;

	err = exactY - interpolated;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << "Test problem 3 (numerical J / uniform mesh ): "<< std::endl 
				 << "                / 1                        " << std::endl
				 << " y(t) = f(t) +  |   |t-s|^1/4 [y(s)]^2 ds " << std::endl
				 << "                / 0                        " << std::endl
				 << std::endl
				 << " z(t) = (y(t))^2  ; f s.t. y = t^3/4 is a solution" << std::endl;

	HammersteinEquation Problem3( 0, 1, f_test3, g_test3, K_test3 );
	Problem3.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder );
	
	std::cout << "Precomputation done." << std::endl;

	KinsolErrorWrapper( KINSetUserData( kinMem, static_cast<void*>( &Problem3 ) ), "KINSetUserData" );
	KinsolErrorWrapper( KINSetJacFn( kinMem, nullptr ), "Set Jacobian" );
	Problem3.computeCoefficients( NV_DATA_S( zDataInit ), []( double t ){ return ::sqrt( t );} );
	KinsolErrorWrapper( KINSol( kinMem, zDataInit, KIN_LINESEARCH, one, one ), "KINSol" );
	Problem3.setzData( zDataInit );
	
	std::cout << "Checking z " << std::endl;

	for ( Eigen::Index i=0; i<=N_Samples; ++i )
	{
		double x = ( 1.0 * i )/static_cast<double>( N_Samples );
		samples( i ) = Problem3.EvaluateZ( x );
		exactZ( i ) = ::pow( x, 1.5 );
		interpolated( i ) = Problem3.EvaluateY( x );
		exactY( i ) = ::pow( x, 0.75 );
	}

	std::cout << "Answer at t = 1.0 should be " << 1.0  << " and is numerically " << Problem3.EvaluateZ( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.5 should be " << ::pow( 0.5, 1.5 ) << " and is numerically " << Problem3.EvaluateZ( 0.5 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << 0.0  << " and is numerically " << Problem3.EvaluateZ( 0.0 ) << std::endl;

	std::cout << std::endl;

	err = exactZ - samples;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << "Checking y " << std::endl;

	std::cout << "Answer at t = 1.0 should be " << 1.0 << " and is numerically " << Problem3.EvaluateY( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.5 should be " << ::pow( 0.5, 0.75 ) << " and is numerically " << Problem3.EvaluateY( 0.5 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << 0.0 << " and is numerically " << Problem3.EvaluateY( 0.0 ) << std::endl;

	std::cout << std::endl;

	err = exactY - interpolated;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << std::endl << " --------------------------------------------------------------------- " << std::endl << std::endl;


	KINFree( &kinMem );
	kinMem = KINCreate( sunctx );

	std::cout << "Test problem 4a (numerical J / uniform mesh ): "<< std::endl 
				 << "                / 1                               " << std::endl
				 << " y(t) = f(t) +  |   |t-s|^1/4 [y(s)]^2 ds " << std::endl
				 << "                / 0                               " << std::endl
				 << std::endl
				 << " z(t) = (y(t))^2  ; f s.t. y = t^3/4 is a solution" << std::endl;

	GeneralizedHammersteinEquation GProblem_a( 0, 1, f_gtest_a, g_gtest_a, K_gtest_a, f_prime_gtest_a, K_prime_gtest_a );
	GProblem_a.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder );
	
	std::cout << "Precomputation done." << std::endl;

	KinsolErrorWrapper( KINInit( kinMem, GeneralizedHammersteinEquation::KINSOL_GeneralizedHammerstein, zDataInit ), "KINInit" );
	KinsolErrorWrapper( KINSetUserData( kinMem, static_cast<void*>( &GProblem_a ) ), "KINSetUserData" );
	KinsolErrorWrapper( KINSetLinearSolver( kinMem, LS, Jac ), "KINSetLinearSolver" );
	KinsolErrorWrapper( KINSetMaxSetupCalls( kinMem, 2 ), "KINSetMaxSetupCalls" );

	KinsolErrorWrapper( KINSetFuncNormTol( kinMem, ftol ), "KINSetFuncNormTol" );
	KinsolErrorWrapper( KINSetScaledStepTol( kinMem, scstol ), "KINSetScaledStepTol" );
	KinsolErrorWrapper( KINSetRelErrFunc( kinMem, jtol ), "KINSetRelErrFunc" );

	KinsolErrorWrapper( KINSetJacFn( kinMem, nullptr ), "Set Jacobian" );
	GProblem_a.computeCoefficients( NV_DATA_S( zDataInit ), []( double t ){ return ::sqrt( t );} );
	KinsolErrorWrapper( KINSol( kinMem, zDataInit, KIN_LINESEARCH, one, one ), "KINSol" );
	GProblem_a.setzData( zDataInit );
	
	std::cout << "Checking z " << std::endl;

	for ( Eigen::Index i=0; i<=N_Samples; ++i )
	{
		double x = ( 1.0 * i )/static_cast<double>( N_Samples );
		samples( i ) = GProblem_a.EvaluateZ( x );
		exactZ( i ) = ::pow( x, 1.5 );
		interpolated( i ) = GProblem_a.EvaluateY( x );
		exactY( i ) = ::pow( x, 0.75 );
	}

	std::cout << "Answer at t = 1.0 should be " << 1.0  << " and is numerically " << GProblem_a.EvaluateZ( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.5 should be " << ::pow( 0.5, 1.5 ) << " and is numerically " << GProblem_a.EvaluateZ( 0.5 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << 0.0  << " and is numerically " << GProblem_a.EvaluateZ( 0.0 ) << std::endl;

	std::cout << std::endl;

	err = exactZ - samples;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << "Checking y " << std::endl;

	std::cout << "Answer at t = 1.0 should be " << 1.0 << " and is numerically " << GProblem_a.EvaluateY( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.5 should be " << ::pow( 0.5, 0.75 ) << " and is numerically " << GProblem_a.EvaluateY( 0.5 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << 0.0 << " and is numerically " << GProblem_a.EvaluateY( 0.0 ) << std::endl;

	std::cout << std::endl;

	err = exactY - interpolated;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << "Test problem 4b (numerical J / uniform mesh ): "<< std::endl 
				 << "                / 1                               " << std::endl
				 << " y(t) = f(t) +  |   |t-s|^1/4 [y(s)^2/y'(s)] ds " << std::endl
				 << "                / 0                               " << std::endl
				 << std::endl
				 << " z(t) = (y(t))^2/(y'(t))  ; f s.t. y = t^3/4 is a solution" << std::endl;

	GeneralizedHammersteinEquation GProblem( 0, 1, f_gtest, g_gtest, K_gtest, f_prime_gtest, K_prime_gtest );
	GProblem.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder );
	
	std::cout << "Precomputation done." << std::endl;

	KinsolErrorWrapper( KINSetUserData( kinMem, static_cast<void*>( &GProblem ) ), "KINSetUserData" );


	KinsolErrorWrapper( KINSetJacFn( kinMem, nullptr ), "Set Jacobian" );
	GProblem.computeCoefficients( NV_DATA_S( zDataInit ), []( double t ){ return ::pow( t, 0.6 );}, []( double t ){return 0.6/::pow( t, 0.4 );} );
	
	KinsolErrorWrapper( KINSol( kinMem, zDataInit, KIN_LINESEARCH, one, one ), "KINSol" );
	GProblem.setzData( zDataInit );
	
	std::cout << "Checking z " << std::endl;

	for ( Eigen::Index i=0; i<=N_Samples; ++i )
	{
		double x = ( 1.0 * i )/static_cast<double>( N_Samples );
		samples( i ) = GProblem.EvaluateZ( x );
		exactZ( i ) = ( 4.0/3.0 )*::pow( x, 7.0/4.0 );
		interpolated( i ) = GProblem.EvaluateY( x );
		exactY( i ) = ::pow( x, 0.75 );
	}

	std::cout << "Answer at t = 1.0 should be " << ( 4.0/3.0 )  << " and is numerically " << GProblem.EvaluateZ( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.5 should be " << ( 4.0/3.0 )*::pow( 0.5, 7.0/4.0 ) << " and is numerically " << GProblem.EvaluateZ( 0.5 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << 0.0  << " and is numerically " << GProblem.EvaluateZ( 0.0 ) << std::endl;

	std::cout << std::endl;

	err = exactZ - samples;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;

	std::cout << "Checking y " << std::endl;

	std::cout << "Answer at t = 1.0 should be " << 1.0 << " and is numerically " << GProblem.EvaluateY( 1.0 ) << std::endl;
	std::cout << "Answer at t = 0.5 should be " << ::pow( 0.5, 0.75 ) << " and is numerically " << GProblem.EvaluateY( 0.5 ) << std::endl;
	std::cout << "Answer at t = 0.0 should be " << 0.0 << " and is numerically " << GProblem.EvaluateY( 0.0 ) << std::endl;

	std::cout << std::endl;

	err = exactY - interpolated;
	std::cout << "\tL_1 error = " << err.lpNorm<1>() << std::endl;
	std::cout << "\tL_2 error = " << err.lpNorm<2>() << std::endl;
	std::cout << "\tL_Inf error = " << err.lpNorm<Eigen::Infinity>() << std::endl;




	return 0;
}

