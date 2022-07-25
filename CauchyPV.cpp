#include <functional>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <iostream>

/*
 * Use a routine to compute Cauchy principle value integrals
 */

#include "HammersteinEquation.hpp"

/*
	Compute 
	      / b    f( x )
	I =PV |    --------- dx
	      / a   x - tau
	
	by the method of [ P. Keller, I. Wrobel / Journal of Computational and Applied Mathematics 294 (2016) 323–341 ]

	I = f( tau ) log[ ( b - tau )/( tau - a ) ] + I[ g( x ), {x,a,tau-delta} ] + I[ g( x ), {x,tau+delta,b} ] + I[ h( x ), {x, 0, delta} ]

	g( x ) = ( f( x )-f( tau ) )/( x - tau );

	h( x ) = (  f( tau + x ) - f( tau - x ) )/x;


 */
double CauchyPV( std::function<double( double )> f, double a, double b, double tau)
{
	double delta;

	double I;

	auto g = [ & ]( double x ){
		return ( f( x ) - f( tau ) )/( x - tau );
	};

	auto h = [ & ]( double x ){
		return ( f( tau + x ) - f( tau - x ) )/( x );
	};

	boost::math::quadrature::tanh_sinh<double> Int( 5, 1e-15 );

	I = f( tau )*::log( ( b - tau )/( tau - a ) );

	if ( tau - a < b - tau ) {
		delta = tau - a;
		I += Int.integrate( g, tau + delta, b );
	} else if ( b - tau < tau - a ) {
		delta = b - tau;
		I += Int.integrate( g, a, tau - delta );
	} else if ( b - tau == tau - a ) {
		delta = b - tau;
		I = 0;
	}
	
	I += Int.integrate( h, 0, delta );

	return I;

}


int main( int, char** )
{
	unsigned int n = 5;
	// PV[ cos(n x)/(cos(x) - cos(theta)), {x,0,pi} ] = pi * sin(n theta)/sin(theta)
	// 
	// we rewrite the integrand as
	//
	// f(x) * ( a/(x-t) - a/(x-t) + 1/(cos(x)-cos(t)) )
	// 
	// with f(x) = cos(n x)
	// a = -Cosec(theta)

	double t = M_PI*0.15;
	double a = -1.0/::sin( t );
	auto cauchy_f = [ & ]( double x ){
		return ::cos( n*x ) * ( a );
	};
	auto bounded_f = [ & ]( double x ){
		double f = ::cos( n * x );
		if (  ::fabs( x - t ) > 1e-4 ) {
			return f*( 1.0/( ::cos( x ) - ::cos( t ) ) - a / ( x - t ) );
		} else {
			double cot = 1.0/::tan( t );
			double csc = 1.0/::sin( t );
			double delta = x-t;
			return f * ( 0.5*cot*csc - ( 1.0/12.0 )*( 2.0 + 3.0*cot*cot )*csc*delta + ( 1.0/8.0 )*( cot*csc*csc*csc )*delta*delta
					             -( 1.0/720.0 )*( 14.0 + 60.0*cot*cot + 45.0*cot*cot*cot*cot )*csc*delta*delta*delta );
		}
	};
	boost::math::quadrature::tanh_sinh<double> Int( 5, 1e-15 );
	double I = CauchyPV( cauchy_f, 0.0, M_PI, t ) + Int.integrate( bounded_f, 0.0, M_PI );
	std::cerr << std::setprecision( 16 );
	std::cerr << "Integral = " << I << std::endl;
	std::cerr << "Glauert  = " << M_PI*::sin( n*t )/::sin( t ) << std::endl;
	return 0;
}
