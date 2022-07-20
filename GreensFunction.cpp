
#include "GreensFunction.hpp"

#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <cmath>
#include <numbers>

double GradShafranovGreensFunction( double R, double R_star, double Z, double Z_star )
{
	double answer = std::numbers::inv_pi / 2.0;


	double k_squared = 4 * R * R_star / ( ( R + R_star )*( R + R_star ) + ( Z - Z_star )*( Z - Z_star ) );
	double k = ::sqrt( k_squared );

	answer *= ::sqrt( ( R + R_star )*( R + R_star ) + ( Z - Z_star )*( Z - Z_star ) ) * ( ( 1.0 - 0.5*k_squared )*boost::math::ellint_1( k ) - boost::math::ellint_2( k ) );

	return answer;
}

// delta  = R - R_star;
double GradShafranovGreensFunction1D(double R, double R_star, double delta )
{
	double eps = 1e-4;
	double answer = 0;
	if ( ::fabs( R - R_star ) > eps || delta == 0 ) {
		answer = std::numbers::inv_pi / 2.0;

		double k_squared = 4 * R * R_star / ( ( R + R_star )*( R + R_star ) );
		double k = ::sqrt( k_squared );

		if ( ::fabs( R - R_star ) < 1e-4 )
		{
			delta = R - R_star;
			k_squared = ( 1 + delta/R_star )/( ( 1 + 0.5*delta/R_star )*( 1 + 0.5*delta/R_star ) );
			k = ::sqrt( k_squared );
		}

		answer *= ( R + R_star ) * ( ( 1.0 - 0.5*k_squared )*boost::math::ellint_1( k ) - boost::math::ellint_2( k ) );

		return answer;
	} else {
		double y = delta/R_star;
		double logY = ::log( ::fabs( y ) );
		double a    =  ( -4.0 + 6.0*M_LN2 - 2.0*logY )*( R_star*std::numbers::inv_pi/4.0   );
		double bx   =  ( -2.0 + 6.0*M_LN2 - 2.0*logY )*( R_star*std::numbers::inv_pi/8.0   ) * y;
		double cxx  =  ( -2.0 + 6.0*M_LN2 - 2.0*logY )*( R_star*std::numbers::inv_pi/64.0  ) * y * y;
		double dxxx = -( -4.0 + 6.0*M_LN2 - 2.0*logY )*( R_star*std::numbers::inv_pi/128.0 ) * y * y * y;
		return a + bx + cxx + dxxx;
	}
}

// H(R,R*) =  d G(R, R*) / dR
//
// psi(R,z=0) = Int[ G(R,r) J_phi(r), {r,0,Infinity}]
// B_z(R,z=0) = (1/R) d Psi / dR = (1/R) Int [ H(R,r) J_phi, {r,0,Infinity}], where this is a principal value integral

// H(R,r) = a/(R-r) + H_weak(R,r)
// where H_weak is only logarithmically singular at R=r

double DerivativeGreensFunction1D( double R, double R_star )
{
	double eps = 1e-4;
	double answer = 0;
	if ( ::fabs( R - R_star ) > eps ) {

		double k_squared = 4 * R * R_star / ( ( R + R_star )*( R + R_star ) );
		double k = ::sqrt( k_squared );

		answer =  ( std::numbers::inv_pi*R / 2.0 ) * ( boost::math::ellint_2( k )/( R_star - R ) + boost::math::ellint_1( k )/( R_star + R ) );

		return answer;
	} else {
		throw std::invalid_argument( "Use the singular decomposition,not the raw function" );
	}
}

double DerivativeGreensFunction1D_Residue( double R_star )
{
	return -( std::numbers::inv_pi/2.0 ) * R_star;
}

// x = R - R_star;
// This function returns H(R,R*) - a/(R-R*), by using a series expansion for small x and doing the subtraction otherwise
double DerivativeGreensFunction1D_Weak( double x, double R_star )
{
	double eps = 1e-4;
	if ( ::fabs( x ) > eps ){
		return DerivativeGreensFunction1D( x + R_star, R_star ) - DerivativeGreensFunction1D_Residue( R_star )/( x );
	} else {
		double y = x/R_star;
		double logY = ::log( ::fabs( y ) ); 
		double a =  ( std::numbers::inv_pi/8.0    )*(  -4.0 +   6.0*M_LN2 -   2.0*logY );
		double b =  ( std::numbers::inv_pi/32.0   )*(   5.0 +   6.0*M_LN2 -   2.0*logY )*y;
		double c = -( std::numbers::inv_pi/128.0  )*(   2.0 +  18.0*M_LN2 -   6.0*logY )*y*y;
		double d =  ( std::numbers::inv_pi/6144.0 )*( -89.0 + 612.0*M_LN2 - 204.0*logY )*y*y*y;
		return a + b + c + d;
	}
}


/*
double MidplaneB( double R, double R_coil, double Z_coil )
{
	double k_squared = 4 * R * R_coil / ( ( R + R_coil )*( R + R_coil ) + Z_coil * Z_coil );
	double k = ::sqrt( k_squared );

	double answer = 0;

	answer = ( ( ( R - R_coil )*( R - R_coil ) + Z_coil*Z_coil )*booost::math::ellint_1( k ) - ( R*R - R_coil*R_coil + Z_coil*Z_coil )*boost::math::ellint_2( k ) ) / ( M_PI * ( ( R - R_coil )*( R - R_coil ) + Z_coil*Z_coil ) * ( ::sqrt( ( R + R_coil )*( R + R_coil ) + Z_coil * Z_coil ) ) );

	return answer;
}
*/




