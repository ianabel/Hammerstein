
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

double GradShafranovGreensFunction1D(double R, double R_star)
{
	double answer = std::numbers::inv_pi / 2.0;

	double k_squared = 4 * R * R_star / ( ( R + R_star )*( R + R_star ) );
	double k = ::sqrt( k_squared );

	answer *= ( R + R_star ) * ( ( 1.0 - 0.5*k_squared )*boost::math::ellint_1( k ) - boost::math::ellint_2( k ) );

	return answer;

}

// for evaluating near R=R' and Z=Z'
double GradShafranovGreensFunctionSingular( double R, double Z, double dR, double dZ )
{
	double answer = std::numbers::inv_pi / 2.0;

	double d = dR*dR + dZ*dZ;

	// Dont use this if you're not close
	if ( d > 1e-8 )
		return GradShafranovGreensFunction( R, R + dR, Z, Z + dZ );
	else 
	{
		// k_squared = (1 + dR) / ( 1 + dR + dR^2/4 + dZ^2/4 ) ;
		// k_squared ~= (1 + dR)*( 1 - (dR + dR^2/4 + dZ^2/4) + (dR + dR^2/4 + dZ^2/4)^2))
		//   ~ = 1 - .25*dR^2 - .25*dZ^2;
		// Thus, m = 1 - dR*dR*.25 - dZ*dZ*.25;
		// K(m) ~= -(1/2) ln( 1 - m ) + ln(4) + |m-1|;
		// Set k = k^2 = 1 outside of the argument of K(). Keep linear terms, as doubles can detect those, but the squares are 
		// beyond double resolution.
		//
		// log( 1 - m ) ~= log( dR^2 + dZ^2 ) - log(4)
		double K_approx;
		double log64 = M_LOG2E * 6.0;
		K_approx = ( 1.0 / 2.0 ) *( log64 - ::log( d ) ) - d*::log( d )/32.0 + ( 1.0 / 32.0 )*( log64 - 2.0 ) * d;
		double E_approx;
		E_approx = 1.0 + ( 1.0/16.0 )*( log64 - 1.0 - ::log( d ) )*d;

		answer *= ( 2*R + dR + dZ*dZ/( 4.0*R ) ) * ( ( 1.0/2.0 + d/8.0 )*K_approx - E_approx );
	}

	return answer;
}



