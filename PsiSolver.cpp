
#include <kinsol/kinsol.h>             /* access to KINSOL func., consts. */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector       */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */

#include <functional>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cfloat>

#include <boost/math/quadrature/tanh_sinh.hpp>

#include "HammersteinEquation.hpp"
#include "GreensFunction.hpp"

void KinsolErrorWrapper( int errorFlag, std::string&& fName )
{
	if ( errorFlag == KIN_SUCCESS )
		return;
	else
	{
		throw std::runtime_error( "Error " + std::to_string( errorFlag ) + " returned from KINSol function: " + fName );
	}
}

double PsiCoils( double R, double R_coil, double Z_coil )
{
	double Z = 0;
	return GradShafranovGreensFunction( R, R_coil, Z, Z_coil ) + GradShafranovGreensFunction( R, R_coil, Z, -Z_coil );
}

double MidplaneB( double R, double R_coil, double Z_coil )
{
	double k_squared = 4 * R * R_coil / ( ( R + R_coil )*( R + R_coil ) + Z_coil * Z_coil );
	double k = ::sqrt( k_squared );

	double answer = 0;

	answer = ( ( ( R - R_coil )*( R - R_coil ) + Z_coil*Z_coil )*boost::math::ellint_1( k ) - ( R*R - R_coil*R_coil + Z_coil*Z_coil )*boost::math::ellint_2( k ) ) / ( M_PI * ( ( R - R_coil )*( R - R_coil ) + Z_coil*Z_coil ) * ( ::sqrt( ( R + R_coil )*( R + R_coil ) + Z_coil * Z_coil ) ) );

	return answer;
}
//  GradShafranovGreensFunction( R, Rprime, 0.0, 0.0 ) * om * om * Rprime / Bz( OldPsi, Rprime );

constexpr double OmegaMax = 0.1;
double PsiInner,PsiOuter;
double omega( double psi )
{
	if ( psi < PsiInner )
		return 0.0;
	if ( psi > PsiOuter )
		return 0.0;
	double psiWidth2 = ( PsiInner - PsiOuter )*( PsiInner - PsiOuter );
	return OmegaMax * 4.0 * ( psi - PsiInner )*( PsiOuter - psi )/psiWidth2;
}

double n_i_bar( double psi )
{
	return 1.0;
}

double R_c = 1.75;
double Z_c = 2.5;

double Jtor( double R, double psi )
{
	// n_i_bar * m_i * omega^2 * R / B_z/mu_0
	
	return -n_i_bar( psi )*omega( psi )*omega( psi )*R/MidplaneB( R, R_c, Z_c );
}

std::function<double( double )> CurrentB;
double Jtor2( double R, double psi )
{
	return -n_i_bar( psi )*omega( psi )*omega( psi )*R/CurrentB( R );
}

/*
	The integral equation for psi is
                              / R_max
	psi( R ) = PsiCoils( R ) - |  dR   G( R, R' )*( n_i_bar(psi) * m_i * omega^2(psi) * R' / B_z/mu_0 ) dR
	                           / 0
 */

int main( int, char** )
{
	// R=0.2 & R=0.5 in vacuum
	PsiInner = 0.005261448006626289;
	PsiOuter = 0.02895524662380254;

	std::cout << std::setprecision( 16 ) << std::endl;
	
	unsigned int N_Intervals = 50;
	unsigned int PolynomialOrder = 3;

	std::function<double( double )> CoilPsi = std::bind( PsiCoils, std::placeholders::_1, R_c, Z_c );

	std::cout << CoilPsi( 0.2 ) << '\t' << CoilPsi( 0.5 ) << '\t' << CoilPsi( 0.8 ) << std::endl;

	CurrentB = std::bind( MidplaneB, std::placeholders::_1, R_c, Z_c );
	HammersteinEquation PsiProblem( 0.0, 1.0, CoilPsi, Jtor2, GradShafranovGreensFunction1D );

	sundials::Context sunctx;

	void *kinMem = KINCreate( sunctx );

	PsiProblem.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder, HammersteinEquation::BasisType::DGLegendre );

	auto Jphi = []( double t ){ 
		if ( t < 0.2 || t > 0.5 )
			return 0.0;
		else
			return 0.1*( t - 0.2 )*( 0.5 - t ) / ( 0.15 * 0.15 );
	};
	Eigen::VectorXd zTmp = PsiProblem.computeZCoefficients( Jphi );

	PsiProblem.setzData( zTmp );
	double R_eval=0.25;
	std::cout << CurrentB( 0.25 ) << std::endl;
	auto Hintegrand = [ & ]( double r ){
		return ( DerivativeGreensFunction1D_Weak( R_eval, r, R_eval-r ) )*Jphi( r );
	};

	boost::math::quadrature::tanh_sinh<double> integrator( 8, 1e-15 );

	double pvp = CauchyPV( [ & ]( double x ){ return DerivativeGreensFunction1D_Residue( x )*Jphi( x );}, 0.2, 0.5, 0.25 ) ;
	std::cout << MidplaneB( R_eval, R_c, Z_c ) - ( pvp + integrator.integrate( Hintegrand, 0.2, R_eval ) + integrator.integrate( Hintegrand, R_eval,0.5 ) )/( R_eval ) << std::endl;
	

	auto PsiIntegrand = [ & ]( double R, double Rs, double x ) {
		if ( ::fabs( R - Rs )>1e-4 )
		{
			return GradShafranovGreensFunction1D( R, Rs, R-Rs ) * Jphi( Rs );
		}
		else
		{
			return GradShafranovGreensFunction1D( R, Rs, x )*Jphi( Rs );
		}
	};
	auto PsiNew = [ & ]( double R ) {
		boost::math::quadrature::tanh_sinh<double> integrator( 8, 1e-15 );
		auto PI = std::bind( PsiIntegrand, R, std::placeholders::_1, std::placeholders::_2 );
		return CoilPsi( R ) - integrator.integrate( PI, 0.2, R ) - integrator.integrate( PI, R, 0.5 );
	};

	std::cout << PsiNew( 0.24 ) << '\t' << PsiNew( 0.26 ) << std::endl;
	std::cout << ( PsiNew( 0.251 ) - PsiNew( 0.249 ) )/( 0.002*0.25 ) << std::endl;

	auto B_operator_smooth = [ & ]( double R, double Rs, double sc ){
		if ( ::fabs( R-Rs ) > 1e-4 )
			return ( DerivativeGreensFunction1D_Weak( R, Rs, R - Rs ));
		else
			return ( DerivativeGreensFunction1D_Weak( R, Rs, sc ) );
	};

	auto B_operator_pv = [ & ]( double Rs ) {
		return DerivativeGreensFunction1D_Residue( Rs );
	};

	Eigen::VectorXd B_data = PsiProblem.applyIntegralOperator( B_operator_smooth, B_operator_pv, [ & ]( double R ){return 1.0/R;} );
	std::cout << MidplaneB( 0.15, R_c, Z_c ) - PsiProblem.Evaluate( B_data, 0.15 ) << std::endl;
	std::cout << MidplaneB( 0.25, R_c, Z_c ) - PsiProblem.Evaluate( B_data, 0.25 ) << std::endl;
	sunindextype NDims = PsiProblem.getDimension();
	N_Vector zDataInit = N_VNew_Serial( NDims, sunctx );

	// Initial Condition near known answer - M_A is small enough
	// that we can try the strong-field solution first
	// set z = 0, i.e. J_tor = 0

	N_VConst( 0, zDataInit );

	KinsolErrorWrapper( KINInit( kinMem, HammersteinEquation::KINSOL_Hammerstein, zDataInit ), "KINInit" );
	KinsolErrorWrapper( KINSetPrintLevel( kinMem, 0 ), "KINSetPrintLevel" );
	KinsolErrorWrapper( KINSetUserData( kinMem, static_cast<void*>( &PsiProblem ) ), "KINSetUserData" );

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

	KinsolErrorWrapper( KINSetMaxSetupCalls( kinMem, 0 ), "KINSetMaxSetupCalls" );

	KinsolErrorWrapper( KINSol( kinMem, zDataInit, KIN_LINESEARCH, one, one ), "KINSol" );

	PsiProblem.setzData( zDataInit );


	std::fstream out( "Psi.dat", std::ios_base::out );


	unsigned int N_samples = 256;
	out << "# R\tPsi" << std::endl;
	for ( unsigned int i=0; i <= N_samples; i++ )
	{
		double R = 0.0 + ( 1.0 - 0.0 )*( static_cast<double>( i )/static_cast<double>( N_samples ) );
		out << R << '\t' << CoilPsi( R ) << '\t' << PsiProblem.EvaluateY( R ) << std::endl;
	}
	out.close();

	return 0;
}
