
#include <kinsol/kinsol.h>             /* access to KINSOL func., consts. */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector       */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */

#include <functional>
#include <iostream>
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
	
	unsigned int N_Intervals = 16;
	unsigned int PolynomialOrder = 2;

	std::function<double( double )> CoilPsi = std::bind( PsiCoils, std::placeholders::_1, R_c, Z_c );

	std::cout << CoilPsi( 0.2 ) << '\t' << CoilPsi( 0.5 ) << '\t' << CoilPsi( 0.8 ) << std::endl;

	CurrentB = std::bind( MidplaneB, std::placeholders::_1, R_c, Z_c );
	HammersteinEquation PsiProblem( 0.1, 2, CoilPsi, Jtor2, GradShafranovGreensFunction1D );

	sundials::Context sunctx;

	void *kinMem = KINCreate( sunctx );


	PsiProblem.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder );

	sunindextype NDims = PsiProblem.getDimension();
	N_Vector zDataInit = N_VNew_Serial( NDims, sunctx );

	// Initial Condition near known answer - M_A is small enough
	// that we can try the strong-field solution first
	// set z = 0, i.e. J_tor = 0

	N_VConst( 0, zDataInit );

	KinsolErrorWrapper( KINInit( kinMem, HammersteinEquation::KINSOL_Hammerstein, zDataInit ), "KINInit" );
	KinsolErrorWrapper( KINSetPrintLevel( kinMem, 1 ), "KINSetPrintLevel" );
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

	KinsolErrorWrapper( KINSetMaxSetupCalls( kinMem, 1 ), "KINSetMaxSetupCalls" );

	/*
	N_Vector F_out = N_VNew_Serial( NDims, sunctx );
	PsiProblem.ComputeResidual( zDataInit, F_out );
	PsiProblem.setzData( F_out );
	std::cout << PsiProblem.EvaluateZ( 0.2 ) << '\t' << PsiProblem.EvaluateZ( 0.5 ) << '\t' << PsiProblem.EvaluateZ( 0.8 ) << std::endl;
	std::cout << Jtor( 0.2, CoilPsi( 0.2 ) ) << '\t' << Jtor( 0.5, CoilPsi( 0.5 ) ) << '\t' << Jtor( 0.8, CoilPsi( 0.8 ) ) << std::endl;
	std::cout << PsiProblem.EvaluateY( 0.2 ) << '\t' << PsiProblem.EvaluateY( 0.5 ) << '\t' << PsiProblem.EvaluateY( 0.8 ) << std::endl;

	return 0;
	*/
	KinsolErrorWrapper( KINSol( kinMem, zDataInit, KIN_LINESEARCH, one, one ), "KINSol" );

	PsiProblem.setzData( zDataInit );

	std::cout << PsiProblem.EvaluateY( 0.2 ) << '\t' << PsiProblem.EvaluateY( 0.5 ) << '\t' << PsiProblem.EvaluateY( 0.8 ) << std::endl;


	return 0;
}
