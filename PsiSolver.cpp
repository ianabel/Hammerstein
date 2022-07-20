

#include <string>

#include <cmath>
#include <cfloat>

#include <functional>
#include <iostream>
#include <iomanip>

#include "GreensFunction.hpp"

constexpr double Z_coil = 1.0;
constexpr double R_coil = 0.7;
constexpr double R_max = 1.0;


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

	answer = ( ( ( R - R_coil )*( R - R_coil ) + Z_coil*Z_coil )*booost::math::ellint_1( k ) - ( R*R - R_coil*R_coil + Z_coil*Z_coil )*boost::math::ellint_2( k ) ) / ( M_PI * ( ( R - R_coil )*( R - R_coil ) + Z_coil*Z_coil ) * ( ::sqrt( ( R + R_coil )*( R + R_coil ) + Z_coil * Z_coil ) ) );

	return answer;
}
//  GradShafranovGreensFunction( R, Rprime, 0.0, 0.0 ) * om * om * Rprime / Bz( OldPsi, Rprime );

constexpr double OmegaMax = 2.0;
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

	HammersteinEquation PsiProblem( -1, 1, f, g_test, K_test, K_res );

	sundials::Context sunctx;

	void *kinMem = KINCreate( sunctx );


	PsiProblem.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder );

	sunindextype NDims = PsiProblem.getDimension();
	N_Vector zDataInit = N_VNew_Serial( NDims, sunctx );

	// Initial Condition near known answer.

	auto initial_y = [&]( double t ){
		return ::pow( t, 4 );
	};

	PsiProblem.computeCoefficients( NV_DATA_S( zDataInit ), initial_y );

	KinsolErrorWrapper( KINInit( kinMem, SingularHammersteinEquation::KINSOL_Hammerstein, zDataInit ), "KINInit" );
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

	KinsolErrorWrapper( KINSetConstraints( kinMem, zero ), "KINSetConstraints" );
	KinsolErrorWrapper( KINSetMaxSetupCalls( kinMem, 2 ), "KINSetMaxSetupCalls" );

	KinsolErrorWrapper( KINSol( kinMem, zDataInit, KIN_LINESEARCH, one, one ), "KINSol" );

	PsiProblem.setzData( zDataInit );



	return 0;
}
