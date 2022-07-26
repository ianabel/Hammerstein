
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

#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <boost/math/tools/roots.hpp>

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

double CoilStrength = 1.5; // Units of T.m (= mu_0 * I_coil)

double PsiCoils( double R, double Z, double R_coil, double Z_coil )
{
	return CoilStrength * ( GradShafranovGreensFunction( R, R_coil, Z, Z_coil ) + GradShafranovGreensFunction( R, R_coil, Z, -Z_coil ) );
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

constexpr double OmegaMax = 0.3;
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

double R_c = 0.4;
double Z_c = 1.0;

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

	std::cout << std::setprecision( 16 ) << std::endl;
	
	unsigned int N_Intervals = 100;
	unsigned int PolynomialOrder = 3;

	std::function<double( double )> CoilPsi = std::bind( PsiCoils, std::placeholders::_1, 0, R_c, Z_c );

	double R_electrode = 0.025;
	PsiInner = PsiCoils( R_electrode, Z_c, R_c, Z_c );
	double R_limiter = 0.25;
	double Z_limiter = 0.1;
	PsiOuter = PsiCoils( R_limiter, Z_limiter, R_c, Z_c );

	double PsiMid = ( PsiInner + PsiOuter )/2.0;
	double R_inner,R_outer,R_mid;
	
	{
		unsigned long m_iter =25;
		auto [ Riv_l,Riv_u ] = boost::math::tools::toms748_solve( [&]( double R ){ return CoilPsi( R ) - PsiInner;}, 0.0,0.25, boost::math::tools::eps_tolerance<double>(), m_iter  );
		R_inner = ( Riv_l+Riv_u )/2;
		m_iter = 25;
		auto [ Rov_l,Rov_u ] = boost::math::tools::toms748_solve( [&]( double R ){ return CoilPsi( R ) - PsiOuter;}, 0.2,R_c, boost::math::tools::eps_tolerance<double>(), m_iter );
		R_outer = ( Rov_l+Rov_u )/2;
		auto [ Rmid_l, Rmid_u ] = boost::math::tools::toms748_solve( [&]( double R ){ return CoilPsi( R ) - PsiMid;}, 0.0,R_c, boost::math::tools::eps_tolerance<double>(), m_iter );
		R_mid = ( Rmid_l+Rmid_u )/2;
	}


	std::cout << std::setprecision( 3 );
	std::cout << " Plasma appears to be between R = " << R_inner << " and " << R_outer << std::endl;
	std::cout << " Plasma centrline starts at   R = " << R_mid << std::endl;
	std::cout << " Magnetic field at centreline is " << MidplaneB( R_mid, R_c, Z_c ) << " T" << std::endl;

	std::cout << " Approximate Alfven Mach is initially " << OmegaMax *( R_mid/MidplaneB( R_mid, R_c, Z_c ) ) << std::endl; 
	
	CurrentB = std::bind( MidplaneB, std::placeholders::_1, R_c, Z_c );
	HammersteinEquation PsiProblem( 0.0, 1.0, CoilPsi, Jtor2, GradShafranovGreensFunction1D );

	sundials::Context sunctx;

	void *kinMem = KINCreate( sunctx );

	PsiProblem.SetResolutionAndPrecompute( N_Intervals, PolynomialOrder, HammersteinEquation::BasisType::DGLegendre );

	auto B_operator_smooth = [ & ]( double R, double Rs, double sc ){
		if ( ::fabs( R-Rs ) > 1e-4 )
			return ( DerivativeGreensFunction1D_Weak( R, Rs, R - Rs ));
		else
			return ( DerivativeGreensFunction1D_Weak( R, Rs, sc ) );
	};

	auto B_operator_pv = [ & ]( double Rs ) {
		return DerivativeGreensFunction1D_Residue( Rs );
	};

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

	Eigen::VectorXd B_data = PsiProblem.applyIntegralOperator( B_operator_smooth, B_operator_pv, [ & ]( double R ){return 1.0/R;} );

	std::cout << MidplaneB( 0.15, R_c, Z_c ) << '\t' << MidplaneB( 0.15, R_c, Z_c ) + PsiProblem.Evaluate( B_data, 0.15 ) << std::endl;
	std::cout << MidplaneB( 0.25, R_c, Z_c ) << '\t' << MidplaneB( 0.25, R_c, Z_c ) + PsiProblem.Evaluate( B_data, 0.25 ) << std::endl;

	{
		unsigned long m_iter =25;
		auto [ Riv_l,Riv_u ] = boost::math::tools::toms748_solve( [&]( double R ){ return PsiProblem.EvaluateY( R ) - PsiInner;}, 0.0,0.25, boost::math::tools::eps_tolerance<double>(), m_iter  );
		R_inner = ( Riv_l+Riv_u )/2;
		m_iter = 25;
		auto [ Rov_l,Rov_u ] = boost::math::tools::toms748_solve( [&]( double R ){ return PsiProblem.EvaluateY( R ) - PsiOuter;}, 0.2,0.5, boost::math::tools::eps_tolerance<double>(), m_iter );
		R_outer = ( Rov_l+Rov_u )/2;
		auto [ Rmid_l, Rmid_u ] = boost::math::tools::toms748_solve( [&]( double R ){ return PsiProblem.EvaluateY( R ) - PsiMid;}, 0.0,R_c, boost::math::tools::eps_tolerance<double>(), m_iter );
		R_mid = ( Rmid_l+Rmid_u )/2;
	}

	std::cout << " Plasma is now between R = " << R_inner << " and " << R_outer << std::endl;
	std::cout << " Plasma centrline ends at   R = " << R_mid << std::endl;
	double B_mid = MidplaneB( R_mid, R_c, Z_c ) + PsiProblem.Evaluate( B_data, R_mid );
	std::cout << " Magnetic field at centreline is " << B_mid << " T" << std::endl;
	std::cout << " Approximate Alfven Mach is finally " << OmegaMax * R_mid / B_mid << std::endl; 


	std::fstream out( "Psi.dat", std::ios_base::out );


	unsigned int N_samples = 256;
	out << "# R\tPsi" << std::endl;
	for ( unsigned int i=0; i <= N_samples; i++ )
	{
		double R = 0.0 + ( 1.0 - 0.0 )*( static_cast<double>( i )/static_cast<double>( N_samples ) );
		double psi = PsiProblem.EvaluateY( R );
		out << R << '\t' << CoilPsi( R ) << '\t' << psi << '\t' << MidplaneB( R, R_c, Z_c ) << '\t' << MidplaneB( R, R_c, Z_c ) + PsiProblem.Evaluate( B_data, R ) << '\t' << omega( psi ) << std::endl;
	}
	out.close();


	/*
	std::fstream out2( "Psi2D.dat", std::ios_base::out );


	N_samples = 256;
	out2 << "# R\tPsi" << std::endl;
	double R_min = 0.0,R_max = 4*R_c,Z_min = 0.0,Z_max = 0.75*Z_c;
	for ( unsigned int i=0; i <= N_samples; i++ )
	{
		double R = R_min + ( R_max - R_min )*( static_cast<double>( i )/static_cast<double>( N_samples ) );
		for ( unsigned int j=1; j <= N_samples; j++ ) {
			double Z = Z_min + ( Z_max - Z_min )*( static_cast<double>( j )/static_cast<double>( N_samples ) );
			boost::math::quadrature::tanh_sinh<double> Int( 5, 1e-15 );
			auto PsiIntegrand = [ & ]( double Rs ) {
				return GradShafranovGreensFunction( R, Z, Rs, 0 )*Jtor2( Rs, PsiProblem.EvaluateY( Rs ) );
			};

			double psi = PsiCoils( R, Z, R_c, Z_c ) + Int.integrate( PsiIntegrand, R_inner, R_outer );

			out2 << R << '\t' << Z << '\t' << psi << std::endl;
		}
	}
	out2.close();
	*/




	return 0;
}
