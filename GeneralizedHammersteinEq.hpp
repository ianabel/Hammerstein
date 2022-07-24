#ifndef GENERALIZEDHAMMERSTEINEQ_HPP
#define GENERALIZEDHAMMERSTEINEQ_HPP

#include <functional>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include "CollocationApprox.hpp"

/*
 * Code for solving nonlinear integral equations of the generalized Hammerstein form
 *                    / b
 * y( x ) = f( x ) +  |   ds K( x, s ) g( s, y( s ), y'(s) )    ( 1 )
 *                    / a
 *
 * by the method of Kumar and Sloan [ mathematics of computation volume 48. number 178 april 1987. pages 585-593 ]
 * using the collocation points of Kumar [ SIAM Journal on Numerical Analysis, Vol. 25, No. 2 (Apr., 1988), pp. 328-341 ]
 *
 * The equation that is solved is 
 *                             / b                           / b  dK
 * z( x ) = g (  x, f( x ) +   |   ds K( x, s ) z( s ), f' + | ds --(x,s) z(s) ) ( 2 )
 *                             / a                           / a  dx
 *
 * We expand z as a sum of basis functions. In particular we use discontinuous piecewise polynomial functions.
 * The basis functions are denoted u_i, and are scaled legendre polynomials on each interval.
 * We then test the integral equation at a set of collocation points, denoted tau_i,
 * rather than taking an inner product as one does in a galerkin method.
 *
 * This results in
 *                                                   / b                                                   / b   dK
 * Sum_j u_j( tau_i ) a_j  = g ( tau_i, f( tau_i ) + |  ds K( tau_i, s ) Sum_j u_j( s ) a_j, f'( tau_i ) + |  ds --( tau_i, s ) Sum_j u_j( s ) a_j ) ( 2 )
 *                                                   / a                                                   / a   dx
 * 
 */

class GeneralizedHammersteinEquation {
	public:
		GeneralizedHammersteinEquation( double A, double B, std::function<double( double )> F, std::function<double( double, double, double )> G, std::function<double( double, double )> k,
		                                  std::function<double( double )> Fp, std::function<double( double, double )> kP )
			: a( A ), b( B ), f( F ), g( G ), K( k ), fPrime( Fp ), KPrime( kP ), basis( nullptr ), zData( nullptr, 0 )
		{

		};

		constexpr static const double tanhsinh_tol = 1e-14;

		~GeneralizedHammersteinEquation() {
			if ( basis != nullptr )
				delete basis;
		}

		double a,b;
		std::function<double( double )> f,fPrime;
		std::function<double( double, double )> K,KPrime;
		std::function<double( double, double, double )> g,dgdy,dgdyPrime;

		Eigen::MatrixXd Mass;
		Eigen::MatrixXd K_ij,KPrime_ij;
		Eigen::VectorXd fVals,fPrimeVals;

		/* Determines numerical resolution and precomputes the needed values */
		void SetResolutionAndPrecompute( unsigned int NIntervals, unsigned int Order, bool gradedMesh = false, double gradingAlpha = 1.0 )
		{
			N_Intervals = NIntervals;
			N_Gauss = Order + 1;
			N = N_Intervals * N_Gauss;

			Mass = Eigen::MatrixXd::Zero( N, N );
			K_ij = Eigen::MatrixXd::Zero( N, N );
			KPrime_ij = Eigen::MatrixXd::Zero( N, N );
			fVals = Eigen::VectorXd::Zero( N );
			fPrimeVals = Eigen::VectorXd::Zero( N );


			Mesh.clear();
			if ( !gradedMesh ) {
			// Construct uniform mesh
			double h = static_cast<double>( b - a )/N_Intervals;
			for ( Eigen::Index i=0; i < N_Intervals; i++ )
			{
				Mesh.emplace_back( a + i*h, a + ( i + 1 )*h );
			}
			} else { 
				// m in the papers is N_Gauss here
				double q = static_cast<double>( N_Gauss )/( gradingAlpha );
				Eigen::VectorXd breakpoints( N_Intervals + 1 );
				for ( Eigen::Index i=0; i <= N_Intervals; ++i )
				{
					if ( i <= static_cast<double>( N_Intervals / 2.0 ) ) {
						breakpoints( i ) = a + ( ( b-a )/2 )*( ::pow( ( 2.0 * i / N_Intervals ), q ) );
					} else {
						breakpoints( i ) = b + a - breakpoints( N_Intervals - i );
					}
				}

				for ( Eigen::Index i=0; i < N_Intervals; ++i ) {
					Mesh.emplace_back( breakpoints( i ), breakpoints( i + 1 ) );
				}
			}

			if ( basis != nullptr )
				delete basis;
			basis = new DGLegendreBasis( Mesh, Order );

			// Form the matrix of elements u_j( tau_i ) where u_j are the basis functions
			// and tau_i are the collocation points.
			#pragma omp parallel for
			for ( Eigen::Index i=0; i < N; i++ )
				for ( Eigen::Index j=0; j < N; j++ )
					Mass( i, j ) = basis->EvaluateBasis( j, basis->CollocationPoints[ i ] );

			// Now construct the matrix that has elements
			//        / b
			// K_ij = |   K( tau_i, s ) u_j( s ) ds
			//        / a
			#pragma omp parallel for
			for ( Eigen::Index i=0; i < N; i++ )
				for ( Eigen::Index j=0; j < N; j++ )
				{
					// With a discontinuous basis, the integral is only over one of the intervals.
					Eigen::Index Int = j / N_Gauss;

					auto K_integrand = [ & ]( double s ) {
						return K( basis->CollocationPoints[ i ], s )*basis->EvaluateBasis( j, s );
					};
					auto KPrime_integrand = [ & ]( double s ){
						double retval = KPrime( basis->CollocationPoints[ i ], s ) * basis->EvaluateBasis( j, s );
						if ( ::isfinite( retval ) )
							return retval;
						else
							throw std::logic_error( "Could not precompute integrals, dK/dx seems to be too singular" );
					};
					boost::math::quadrature::tanh_sinh<double> integrator( 4, tanhsinh_tol ); // sufficient for 1e-8 precision, without specially-equipped kernel functions
					if ( Mesh[ Int ].x_l <= basis->CollocationPoints[ i ] && basis->CollocationPoints[ i ] < Mesh[ Int ].x_u ) {
						K_ij( i, j )      = integrator.integrate( K_integrand,      Mesh[ Int ].x_l, basis->CollocationPoints[ i ] ) + integrator.integrate( K_integrand,      basis->CollocationPoints[ i ], Mesh[ Int ].x_u ) ;
						KPrime_ij( i, j ) = integrator.integrate( KPrime_integrand, Mesh[ Int ].x_l, basis->CollocationPoints[ i ] ) + integrator.integrate( KPrime_integrand, basis->CollocationPoints[ i ], Mesh[ Int ].x_u ) ;
					} else {
						K_ij( i, j )      = integrator.integrate( K_integrand,      Mesh[ Int ].x_l, Mesh[ Int ].x_u );
						KPrime_ij( i, j ) = integrator.integrate( KPrime_integrand, Mesh[ Int ].x_l, Mesh[ Int ].x_u );
					}
				}
			MassSolver.compute( Mass );

			#pragma omp parallel for
			for ( Eigen::Index i=0 ; i < N; ++i )
			{
				fVals( i )      = f( basis->CollocationPoints[ i ] );
				fPrimeVals( i ) = fPrime( basis->CollocationPoints[ i ] );
			}
		}

		/* 
		 * In terms of the above matrices, the z equation is now an equation for the a_j coefficients:
		 * 
		 * ( M a )_i = g( tau_i, f( tau_i ) + ( K * a )_i, f'( tau_i ) + ( K' * a )_i  )
		 *
		 * for our nonlinear problem we set
		 *
		 * F( a )_i = g( tau_i, f( tau_i ) + ( K*a )_i,  ) - ( M a )_i
		 */

		int ComputeResidual( N_Vector u, N_Vector F )
		{
			new( &zData ) VecMap( NV_DATA_S( u ), N );
			VecMap output( NV_DATA_S( F ), N );

			Eigen::VectorXd Ma( N ),Ka( N ),KPa( N ),Fa( N );
			Ma = Mass * zData;
			Ka = K_ij * zData;
			KPa = KPrime_ij * zData;
			#pragma omp parallel for
			for ( Eigen::Index i=0; i < N; i++ )
				Fa( i ) = g( basis->CollocationPoints[ i ], fVals( i ) + Ka( i ), fPrimeVals( i ) + KPa( i ) ) - Ma( i );
			output = Fa;
			return KIN_SUCCESS;
		}

		void setgPrime( std::function<double( double, double, double )> gP1, std::function<double( double, double, double )> gP2 )
		{
			dgdy = gP1;
			dgdyPrime = gP2;
		};

		// data needs to be allocated already
		// you must have already precomputed the Mass matrix.
		void computeCoefficients( double *data, std::function<double( double )> yZero )
		{
			VecMap x( data, N );
			Eigen::VectorXd b_vals( N );
			for ( Eigen::Index i=0; i<N; ++i )
			{
				double tau_i = basis->CollocationPoints[ i ];
				double eps = 1e-5;
				double yPrime = ( yZero( tau_i + eps ) - yZero( tau_i - eps ) )/( 2*eps );
				b_vals( i ) = g( tau_i, yZero( tau_i ), yPrime );
			}
			x = MassSolver.solve( b_vals );
		}

		void computeCoefficients( double *data, std::function<double( double )> yZero, std::function<double( double )> yPrime )
		{
			VecMap x( data, N );
			Eigen::VectorXd b_vals( N );
			for ( Eigen::Index i=0; i<N; ++i )
			{
				double tau_i = basis->CollocationPoints[ i ];
				b_vals( i ) = g( tau_i, yZero( tau_i ), yPrime( tau_i ) );
			}
			x = MassSolver.solve( b_vals );
		}
		/*
		 * Need to have already set gPrime
		 */

		int ComputeJacobian( N_Vector u, SUNMatrix Jac, N_Vector tmp1, N_Vector tmp2 )
		{
			if ( !dgdy or !dgdyPrime )
				throw std::runtime_error( "Evaluating explicit Jacobian without passing derivatives of g to problem object" );
			Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > JMatrix( SM_DATA_D( Jac ), N, N );
			Eigen::Map<Eigen::VectorXd> tV1( NV_DATA_S( tmp1 ), N );
			Eigen::Map<Eigen::VectorXd> tV2( NV_DATA_S( tmp2 ), N );
			new( &zData ) VecMap( NV_DATA_S( u ), N );

			tV1 = fVals + K_ij * zData;
			tV2 = fPrimeVals + KPrime_ij * zData;
			for ( Eigen::Index i=0; i < N; i++ )
			{
				double tmp1 = tV1( i );
				double tmp2 = tV2( i );
				tV1( i ) = dgdy( basis->CollocationPoints[ i ], tmp1, tmp2 );
				tV2( i ) = dgdyPrime( basis->CollocationPoints[ i ], tmp1, tmp2 );
			}

			JMatrix = ( tV1.asDiagonal() * K_ij + tV2.asDiagonal() * KPrime_ij ) - Mass;
			return KIN_SUCCESS;
		}

		void setzData( N_Vector u )
		{
			new( &zData ) VecMap( NV_DATA_S( u ), N );
		}

		static int KINSOL_GeneralizedHammerstein( N_Vector u, N_Vector F, void* problemData )
		{
			GeneralizedHammersteinEquation *pHammer = reinterpret_cast<GeneralizedHammersteinEquation*>( problemData );
			return pHammer->ComputeResidual( u, F );
		}

		static int KINSOL_GeneralizedHammersteinJacobian( N_Vector u, N_Vector F, SUNMatrix J, void * problemData, N_Vector temp1, N_Vector temp2 )
		{
			GeneralizedHammersteinEquation *pHammer = reinterpret_cast<GeneralizedHammersteinEquation*>( problemData );
			return pHammer->ComputeJacobian( u, J, temp1, temp2 );
		}

		unsigned int getDimension() const {
			return N;
		};

		double EvaluateZ( double x ) {

			double Sum;
			Sum = 0;
			#pragma omp parallel for reduction( +: Sum )
			for ( Eigen::Index i = 0; i < N; ++i )
			{
				Sum += zData[ i ] * basis->EvaluateBasis( i, x );
			}
			return Sum;
		};

		/*
			Evaluate y(t) using
			                  / b
			y( t ) = f( t ) + |   K( t, s ) z( s )
			                  / a
			which becomes
			                            / b
			y( t ) = f( t ) + Sum_j a_j |   K( t, s ) u_j( s ) 
			                            / a
		*/
		double EvaluateY( double x ) {
			double y_val = f( x );
			for ( Eigen::Index j=0; j < N; j++ )
			{
				double KIntegral = 0;
				// With a discontinuous basis, the integral is only over one of the intervals.
				Eigen::Index Int = j / N_Gauss;
				/*
				// Sum over the Gauss nodes in the interval with gaussian weights to perform the integral
				for ( Eigen::Index k=0; k < N_Gauss; k++ )
					KIntegral += K( x, basis->CollocationPoints[ Int * N_Gauss + k ] ) * basis->EvaluateBasis( j, basis->CollocationPoints[ Int * N_Gauss + k ] ) * basis->gauss.weights[ k ] * Mesh[ Int ].h()/2.0;
				*/
				auto K_integrand = [ & ]( double s ) {
						return K( x, s )*basis->EvaluateBasis( j, s );
				};

				boost::math::quadrature::tanh_sinh<double> integrator( 4, tanhsinh_tol ); // sufficient for 1e-8 precision, without specially-equipped kernel functions
				if ( Mesh[ Int ].x_l <= x && x < Mesh[ Int ].x_u ) {
					double I_l,I_u;
					// If we're sampling very close to a meshpoint we can get an error
					if ( ( x - Mesh[ Int ].x_l ) < tanhsinh_tol )
						I_l = 0;
					else
						I_l = integrator.integrate( K_integrand, Mesh[ Int ].x_l, x );
					if ( ( Mesh[ Int ].x_u - x ) < tanhsinh_tol )
						I_u = 0;
					else
						I_u = integrator.integrate( K_integrand, x, Mesh[ Int ].x_u ) ;
					KIntegral = I_l + I_u;
				} else {
					KIntegral = integrator.integrate( K_integrand, Mesh[ Int ].x_l, Mesh[ Int ].x_u );
				}
				y_val += zData[ j ]*KIntegral;
			}
			return y_val;
		};

	private:
		unsigned int N_Intervals,N_Gauss,N;
		std::vector<Interval> Mesh;
		using VecMap = Eigen::Map<Eigen::VectorXd>;
		Eigen::PartialPivLU<Eigen::MatrixXd> MassSolver;
		VecMap zData;

		CollocationBasis *basis;

};
#endif // GENERALIZEDHAMMERSTEINEQ_HPP

