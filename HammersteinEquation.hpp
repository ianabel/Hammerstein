#ifndef HAMMERSTEINEQUATION_HPP
#define HAMMERSTEINEQUATION_HPP

#include <functional>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include "CollocationApprox.hpp"

/*
 * Code for solving nonlinear integral equations of the Hammerstein form
 *                    / b
 * y( x ) = f( x ) +  |   ds K( x, s ) g( s, y( s ) )    ( 1 )
 *                    / a
 *
 * by the method of Kumar and Sloan [ mathematics of computation volume 48. number 178 april 1987. pages 585-593 ]
 * using the collocation points of Kumar [ SIAM Journal on Numerical Analysis, Vol. 25, No. 2 (Apr., 1988), pp. 328-341 ]
 *
 * The equation that is solved is 
 *                             / b
 * z( x ) = g (  x, f( x ) +   |   ds K( x, s ) z( s ) ) ( 2 )
 *                             / a
 *
 * We expand z as a sum of basis functions. In particular we use discontinuous piecewise polynomial functions.
 * The basis functions are denoted u_i, and are scaled legendre polynomials on each interval.
 * We then test the integral equation at a set of collocation points, denoted tau_i,
 * rather than taking an inner product as one does in a galerkin method.
 *
 * This results in
 *                                                   / b
 * Sum_j u_j( tau_i ) a_j  = g ( tau_i, f( tau_i ) + |  ds K( tau_i, s ) Sum_j u_j( s ) a_j ) ( 2 )
 *                                                   / a
 * 
 */

class HammersteinEquation {
	public:
		HammersteinEquation( double A, double B, std::function<double( double )> F, std::function<double( double, double )> G, std::function<double( double, double )> k )
			: a( A ), b( B ), f( F ), g( G ), K( k ),K_singular( nullptr ), basis( nullptr ), zData( nullptr, 0 )
		{

		};

		HammersteinEquation( double A, double B, std::function<double( double )> F, std::function<double( double, double )> G, std::function<double( double, double, double )> k )
			: a( A ), b( B ), f( F ), g( G ), K( nullptr ),K_singular( k ), basis( nullptr ), zData( nullptr, 0 )
		{

		};


		constexpr static const double tanhsinh_tol = 1e-15;

		~HammersteinEquation() {
			if ( basis != nullptr )
				delete basis;
		}

		double a,b;
		std::function<double( double )> f;
		std::function<double( double, double )> g,K,gPrime;
		std::function<double( double, double, double )> K_singular;

		Eigen::MatrixXd Mass;
		Eigen::MatrixXd K_ij;
		Eigen::VectorXd fVals;

		/* Determines numerical resolution and precomputes the needed values */
		void SetResolutionAndPrecompute( unsigned int NIntervals, unsigned int Order, bool gradedMesh = false, double gradingAlpha = 1.0 )
		{
			N_Intervals = NIntervals;
			N_Gauss = Order + 1;
			N = N_Intervals * N_Gauss;

			Mass = Eigen::MatrixXd::Zero( N, N );
			K_ij = Eigen::MatrixXd::Zero( N, N );

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
			basis = new CollocationBasis( Mesh, Order );

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
					double Kij = 0;
					// With a discontinuous basis, the integral is only over one of the intervals.
					Eigen::Index Int = j / N_Gauss;

					if ( !K_singular ) {
						auto K_integrand = [ & ]( double s ) {
							return K( basis->CollocationPoints[ i ], s )*basis->EvaluateBasis( j, s );
						};
						boost::math::quadrature::tanh_sinh<double> integrator( 4, tanhsinh_tol ); // sufficient for 1e-8 precision, without specially-equipped kernel functions
						if ( Mesh[ Int ].x_l <= basis->CollocationPoints[ i ] && basis->CollocationPoints[ i ] < Mesh[ Int ].x_u ) {
							K_ij( i, j ) = integrator.integrate( K_integrand, Mesh[ Int ].x_l, basis->CollocationPoints[ i ] ) +
								integrator.integrate( K_integrand, basis->CollocationPoints[ i ], Mesh[ Int ].x_u ) ;
						} else {
							K_ij( i, j ) = integrator.integrate( K_integrand, Mesh[ Int ].x_l, Mesh[ Int ].x_u );
						}
					} else {
						auto K_integrand = [ & ]( double s, double sc ) {
							return K_singular( basis->CollocationPoints[ i ], s, -sc )*basis->EvaluateBasis( j, s );
						};
						boost::math::quadrature::tanh_sinh<double> integrator( 4, tanhsinh_tol ); // sufficient for 1e-8 precision, without specially-equipped kernel functions
						if ( Mesh[ Int ].x_l <= basis->CollocationPoints[ i ] && basis->CollocationPoints[ i ] < Mesh[ Int ].x_u ) {
							K_ij( i, j ) = integrator.integrate( K_integrand, Mesh[ Int ].x_l, basis->CollocationPoints[ i ] ) +
								integrator.integrate( K_integrand, basis->CollocationPoints[ i ], Mesh[ Int ].x_u ) ;
						} else {
							K_ij( i, j ) = integrator.integrate( K_integrand, Mesh[ Int ].x_l, Mesh[ Int ].x_u );
						}
					}
				}
			MassSolver.compute( Mass );

			fVals = Eigen::VectorXd::Zero( N );
			// Precompute
			#pragma omp parallel for
			for ( Eigen::Index i=0; i < N; i++ )
				fVals( i ) = f( basis->CollocationPoints[ i ] );
		}

		/* 
		 * In terms of the above matrices, the z equation is now an equation for the a_j coefficients:
		 * 
		 * ( M a )_i = g( tau_i, f( tau_i ) + ( K * a )_i )
		 *
		 * for our nonlinear problem we set
		 *
		 * F( a )_i = g( tau_i, f( tau_i ) + ( K*a )_i ) - ( M a )_i
		 */

		int ComputeResidual( N_Vector u, N_Vector F )
		{
			new( &zData ) VecMap( NV_DATA_S( u ), N );
			VecMap output( NV_DATA_S( F ), N );

			Eigen::VectorXd Ma( N ),Ka( N ),Fa( N );
			Ma = Mass * zData;
			Ka = K_ij * zData;
			#pragma omp parallel for
			for ( Eigen::Index i=0; i < N; i++ )
				Fa( i ) = g( basis->CollocationPoints[ i ], fVals( i ) + Ka( i ) ) - Ma( i );
			output = MassSolver.solve( Fa );
			return KIN_SUCCESS;
		}

		void setgPrime( std::function<double( double, double )> gP )
		{
			gPrime = gP;
		};

		// data needs to be allocated already
		// you must have already precomputed the Mass matrix.
		void computeCoefficients( double *data, std::function<double( double )> yZero )
		{
			VecMap x( data, N );
			Eigen::VectorXd b_vals( N );
			#pragma omp parallel for
			for ( Eigen::Index i=0; i<N; ++i )
			{
				double tau_i = basis->CollocationPoints[ i ];
				b_vals( i ) = g( tau_i, yZero( tau_i ) );
			}
			x = MassSolver.solve( b_vals );
		}

		/*
		 * Need to have already set gPrime
		 */

		int ComputeJacobian( N_Vector u, SUNMatrix Jac, N_Vector tmp1, N_Vector tmp2 )
		{
			if ( !gPrime )
				throw std::runtime_error( "Evaluating explicit Jacobian without passing gPrime" );
			Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > JMatrix( SM_DATA_D( Jac ), N, N );
			Eigen::Map<Eigen::VectorXd> tV1( NV_DATA_S( tmp1 ), N );
			Eigen::Map<Eigen::VectorXd> tV2( NV_DATA_S( tmp2 ), N );
			new( &zData ) VecMap( NV_DATA_S( u ), N );

			tV1 = fVals + K_ij * zData;
			#pragma omp parallel for
			for ( Eigen::Index i=0; i < N; i++ )
				tV2( i ) = gPrime( basis->CollocationPoints[ i ], tV1( i ) );

			JMatrix = MassSolver.solve( tV2.asDiagonal() * K_ij ) - Eigen::MatrixXd::Identity( N, N );
			return KIN_SUCCESS;
		}

		void setzData( N_Vector u )
		{
			new( &zData ) VecMap( NV_DATA_S( u ), N );
		}

		static int KINSOL_Hammerstein( N_Vector u, N_Vector F, void* problemData )
		{
			HammersteinEquation *pHammer = reinterpret_cast<HammersteinEquation*>( problemData );
			return pHammer->ComputeResidual( u, F );
		}

		static int KINSOL_HammersteinJacobian( N_Vector u, N_Vector F, SUNMatrix J, void * problemData, N_Vector temp1, N_Vector temp2 )
		{
			HammersteinEquation *pHammer = reinterpret_cast<HammersteinEquation*>( problemData );
			return pHammer->ComputeJacobian( u, J, temp1, temp2 );
		}

		unsigned int getDimension() const {
			return N;
		};

		double EvaluateZ( double x ) {
			double Sum = 0;
			for ( Eigen::Index i = 0; i < N; ++i )
			{
				Sum += zData[ i ] * basis->EvaluateBasis( i, x );
			}
			unsigned int N_Intervals = std::count_if ( Mesh.begin(), Mesh.end(), 
					[ x ]( Interval const& I ){
						return ( x >= I.x_l && x <= I.x_u );
					} );
			if ( N_Intervals == 1 )
				return Sum;
			if ( N_Intervals == 2 )
				return Sum/2.0;
			if ( N_Intervals == 0 )
				return 0.0;
			else {
				throw std::logic_error( "Mesh error detected" );
				return 0.0;
			}
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
				if ( !K_singular ) {
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
				} else {
					auto K_integrand = [ & ]( double s, double sc ) {
						return K_singular( x, s, -sc )*basis->EvaluateBasis( j, s );
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
#endif // HAMMERSTEINEQUATION_HPP
