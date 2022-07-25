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

	I = f( tau )*::log( ::fabs( ( b - tau )/( tau - a ) ) );

	if ( tau - a < b - tau ) {
		delta = tau - a;
		if ( b - tau - delta > 1e-10 ) // g is a bounded function, so we can ignore if the interval is small
			I += Int.integrate( g, tau + delta, b );
	} else if ( b - tau < tau - a ) {
		delta = b - tau;
		if ( tau - delta - a > 1e-10 ) // as above
			I += Int.integrate( g, a, tau - delta );
	} else if ( b - tau == tau - a ) {
		delta = b - tau;
		I = 0;
	}
	
	I += Int.integrate( h, 0, delta );

	return I;

}

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

		enum BasisType {
			DGLegendre,
			DGLagrange,
			Lagrange,
			GlobalChebyshev,
			Hermite
		} b_type;

		/* Determines numerical resolution and precomputes the needed values */
		void SetResolutionAndPrecompute( unsigned int NIntervals, unsigned int Order, BasisType basisType = DGLegendre, bool gradedMesh = false, double gradingAlpha = 1.0 )
		{
			N_Intervals = NIntervals;
			N_Gauss = Order + 1;

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
			
			switch( basisType )
			{
				case DGLegendre:
					basis = new DGLegendreBasis( Mesh, Order );
					break;
				case DGLagrange:
					basis = new DGLegendreBasis( Mesh, Order );
					break;
				case GlobalChebyshev:
					basis = new GlobalChebyshevBasis( Mesh, Order );
					break;
				case Lagrange:
					basis = new LagrangeBasis( Mesh, Order );
					break;
				case Hermite:
					basis = new HermiteBasis( Mesh );
					break;
				default:
					basis = nullptr;
			}
			
			if ( basis == nullptr )
				throw std::runtime_error( "Could not instantiate the requested collocation basis" );
			
			N = basis->DegreesOfFreedom();

			Mass = Eigen::MatrixXd::Zero( N, N );
			K_ij = Eigen::MatrixXd::Zero( N, N );

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

					// Basis elements are rarely globally supported.
					// Integrate only the non-zero region.
					Interval basisSupport = basis->supportOfElement( j );
					if ( !K_singular ) {
						auto K_integrand = [ & ]( double s ) {
							return K( basis->CollocationPoints[ i ], s )*basis->EvaluateBasis( j, s );
						};
						boost::math::quadrature::tanh_sinh<double> integrator( 4, tanhsinh_tol ); // sufficient for 1e-8 precision, without specially-equipped kernel functions
						if ( basisSupport.x_l <= basis->CollocationPoints[ i ] && basis->CollocationPoints[ i ] < basisSupport.x_u ) {
							K_ij( i, j ) = integrator.integrate( K_integrand, basisSupport.x_l, basis->CollocationPoints[ i ] ) +
								integrator.integrate( K_integrand, basis->CollocationPoints[ i ], basisSupport.x_u ) ;
						} else {
							K_ij( i, j ) = integrator.integrate( K_integrand, basisSupport.x_l, basisSupport.x_u );
						}
					} else {
						auto K_integrand = [ & ]( double s, double sc ) {
							return K_singular( basis->CollocationPoints[ i ], s, -sc )*basis->EvaluateBasis( j, s );
						};
						boost::math::quadrature::tanh_sinh<double> integrator( 4, tanhsinh_tol ); // sufficient for 1e-8 precision, without specially-equipped kernel functions
						if ( basisSupport.x_l <= basis->CollocationPoints[ i ] && basis->CollocationPoints[ i ] < basisSupport.x_u ) {
							if ( basisSupport.x_l == basis->CollocationPoints[ i ] || basisSupport.x_u == basis->CollocationPoints[ i ] )
								K_ij( i, j ) = integrator.integrate( K_integrand, basisSupport.x_l, basisSupport.x_u ) ;
							else
								K_ij( i, j ) = integrator.integrate( K_integrand, basisSupport.x_l, basis->CollocationPoints[ i ] ) +
									integrator.integrate( K_integrand, basis->CollocationPoints[ i ], basisSupport.x_u ) ;
						} else {
							K_ij( i, j ) = integrator.integrate( K_integrand, basisSupport.x_l, basisSupport.x_u );
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

		// data needs to be allocated already
		// you must have already precomputed the Mass matrix.
		Eigen::VectorXd computeZCoefficients( std::function<double( double )> zZero )
		{
			Eigen::VectorXd b_vals( N );
			#pragma omp parallel for
			for ( Eigen::Index i=0; i<N; ++i )
			{
				double tau_i = basis->CollocationPoints[ i ];
				b_vals( i ) = zZero( tau_i );
			}
			return MassSolver.solve( b_vals );
		}

		// Apply an integral operator to z, given in the form
		//
		// Iz = f*Int[ (K_pv(s)/(x-s) + K_I(x,s))*z, {x,a,b} ]
		// 
		// Returns Iz as a vector in the existing basis

		Eigen::VectorXd applyIntegralOperator( std::function<double( double, double )> K_I, std::function<double( double )> K_pv = nullptr, std::function<double( double )> f = nullptr ) const
		{
			return applyIntegralOperator( [ & ]( double x, double s, double ){return K_I( x,s );}, K_pv, f );
		}

		Eigen::VectorXd applyIntegralOperator( std::function<double( double, double, double )> K_I, std::function<double( double )> K_pv = nullptr, std::function<double( double )> f = nullptr ) const
		{
			Eigen::MatrixXd I_ij    = Eigen::MatrixXd::Zero( N, N );
			Eigen::MatrixXd I_pv_ij = Eigen::MatrixXd::Zero( N, N );
			Eigen::VectorXd result = Eigen::VectorXd::Zero( N );
			#pragma omp parallel for
			for ( Eigen::Index i=0; i < N; i++ )
				for ( Eigen::Index j=0; j < N; j++ )
				{
					// Basis elements are rarely globally supported.
					// Integrate only the non-zero region.
					Interval basisSupport = basis->supportOfElement( j );

					auto K_integrand = [ & ]( double s, double sc ) {
						return K_I( basis->CollocationPoints[ i ], s, -sc )*basis->EvaluateBasis( j, s );
					};
					boost::math::quadrature::tanh_sinh<double> integrator( 4, tanhsinh_tol ); // sufficient for 1e-8 precision, without specially-equipped kernel functions
					if ( basisSupport.x_l <= basis->CollocationPoints[ i ] && basis->CollocationPoints[ i ] < basisSupport.x_u ) {
						if ( basisSupport.x_l == basis->CollocationPoints[ i ] || basisSupport.x_u == basis->CollocationPoints[ i ] )
							I_ij( i, j ) = integrator.integrate( K_integrand, basisSupport.x_l, basisSupport.x_u ) ;
						else
							I_ij( i, j ) = integrator.integrate( K_integrand, basisSupport.x_l, basis->CollocationPoints[ i ] ) +
								integrator.integrate( K_integrand, basis->CollocationPoints[ i ], basisSupport.x_u ) ;
					} else {
						I_ij( i, j ) = integrator.integrate( K_integrand, basisSupport.x_l, basisSupport.x_u );
					}

					// Do the Cauchy Principal Value integral
					if ( K_pv ) {
						
						if ( basisSupport.x_l <= basis->CollocationPoints[ i ] && basis->CollocationPoints[ i ] < basisSupport.x_u ) {
							// Really is a PV integral on this interval
							auto K_pv_integrand = [ & ]( double s ){
								return K_pv( s )*basis->EvaluateBasis( j, s );
							};
							I_pv_ij( i, j ) = CauchyPV( K_pv_integrand, basisSupport.x_l, basisSupport.x_u, basis->CollocationPoints[ i ] );
						} else {
							// not really a PV integral, everything is bounded
							boost::math::quadrature::tanh_sinh<double> integrator( 4, tanhsinh_tol ); // sufficient for 1e-8 precision, without specially-equipped kernel functions
							auto K_pv_bounded = [ & ]( double s ){
								return ( K_pv( s )/( s-basis->CollocationPoints[ i ] ) )*basis->EvaluateBasis( j, s );
							};
							I_pv_ij( i, j ) = integrator.integrate( K_pv_bounded, basisSupport.x_l, basisSupport.x_u );
						}
					}
					
					if ( f ) {
						I_ij( i, j ) *= f( basis->CollocationPoints[ i ] );
						I_pv_ij( i, j ) *= f( basis->CollocationPoints[ i ] );
					}

				}

			result = MassSolver.solve( I_ij * zData + I_pv_ij * zData );

			return result;
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

		void setzData( Eigen::VectorXd &u )
		{
			if ( u.size() != N )
				throw std::runtime_error( "Cannot set z data to a vector of the wrong size" );
			else
				new( &zData ) VecMap( u.data(), N );
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

		double Evaluate( Eigen::VectorXd const& data, double x ) const {
			double Sum = 0;
			#pragma omp parallel for reduction( +: Sum )
			for ( Eigen::Index i = 0; i < N; ++i )
			{
				Sum += data[ i ] * basis->EvaluateBasis( i, x );
			}
			return Sum;
		};

		double EvaluateZ( double x ) const {
			return Evaluate( zData, x );
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
				// As above, only compute non-zero integrals
				Interval const& basisSupport = basis->supportOfElement( j );
				if ( !K_singular ) {
					auto K_integrand = [ & ]( double s ) {
						return K( x, s )*basis->EvaluateBasis( j, s );
					};

					boost::math::quadrature::tanh_sinh<double> integrator( 4, tanhsinh_tol ); // sufficient for 1e-8 precision, without specially-equipped kernel functions
					if ( basisSupport.x_l <= x && x < basisSupport.x_u ) {
						double I_l,I_u;
						// If we're sampling very close to a meshpoint we can get an error
						if ( ( x - basisSupport.x_l ) < tanhsinh_tol )
							I_l = 0;
						else
							I_l = integrator.integrate( K_integrand, basisSupport.x_l, x );
						if ( ( basisSupport.x_u - x ) < tanhsinh_tol )
							I_u = 0;
						else
							I_u = integrator.integrate( K_integrand, x, basisSupport.x_u ) ;
						KIntegral = I_l + I_u;
					} else {
						KIntegral = integrator.integrate( K_integrand, basisSupport.x_l, basisSupport.x_u );
					}
				} else {
					auto K_integrand = [ & ]( double s, double sc ) {
						return K_singular( x, s, -sc )*basis->EvaluateBasis( j, s );
					};

					boost::math::quadrature::tanh_sinh<double> integrator( 4, tanhsinh_tol ); // sufficient for 1e-8 precision, without specially-equipped kernel functions
					if ( basisSupport.x_l <= x && x < basisSupport.x_u ) {
						double I_l,I_u;
						// If we're sampling very close to a meshpoint we can get an error
						if ( ( x - basisSupport.x_l ) < tanhsinh_tol )
							I_l = 0;
						else
							I_l = integrator.integrate( K_integrand, basisSupport.x_l, x );
						if ( ( basisSupport.x_u - x ) < tanhsinh_tol )
							I_u = 0;
						else
							I_u = integrator.integrate( K_integrand, x, basisSupport.x_u ) ;
						KIntegral = I_l + I_u;
					} else {
						KIntegral = integrator.integrate( K_integrand, basisSupport.x_l, basisSupport.x_u );
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
