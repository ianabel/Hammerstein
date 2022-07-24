#ifndef COLLOCATIONAPPROX_H
#define COLLOCATIONAPPROX_H

#include <vector>
#include <map>

#include <memory>
#include <algorithm>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/chebyshev.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "GaussNodes.hpp"

/*
 * Classes encpasulating a 1-D collocation approximation on discontinuous piecewise polynomials
 */

class Interval
{
	public:
		Interval( double a, double b ) 
		{
			x_l = ( a > b ) ? b : a;
			x_u = ( a > b ) ? a : b;
		};
		Interval( Interval const &I )
		{
			x_l = I.x_l;
			x_u = I.x_u;
		};

		double x_l,x_u;
		bool contains( double x ) const { return ( x_l <= x ) && ( x <= x_u );};
		double h() const { return ( x_u - x_l );};
		// Maps to [0,1]
		double TInverse( double x ) const { return ( x - x_l )/( x_u - x_l ); };
};


class CollocationBasis
{
	public:
		typedef std::vector<Interval> Mesh_t;
		CollocationBasis( Mesh_t const& _m, unsigned int Order )
			: Mesh( _m ),k( Order )
		{
		};

		virtual ~CollocationBasis() {};

		virtual double EvaluateBasis( Eigen::Index i, double x ) const = 0;

		std::vector<double> CollocationPoints;
		Mesh_t const& Mesh;

		unsigned int DegreesOfFreedom() const {
			return N;
		}

		virtual Interval supportOfElement( unsigned int i ) const = 0;

	protected:
		unsigned int k,N;

};

class DGLegendreBasis : public CollocationBasis 
{
	public:
		DGLegendreBasis( Mesh_t const& _m, unsigned int Order )
			: CollocationBasis( _m, Order ),gauss( Order + 1 )
		{
			N = Mesh.size() * ( k + 1 );
			CollocationPoints.resize( N );

			for ( Eigen::Index i=0; i < Mesh.size(); i++ )
			{
				// Translate gauss nodes to intervals
				for ( Eigen::Index j=0; j < k + 1; j++ )
				{
					CollocationPoints[ i*( k + 1 ) + j ] 
						= ( Mesh[ i ].x_l + Mesh[ i ].x_u + Mesh[ i ].h() * gauss.abscissa[ j ] )/2.0;
				}
			}
		};
		GaussNodes gauss;
		double EvaluateLegendreBasis( Interval const & I, Eigen::Index i, double x ) const
		{
			if ( i > k )
				throw std::runtime_error( "Index bigger than legendre order requested" );
			if ( x < I.x_l || x > I.x_u )	
				return 0;
			if ( x == I.x_u && x < Mesh.back().x_u )
				return 0;
			double y = 2.0*( x - I.x_l )/I.h() - 1.0;
			// double eval = ::sqrt( ( 2.0* i + 1.0 )/( I.h() ) ) * boost::math::legendre_p( i, y );
			double wgt = ::sqrt( ( 2.0* i + 1.0 )/( I.h() ) );
			double eval = boost::math::legendre_p( i, y ); // Don't need orthonormal basis for this, just a basis (still orthogonal, but that's unneeded)
			return eval;
		};

		virtual double EvaluateBasis( Eigen::Index i, double x ) const
		{
			return EvaluateLegendreBasis( Mesh[ i / ( k + 1 ) ], i % ( k + 1 ), x );
		}

		virtual Interval supportOfElement( unsigned int i ) const 
		{
			return Mesh[ i / ( k + 1 ) ];
		}
};

class NodalLagrangeBasis : public CollocationBasis 
{
	public:
		GaussNodes gauss;
		NodalLagrangeBasis( Mesh_t const& _m, unsigned int Order )
			: CollocationBasis( _m, Order ),gauss( Order + 1 )
		{
			N = Mesh.size() * ( k + 1 );
			CollocationPoints.resize( N );

			for ( Eigen::Index i=0; i < Mesh.size(); i++ )
			{
				// Translate gauss nodes to intervals
				for ( Eigen::Index j=0; j < k + 1; j++ )
				{
					CollocationPoints[ i*( k + 1 ) + j ] 
						= ( Mesh[ i ].x_l + Mesh[ i ].x_u + Mesh[ i ].h() * gauss.abscissa[ j ] )/2.0;
				}
			}
		};

		virtual double EvaluateBasis( Eigen::Index i, double x ) const 
		{
			Eigen::Index m = i/( k+1 );
			Interval const &I = Mesh[ m ];
			if ( x == I.x_u && x < Mesh.back().x_u )
				return 0; // ensure only one basis function is ever non-zero at a cell boundary
			if ( I.contains( x ) ) {
				// form lagrange polynomial j = i%(k+1) using the collocation points in the interval I
				double result = 1.0;
				Eigen::Index j = i %( k+1 );
				for ( Eigen::Index l=0; l < k+1; l++ )
				{
					if ( l == j )
						continue;
					else
						result *= ( x - CollocationPoints[ m*( k + 1 ) + l ])/( CollocationPoints[ i ] - CollocationPoints[ m*( k + 1 ) + l ] );
				}
				if ( !::isfinite( result ) )
					throw std::runtime_error( "Blergh" );
				return result;
			} else {
				return 0.0;
			}
		}

		virtual Interval supportOfElement( unsigned int i ) const {
			return Mesh[ i / ( k + 1 ) ];
		}
};

class DGLagrangeBasis : public CollocationBasis 
{
	public:
		GaussNodes gauss;
		DGLagrangeBasis( Mesh_t const& _m, unsigned int Order )
			: CollocationBasis( _m, Order ),gauss( Order + 1 )
		{
			N = Mesh.size() * ( k + 1 );
			CollocationPoints.resize( N );

			for ( Eigen::Index i=0; i < Mesh.size(); i++ )
			{
				// Translate gauss nodes to intervals
				for ( Eigen::Index j=0; j < k + 1; j++ )
				{
					CollocationPoints[ i*( k + 1 ) + j ] 
						= ( Mesh[ i ].x_l + Mesh[ i ].x_u + Mesh[ i ].h() * gauss.abscissa[ j ] )/2.0;
				}
			}
		};


		virtual double EvaluateBasis( Eigen::Index i, double x ) const 
		{
			Eigen::Index m = i/( k+1 );
			Interval const &I = Mesh[ m ];
			if ( x == I.x_u && x < Mesh.back().x_u )
				return 0; // ensure only one basis function is ever non-zero at a cell boundary
			if ( I.contains( x ) ) {
				// form lagrange polynomial j = i%(k+1) using the lagrange nodal points in the interval I
				double result = 1.0;
				Eigen::Index j = i %( k+1 );
				double node_j = Mesh[ m ].x_l + ( static_cast<double>( j )/static_cast<double>( k ))*( Mesh[ m ].x_u - Mesh[ m ].x_l );

				for ( Eigen::Index l=0; l < k+1; l++ )
				{
					double node_l = Mesh[ m ].x_l + ( static_cast<double>( l )/static_cast<double>( k ))*( Mesh[ m ].x_u - Mesh[ m ].x_l );
					if ( l == j )
						continue;
					else
						result *= ( x - node_l )/( node_j - node_l );
				}
				if ( !::isfinite( result ) )
					throw std::runtime_error( "Blergh" );
				return result;
			} else {
				return 0.0;
			}
		}

		virtual Interval supportOfElement( unsigned int i ) const {
			return Mesh[ i / ( k + 1 ) ];
		}
};

class GlobalChebyshevBasis : public CollocationBasis
{
	private:
		double a,b;
		Interval domain;
	public:
		GlobalChebyshevBasis( Mesh_t const& _m, unsigned int Order )
			: CollocationBasis( _m, Order ),a( _m.front().x_l ),b( _m.back().x_u ),domain( _m.front().x_l, _m.back().x_u )
		{
			N = Mesh.size() * ( k );
			CollocationPoints.resize( N );

			for ( Eigen::Index i=0; i < N; i++ )
			{
				// Translate chebychev nodes to interval [a,b]
				CollocationPoints[ i ] = ( a + b )/2.0 - ( ::cos( i * M_PI / ( N - 1 ) ) ) * domain.h()/2.0;
			}
		};


		virtual double EvaluateBasis( Eigen::Index i, double x ) const 
		{
			return boost::math::chebyshev_t( i, ( 2.0*x - ( a+b ) )/( b - a ) );
		}

		virtual Interval supportOfElement( unsigned int i ) const {
			return domain;
		}
};

// Continuous finite elements
class LagrangeBasis : public CollocationBasis
{
	public:
		LagrangeBasis( Mesh_t const& _m, unsigned int Order )
			: CollocationBasis( _m, Order )
		{
			// There are only (|Mesh| * k) + 1 continuous finite elements
			// rather than |Mesh| * ( k + 1 ) discontinuous ones
			N = Mesh.size() * k + 1;
			CollocationPoints.resize( N );

			for ( Eigen::Index i=0; i < Mesh.size(); i++ )
			{
				// Translate uniform lagrange nodes
				for ( Eigen::Index j=0; j < k; j++ )
				{
					CollocationPoints[ i*k + j ] = Mesh[ i ].x_l + Mesh[ i ].h() * ( static_cast<double>( j )/ static_cast<double>( k ) );
				}
			}
			CollocationPoints[ Mesh.size() * k ] = Mesh.back().x_u;
		};


		virtual double EvaluateBasis( Eigen::Index j, double x ) const 
		{
			// Handle endpoints first
			if ( j == Mesh.size()*k ) {
				Interval const&I = Mesh.back();
				if ( I.contains( x ) ) {
					// This should be Lagrange polynomial on the final interval with node x=b
					double result = 1.0;
					double node = I.x_u;

					for ( Eigen::Index l=0; l < k; l++ )
					{
						double node_l = I.x_l + ( static_cast<double>( l )/static_cast<double>( k ))*( I.x_u - I.x_l );
						result *= ( x - node_l )/( node - node_l );
					}
					if ( !::isfinite( result ) )
						throw std::runtime_error( "Blergh" );
					return result;
				} else {
					return 0.0;
				}
			} else if ( j == 0 ) {
				Interval const&I = Mesh.front();
				if ( I.contains( x ) ) {
					// This should be Lagrange polynomial on the final interval with node x=b
					double result = 1.0;
					double node = I.x_l;

					for ( Eigen::Index l=1; l < k+1; l++ )
					{
						double node_l = I.x_l + ( static_cast<double>( l )/static_cast<double>( k ))*( I.x_u - I.x_l );
						result *= ( x - node_l )/( node - node_l );
					}
					if ( !::isfinite( result ) )
						throw std::runtime_error( "Blergh" );
					return result;
				} else {
					return 0.0;
				}
			}

			Eigen::Index i = j/k;
			Eigen::Index m = j % k;
			Interval const &I = Mesh[ i ];

			if ( m > 0 ) {
				if ( I.contains( x ) ) {
					// form lagrange polynomial j = i%(k+1) using the lagrange nodal points in the interval I
					double result = 1.0;
					double node_m = I.x_l + ( static_cast<double>( m )/static_cast<double>( k ))*( I.x_u - I.x_l );

					for ( Eigen::Index l=0; l < k+1; l++ )
					{
						double node_l = I.x_l + ( static_cast<double>( l )/static_cast<double>( k ))*( I.x_u - I.x_l );
						if ( l == m )
							continue;
						else
							result *= ( x - node_l )/( node_m - node_l );
					}
					if ( !::isfinite( result ) )
						throw std::runtime_error( "Blergh" );
					return result;
				} else {
					return 0.0;
				}
			}

			if ( m == 0 ) {
				Interval const& Iminus = Mesh[ i - 1 ]; // i cannot be 0, because then m=0 => j=0 and that was done earlier
				if ( Iminus.contains( x ) ) {
					// This should be Lagrange polynomial on the i-1 interval with node x = Mesh[ i - 1 ].x_u == Mesh[i].x_l
					double result = 1.0;
					double node = Iminus.x_u;

					for ( Eigen::Index l=0; l < k; l++ )
					{
						double node_l = Iminus.x_l + ( static_cast<double>( l )/static_cast<double>( k ))*( Iminus.x_u - Iminus.x_l );
						result *= ( x - node_l )/( node - node_l );
					}
					if ( !::isfinite( result ) )
						throw std::runtime_error( "Blergh" );
					return result;
				} else if ( I.contains( x ) ){
					double result = 1.0;
					double node = I.x_l;

					for ( Eigen::Index l=1; l < k+1; l++ )
					{
						double node_l = I.x_l + ( static_cast<double>( l )/static_cast<double>( k ))*( I.x_u - I.x_l );
						result *= ( x - node_l )/( node - node_l );
					}
					if ( !::isfinite( result ) )
						throw std::runtime_error( "Blergh" );
					return result;
				} else {
					return 0.0;
				}
			}
			if ( m<0 )
				throw std::runtime_error( "WHAT UNGODLY HELL IS A NEGATIVE REMAINDER" );
			return std::nan( "" );
		}

		virtual Interval supportOfElement( unsigned int i ) const {
			if ( i % k == 0 && i > 0 && i < Mesh.size()*k ) 
				return Interval( Mesh[ ( i / k ) - 1 ].x_l, Mesh[ i / k ].x_u );
			else if ( i == Mesh.size()*k )
				return Mesh.back();
			else
				return Mesh[ i / k ];
		}
};


// Continuous finite elements
class HermiteBasis : public CollocationBasis
{
	public:
		HermiteBasis( Mesh_t const& _m )
			: CollocationBasis( _m, 0 )
		{
			N = 2 * ( Mesh.size()  + 1 );
			CollocationPoints.resize( N );

			CollocationPoints[ 0 ] = Mesh.front().x_l;
			CollocationPoints[ 1 ] = Mesh.front().x_l + Mesh.front().h() * 0.25;
			CollocationPoints[ 2 ] = Mesh.front().x_l + Mesh.front().h() * 0.50;

			for ( Eigen::Index i=1; i < Mesh.size()-1; i++ )
			{
				CollocationPoints[ 2*i + 1 ] = Mesh[ i ].x_l;
				CollocationPoints[ 2*i + 2 ] = Mesh[ i ].x_l + Mesh[ i ].h()*(1.0/2.0);
			}


			CollocationPoints[ N - 3 ] = Mesh.back().x_u - Mesh.back().h()*0.50;
			CollocationPoints[ N - 2 ] = Mesh.back().x_u - Mesh.back().h()*0.25;
			CollocationPoints[ N - 1 ] = Mesh.back().x_u;

		};


		virtual double EvaluateBasis( Eigen::Index j, double x ) const 
		{
			// Handle endpoints first
			if ( j < 3 ) {
				Interval const&I = Mesh.front();
				if ( I.contains( x ) ) {
					switch ( j ) {
						case 0:
							return t1( I.TInverse( x ) );
							break;
						case 1:
							return t2( I.TInverse( x ) );
							break;
						case 2:
							return t4( I.TInverse( x ) );
							break;
						default:
							return std::nan( "" );
					}
				} else {
					return 0.0;
				}
			} else if ( j > N - 4 ) {
				Interval const&I = Mesh.back();
				if ( I.contains( x ) ) {
					switch ( N - j - 1 ) {
						case 0:
							return t3( I.TInverse( x ) );
							break;
						case 1:
							return t2( I.TInverse( x ) );
							break;
						case 2:
							return t4( I.TInverse( x ) );
							break;
						default:
							return std::nan( "" );
					}
				} else {
					return 0.0;
				}
			}

			Eigen::Index i = ( j - 1 )/2.0;
			Eigen::Index m = j % 2;
			Interval const &I = Mesh[ i ];
			Interval const& Iminus = Mesh[ i - 1 ]; // i cannot be 0, because then m=0 => j=0 and that was done earlier

			if ( m == 0 ) {
				if ( Iminus.contains( x ) ) {
					return t3( Iminus.TInverse( x ) );
				} else if ( I.contains( x ) ){
					return t1( I.TInverse( x ) );
				} else {
					return 0.0;
				}
			}

			if ( m == 1 ) {
				if ( Iminus.contains( x ) ) {
					return Iminus.h()*t4( Iminus.TInverse( x ) );
				} else if ( I.contains( x ) ){
					return I.h()*t2( I.TInverse( x ) );
				} else {
					return 0.0;
				}
			}
			return std::nan( "" );
		}

		virtual Interval supportOfElement( unsigned int i ) const {
			if ( i > 2 && i < N - 3 )
				return Interval( Mesh[ ( i - 1 )/2 - 1 ].x_l, Mesh[ ( i - 1 )/2 ].x_u );
			else if ( i < 2 )
				return Mesh.front();
			else
				return Mesh.back();
		}
	private:
		double t1( double t ) const { return ( 2.0*t + 1.0 )*( t - 1.0 )*( t - 1.0 ); };
		double t2( double t ) const { return t*( t - 1.0 )*( t - 1.0 ); };
		double t3( double t ) const { return ( 3.0 - 2.0*t )*t*t; };
		double t4( double t ) const { return ( t - 1.0 )*t*t; };
};

#endif // COLLOCATIONAPPROX_H
