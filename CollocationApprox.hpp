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

		virtual Interval const& supportOfElement( unsigned int i ) const = 0;

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

		virtual Interval const& supportOfElement( unsigned int i ) const 
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

		virtual Interval const& supportOfElement( unsigned int i ) const {
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

		virtual Interval const& supportOfElement( unsigned int i ) const {
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

		virtual Interval const& supportOfElement( unsigned int i ) const {
			return domain;
		}
};

// Continuous finite elements
class LagrangeBasis : public CollocationBasis
{
	public:
		GaussNodes gauss;
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
					CollocationPoints[ i*k + j ] = Mesh[ i ].x_l + Mesh[ i ].h() * ( j/ k );
				}
			}
			CollocationPoints[ Mesh.size() * k ] = Mesh.back().x_u;
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

		virtual Interval const& supportOfElement( unsigned int i ) const {
			return Mesh[ i / ( k + 1 ) ];
		}
}
#endif // COLLOCATIONAPPROX_H
