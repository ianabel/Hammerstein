#ifndef COLLOCATIONAPPROX_H
#define COLLOCATIONAPPROX_H

#include <vector>
#include <map>

#include <memory>
#include <algorithm>
#include <boost/math/special_functions/legendre.hpp>
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
			: Mesh( _m ),k( Order ),gauss( Order + 1 )
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

		~CollocationBasis() {};

		double EvaluateLegendreBasis( Interval const & I, Eigen::Index i, double x ) const
		{
			if ( i > k )
				throw std::runtime_error( "Index bigger than legendre order requested" );
			if ( x < I.x_l || x > I.x_u )	
				return 0;
			double y = 2.0*( x - I.x_l )/I.h() - 1.0;
			// double eval = ::sqrt( ( 2.0* i + 1.0 )/( I.h() ) ) * boost::math::legendre_p( i, y );
			double wgt = ::sqrt( ( 2.0* i + 1.0 )/( I.h() ) );
			double eval = boost::math::legendre_p( i, y ); // Don't need orthonormal basis for this, just a basis (still orthogonal, but that's unneeded)
			return eval;
		};

		double EvaluateLagrangeGaussBasis( Eigen::Index i, double x ) const
		{
			Eigen::Index m = i/( k+1 );
			Interval const &I = Mesh[ m ];
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

		double EvaluateLagrangeBasis( Eigen::Index i, double x ) const
		{
			Eigen::Index m = i/( k+1 );
			Interval const &I = Mesh[ m ];
			if ( I.contains( x ) ) {
				// form lagrange polynomial j = i%(k+1) using the lagrange nodal points in the interval I
				double result = 1.0;
				Eigen::Index j = i %( k+1 );
				for ( Eigen::Index l=0; l < k+1; l++ )
				{
					if ( l == j )
						continue;
					else
						result *= ( x - *** )/( CollocationPoints[ i ] - CollocationPoints[ m*( k + 1 ) + l ] );
				}
				if ( !::isfinite( result ) )
					throw std::runtime_error( "Blergh" );
				return result;
			} else {
				return 0.0;
			}
		}
		double EvaluateBasis( Eigen::Index i, double x ) {
			if ( i > N )
				throw std::runtime_error( "No basis function of index i exists" );
			return EvaluateLagrangeBasis( i, x );
		}

		std::vector<double> CollocationPoints;
		GaussNodes gauss;
		Mesh_t const& Mesh;

	private:
		unsigned int k,N;

};


#endif // COLLOCATIONAPPROX_H
