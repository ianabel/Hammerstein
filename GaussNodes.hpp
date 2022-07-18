#ifndef GAUSSNODES_HPP
#define GAUSSNODES_HPP

#include <vector>
#include <boost/math/special_functions/legendre.hpp>
#include <exception>
#include <boost/math/constants/constants.hpp>

class GaussNodes {
	public:
		using vec = std::vector<double>;
		GaussNodes( vec::size_type N ) {
			auto posNodes = boost::math::legendre_p_zeros<double>(N);
			abscissa = posNodes;
			for ( vec::size_type i = 0; i < posNodes.size(); i++ )
			{
				if ( posNodes[ i ] == 0 )
					continue;
				else
					abscissa.insert( abscissa.begin(), -posNodes[ i ] );
			}

			if ( abscissa.size() != N )
				throw std::logic_error( "Soemwhoe we don't have the right number of Gauss Points" );

			weights.resize( N );
			for ( vec::size_type i = 0; i < weights.size(); i++ )
			{
				double x = abscissa[ i ];
				double p = boost::math::legendre_p_prime(N, x);
				weights[ i ] = 2 / ((1 - x * x) * p * p);
			}
		}

		vec abscissa,weights;
};

#endif // GAUSSNODES_HPP
