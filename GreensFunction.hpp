#ifndef GREENSFUNCTION_HPP
#define GREENSFUNCTION_HPP

double GradShafranovGreensFunction( double, double, double, double );
double GradShafranovGreensFunction1D(double R, double R_star, double delta = 0.0 );
double DerivativeGreensFunction1D( double R, double R_star );
double DerivativeGreensFunction1D_Weak( double x, double R_star );
double DerivativeGreensFunction1D_Residue( double R_star );


#endif // GREENSFUNCTION_HPP
