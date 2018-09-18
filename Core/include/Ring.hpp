/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*                                                                  */
/*     Aircraft Plume Chemistry, Emission and Microphysics Model    */
/*                             (APCEMM)                             */
/*                                                                  */
/* Ring Header File                                                 */
/*                                                                  */
/* Author               : Thibaud M. Fritz                          */
/* Time                 : 8/12/2018                                 */
/* File                 : Ring.hpp                                  */
/* Working directory    : /home/fritzt/APCEMM-SourceCode            */
/*                                                                  */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#ifndef RING_H_INCLUDED
#define RING_H_INCLUDED

#include <iostream>

class Ring
{

    public:

        Ring( );
        Ring( double a, double b );
        Ring( const Ring &r );
        Ring& operator=( const Ring &r );
        ~Ring( );
        void Assign( double a, double b );
        void Print( ) const;
        double getHAxis( ) const;
        double getVAxis( ) const;

    protected:

        double horizontalAxis;
        double verticalAxis;

    private:

};

#endif /* RING_H_INCLUDED */
