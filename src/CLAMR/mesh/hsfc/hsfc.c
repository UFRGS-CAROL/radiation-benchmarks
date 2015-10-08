/* ---------------------------------------------------------------------
Author:     H. Carter Edwards 
            hcedwar@sandia.gov

Copyright:  Copyright (C) 1997   H. Carter Edwards
            Graduate Student
            University of Texas

Re-release: Copyright (C) 2011-2012   H. Carter Edwards

Purpose:    Domain paritioning based upon Hilbert Space-Filling Curve
            ordering.

License:    Re-release under the less-restrictive CLAMR software terms.
            Permitted by email with H. Carter Edwards on 9/13/2011

Disclaimer:

    These routines comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions. See License terms in file 'LICENSE'.
--------------------------------------------------------------------- */

/*----------------------------------------------------------------------
Description:
  Inverse of the Hilbert Space-Filling Curve Map from a 2D or 3D
domain to the 1D domain.  Two different 2D and 3D domains are
supported.

For the routines 'hsfc2d' and 'hsfc3d' the 2D and 3D domains are
defined as follows.
Note that
  *     0   is the minimum value of an unsigned integer
  *   ~(0u) is the maximum value of an unsigned integer - all bits set
thus the 2D and 3D domains are
  *   [0,~(0u)] x [0,~(0u)]
  *   [0,~(0u)] x [0,~(0u)] x [0,~(0u)]
respectively.

For the routines 'fhsfc2d' and 'fhsfc3d' the 2D and 3D domains are
defines as:
  *   [0.0,1.0] x [0.0,1.0]
  *   [0.0,1.0] x [0.0,1.0] x [0.0,1.0]
respectively.

The 1D domain is a multiword (array of unsigned integers) key.
This key is essentially an unsigned integer of an arbitrary
number of bits.  The most significant bit is the leading bit
of the first (0th) word of the key.  The least significant
bit is the trailing bit of the last word.

----------------------------------------------------------------------*/

#include <stdlib.h>
#include <limits.h>

/* Bits per unsigned word */

#define MaxBits ( sizeof(unsigned) * CHAR_BIT )

/*--------------------------------------------------------------------*/
/* 2D Hilbert Space-filling curve */

void hsfc2d(
  unsigned   coord[] , /* IN: Normalized integer coordinates */
  unsigned   nkey ,    /* IN: Word length of key */
  unsigned   key[] )   /* OUT: space-filling curve key */
{
  static int init = 0 ;
  static unsigned char gray_inv[ 2 * 2 ] ;

  const unsigned NKey  = ( 2 < nkey ) ? 2 : (nkey) ;
  const unsigned NBits = ( MaxBits * NKey ) / 2 ;

  unsigned i ;
  unsigned char order[2+2] ;
  unsigned char reflect ;
  
  /* GRAY coding */

  if ( ! init ) {
    unsigned char gray[ 2 * 2 ] ;
    register unsigned k ;
    register unsigned j ;

    gray[0] = 0 ;
    for ( k = 1 ; k < sizeof(gray) ; k <<= 1 ) {
      for ( j = 0 ; j < k ; j++ ) gray[k+j] = k | gray[k-(j+1)] ;
    }
    for ( k = 0 ; k < sizeof(gray) ; k++ ) gray_inv[ gray[k] ] = k ;
    init = 1 ;
  }

  /* Zero out the key */

  for ( i = 0 ; i < NKey ; ++i ) key[i] = 0 ;

  order[0] = 0 ;
  order[1] = 1 ;
  reflect = ( 0 << 0 ) | ( 0 );

  for ( i = 1 ; i <= NBits ; i++ ) {
    const unsigned s = MaxBits - i ;
    const unsigned c = gray_inv[ reflect ^ (
      ( ( ( coord[0] >> s ) & 01 ) << order[0] ) |
      ( ( ( coord[1] >> s ) & 01 ) << order[1] ) ) ];
     
    const unsigned off   = 2 * i ;                   /* Bit offset */
    const unsigned which = off / MaxBits ;           /* Which word to update */
    const unsigned shift = MaxBits - off % MaxBits ; /* Which bits to update */

    /* Set the two bits */

    if ( shift == MaxBits ) { /* Word boundary */
      key[ which - 1 ] |= c ;
    }
    else {
      key[ which ] |= c << shift ;
    }

    /* Determine the recursive quadrant */

    switch( c ) {
    case 3:
      reflect ^= 03 ;
    case 0:
      order[2+0] = order[0] ;
      order[2+1] = order[1] ;
      order[0] = order[2+1] ;
      order[1] = order[2+0] ;
      break ;
    }
  }
}

/*--------------------------------------------------------------------*/
/* 3D Hilbert Space-filling curve */

void hsfc3d(
  unsigned   coord[] , /* IN: Normalized integer coordinates */
  unsigned   nkey ,    /* IN: Word length of 'key' */
  unsigned   key[] )   /* OUT: space-filling curve key */
{
  static int init = 0 ;
  static unsigned char gray_inv[ 2*2*2 ] ;

  const unsigned NKey  = ( 3 < nkey ) ? 3 : (nkey) ;
  const unsigned NBits = ( MaxBits * NKey ) / 3 ;

  unsigned i ;
  unsigned char axis[3+3] ;
  
  /* GRAY coding */

  if ( ! init ) {
    unsigned char gray[ 2*2*2 ] ;
    register unsigned k ;
    register unsigned j ;

    gray[0] = 0 ;
    for ( k = 1 ; k < sizeof(gray) ; k <<= 1 ) {
      for ( j = 0 ; j < k ; j++ ) gray[k+j] = k | gray[k-(j+1)] ;
    }
    for ( k = 0 ; k < sizeof(gray) ; k++ ) gray_inv[ gray[k] ] = k ;
    init = 1 ;
  }

  /* Zero out the key */

  for ( i = 0 ; i < NKey ; ++i ) key[i] = 0 ;

  axis[0] = 0 << 1 ;
  axis[1] = 1 << 1 ;
  axis[2] = 2 << 1 ;

  for ( i = 1 ; i <= NBits ; i++ ) {
    const unsigned s = MaxBits - i ;
    const unsigned c = gray_inv[
      (((( coord[ axis[0] >> 1 ] >> s ) ^ axis[0] ) & 01 ) << 0 ) |
      (((( coord[ axis[1] >> 1 ] >> s ) ^ axis[1] ) & 01 ) << 1 ) |
      (((( coord[ axis[2] >> 1 ] >> s ) ^ axis[2] ) & 01 ) << 2 ) ];
    unsigned n ;

    /* Set the 3bits */

    for ( n = 0 ; n < 3 ; ++n ) {
      const unsigned bit   = 01 & ( c >> ( 2 - n ) );  /* Bit value  */
      const unsigned off   = 3 * i + n ;               /* Bit offset */
      const unsigned which = off / MaxBits ;           /* Which word */
      const unsigned shift = MaxBits - off % MaxBits ; /* Which bits */

      if ( MaxBits == shift ) { /* Word boundary */
        key[ which - 1 ] |= bit ;
      }
      else {
        key[ which ] |= bit << shift ;
      }
    }

    /* Determine the recursive quadrant */

    axis[3+0] = axis[0] ;
    axis[3+1] = axis[1] ;
    axis[3+2] = axis[2] ;

    switch( c ) {
    case 0:
      axis[0] = axis[3+2];
      axis[1] = axis[3+1];
      axis[2] = axis[3+0];
      break ;
    case 1:
      axis[0] = axis[3+0];
      axis[1] = axis[3+2];
      axis[2] = axis[3+1];
      break ;
    case 2:
      axis[0] = axis[3+0];
      axis[1] = axis[3+1];
      axis[2] = axis[3+2];
      break ;
    case 3:
      axis[0] = axis[3+2] ^ 01 ;
      axis[1] = axis[3+0] ^ 01 ;
      axis[2] = axis[3+1];
      break ;
    case 4:
      axis[0] = axis[3+2];
      axis[1] = axis[3+0] ^ 01 ;
      axis[2] = axis[3+1] ^ 01 ;
      break ;
    case 5:
      axis[0] = axis[3+0];
      axis[1] = axis[3+1];
      axis[2] = axis[3+2];
      break ;
    case 6:
      axis[0] = axis[3+0];
      axis[1] = axis[3+2] ^ 01 ;
      axis[2] = axis[3+1] ^ 01 ;
      break ;
    case 7:
      axis[0] = axis[3+2] ^ 01 ;
      axis[1] = axis[3+1];
      axis[2] = axis[3+0] ^ 01 ;
      break ;
    default:
      exit(-1);
    }
  }
}

/*--------------------------------------------------------------------*/

void fhsfc2d(
  double     coord[] , /* IN: Normalized floating point coordinates */
  unsigned   nkey ,    /* IN: Word length of key */
  unsigned   key[] )   /* OUT: space-filling curve key */
{
  const double imax = ~(0u);
  unsigned c[2] ;
  c[0] = coord[0] * imax ;
  c[1] = coord[1] * imax ;
  hsfc2d( c , nkey , key );
}

void fhsfc3d(
  double     coord[] , /* IN: Normalized floating point coordinates */
  unsigned   nkey ,    /* IN: Word length of key */
  unsigned   key[] )   /* OUT: space-filling curve key */
{
  const double imax = ~(0u);
  unsigned c[3] ;
  c[0] = coord[0] * imax ;
  c[1] = coord[1] * imax ;
  c[2] = coord[2] * imax ;
  hsfc3d( c , nkey , key );
}

