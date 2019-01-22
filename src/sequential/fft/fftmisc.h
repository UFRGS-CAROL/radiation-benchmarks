int IsPowerOfTwo ( unsigned x );
unsigned NumberOfBitsNeeded ( unsigned PowerOfTwo );
unsigned ReverseBits ( unsigned index, unsigned NumBits );

/*
**   The following function returns an "abstract frequency" of a
**   given index into a buffer with a given number of frequency samples.
**   Multiply return value by sampling rate to get frequency expressed in Hz.
*/
double Index_to_frequency ( unsigned NumSamples, unsigned Index );
