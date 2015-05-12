#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace std;

int main( int argc, char * argv[] )
{
    if ( argc < 2 )
    {
        cerr << "Error: Specify a number of objects to generate.\n";
        exit( 1 );
    }
    int nObj = atoi( argv[1] );
    int nFeat = 0;
    if ( argc > 2 )
        nFeat = atoi( argv[2] );
    if ( nFeat == 0 && argc > 3 )
        nFeat = atoi( argv[3] );
    if ( nFeat == 0 )
        nFeat = 34;
    if ( nObj < 1 || nFeat < 1 )
    {
        cerr << "Error: Invalid argument(s).\n";
        exit( 1 );
    }

    stringstream ss;
    ss << nObj << "_" << nFeat << ".dat";
    srand( time( NULL ) );
    FILE *fout = fopen( ss.str().c_str(), "wb");
    fwrite(&nObj, 1, sizeof(int), fout);
    fwrite(&nFeat, 1, sizeof(int), fout);
    float valuef;
    for ( int i = 0; i < nObj; i++ )
    {
        for ( int j = 0; j < nFeat; j++ ) {
            valuef = ( (float)rand() / (float)RAND_MAX );
            fwrite(&valuef, 1, sizeof(float), fout);
        }
    }

    cout << "Generated " << nObj << " objects with " << nFeat << " floats" << " features in " << ss.str() << ".\n";
}
