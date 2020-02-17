The Gaussian Elimination application solves systems of equations using the
gaussian elimination method.

The application analyzes an n x n matrix and an associated 1 x n vector to solve a 
set of equations with n variables and n unknowns. The matrix and vector describe equations
of the form:

             a0x + b0y + c0z + d0w = e0
             a1x + b1y + c1z + d1w = e1
             a2x + b2y + c2z + d2w = e2
             a3x + b3y + c3z + d3w = e3

where in this case n=4.  The matrix for the above equations would be as follows:

            [a0 b0 c0 d0]
            [a1 b1 c1 d1]
            [a2 b2 c2 d2]
            [a3 b3 c3 d3]
            
and the vector would be:

            [e0]
            [e1]
            [e2]
            [e3]

The application creates a solution vector:

            [x]
            [y]
            [z]
            [w]
            

******Adjustable work group size*****
The kernel 2 has square shape 
The actually dimension = RD_WG_SIZE_1_0 * RD_WG_SIZE_1_1

USAGE:
make clean
make KERNEL_DIM="-DRD_WG_SIZE_0=128 -DRD_WG_SIZE_1=16 "
