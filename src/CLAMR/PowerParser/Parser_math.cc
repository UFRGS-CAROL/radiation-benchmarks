/*
 *  Copyright (c) 2010-2014, Los Alamos National Security, LLC.
 *  All rights Reserved.
 *
 *  Copyright 2010-2014. Los Alamos National Security, LLC. This software was produced 
 *  under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National 
 *  Laboratory (LANL), which is operated by Los Alamos National Security, LLC 
 *  for the U.S. Department of Energy. The U.S. Government has rights to use, 
 *  reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS 
 *  ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR 
 *  ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified
 *  to produce derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Los Alamos National Security, LLC, Los Alamos 
 *       National Laboratory, LANL, the U.S. Government, nor the names of its 
 *       contributors may be used to endorse or promote products derived from 
 *       this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE LOS ALAMOS NATIONAL SECURITY, LLC AND 
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT 
 *  NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL
 *  SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *  
 *  CLAMR -- LA-CC-11-094
 *  This research code is being developed as part of the 
 *  2011 X Division Summer Workshop for the express purpose
 *  of a collaborative code for development of ideas in
 *  the implementation of AMR codes for Exascale platforms
 *  
 *  AMR implementation of the Wave code previously developed
 *  as a demonstration code for regular grids on Exascale platforms
 *  as part of the Supercomputing Challenge and Los Alamos 
 *  National Laboratory
 *  
 *  Authors: Chuck Wingate   XCP-2   caw@lanl.gov
 *           Robert Robey    XCP-2   brobey@lanl.gov
 */

// ***************************************************************************
// ***************************************************************************
// This class collects various parser math functions.
//
// There are two reasons to have this class:
//     1. To keep the command processing class from getting too big.
//     2. Some of these functions are used in more than one class.
// ***************************************************************************
// ***************************************************************************

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <deque>
#include <math.h>

#include "Word.hh"
#include "Parser_math.hh"

namespace PP
{
using std:: string;
using std::cout;
using std::endl;
using std::stringstream;
using std::setprecision;
using std::vector;
using std::deque;


// ===========================================================================
// Default constructor.
// ===========================================================================
Parser_math::Parser_math()
{
}


// ===========================================================================
// Do a single arithmetic binary operation involving 3 words. i1, on the left,
// is supposed to be a number, i2 is the operator, and i3, on the right, is
// supposed to be a number.
// The result is put in word wres.
// ===========================================================================
void Parser_math::do_op(int i1, int i2, int i3,deque <Word> &wq, Word &wres,
                        stringstream &serr, int &ierr)
{
    // The words to the left and right of the operator have to be a number.
    if ((!wq[i1].is_number()) || (!wq[i3].is_number())) {
        wq[i2].fatal_error(serr, ierr);
        serr << "Expected some number " << wq[i2].get_string() <<
            " some number" << endl;
        serr << "But did not find a number, instead found" << endl;
        serr << wq[i1].get_string() << wq[i2].get_string() <<
            wq[i3].get_string() << endl;
        ierr = 2;
        wres.set_value(0.);
        return;
    }


    double d1 = wq[i1].get_double();
    string op = wq[i2].get_string();
    double d2 = wq[i3].get_double();

    double result = 0.;

    if (op == "+") result = d1 + d2;
    if (op == "-") result = d1 - d2;
    if (op == "*") result = d1 * d2;

    if (op == "**") {
        if (d1 == 0. && d2 >= 0.) {
            wres.set_value(0.);
            return;
        }

        if (d1 == 0. && d2 < 0.) {
            wq[i2].fatal_error(serr, ierr);
            serr << "Trying to exponentiate 0 to a negative power." << endl;
            serr << "Base = " << d1 << "  Exponent = " << d2 << endl;
            ierr = 2;
            wres.set_value(0.);
            return;
        }

        if (d1 < 0. && !wq[i3].is_integer()) {
            wq[i2].fatal_error(serr, ierr);
            serr << "Trying to exponentiate a negative number to a non-integer power." << endl;
            serr << "Base = " << d1 << "  Exponent = " << d2 << endl;
            ierr = 2;
            wres.set_value(0.);
            return;
        }

        result = pow(d1,d2);
    }

    if (op == "/") {
        if (d2 == 0.) {
            if (d1 == 0.) result = 0.;
            else          result = 1.e30;
            wq[i2].fatal_error(serr, ierr);
            serr << "Divide by 0." << endl;
            serr << "Numerator = " << d1 << "  Denominator = " << d2 << endl;
            ierr = 2;
            wres.set_value(result);
            return;
        }
        result = d1 / d2;
    }

    // Do not implement the % operator, it is too much like the fortran %
    // operator which is for referencing components of a fortran structure.
    /*
    if (op == "%") {
        if (d2 == 0.) {
            result = 0.;
            wq[i2].fatal_error(serr, ierr);
            serr << "Modulus (%) second argument is 0." << endl;
            serr << "First arg = " << d1 << "  second arg = " << d2 << endl;
            ierr = 2;
            wres.set_value(result);
            return;
        }
        result = ((int)d1) % ((int)d2);
    }
    */

    wres.set_value(result);
}


// ===========================================================================
// Do a single relational binary operation involving 3 words.
// Relational operators include .eq., .ne., .le., ...
// The result is either true or false and is put in word wres.
// ===========================================================================
void Parser_math::do_op_relational(int i1, int i2, int i3, deque <Word> &wq,
                                   Word &wres, stringstream &serr, int &ierr)
{
    string s1 = wq[i1].get_string();
    string op = wq[i2].get_string();
    string s3 = wq[i3].get_string();
    bool result = false;

    //cout << "&&&&&cw op = " << s1 << op << s3 << endl;

    if ((wq[i1].is_bool()) && (wq[i3].is_bool())) {
        if (op == ".gt." || op == ".ge." || op == ".lt." || op == ".le.") {
            wq[i2].fatal_error(serr, ierr);
            serr << "Does not make sense to compare logical values" << endl;
            serr << " with greater than or less than" << endl;
            serr << "    " << s1 << " " << op << " " << s3 << endl;
            ierr = 2;
            wres.set_value(false);
            return;
        }
    }

    if ( ((wq[i1].is_bool()) && (!wq[i3].is_bool())) ||
         ((!wq[i1].is_bool()) && (wq[i3].is_bool()))
         ) {
        wq[i2].fatal_error(serr, ierr);
        serr << "Does not make sense to compare logical and" << endl;
        serr << " non-logical values" << endl;
        serr << "    " << s1 << " " << op << " " << s3 << endl;
        ierr = 2;
        wres.set_value(false);
        return;
    }

    if ( ((wq[i1].is_number()) && (!wq[i3].is_number())) ||
         ((!wq[i1].is_number()) && (wq[i3].is_number()))
         ) {
        wq[i2].fatal_error(serr, ierr);
        serr << "Does not make sense to compare numerical and" << endl;
        serr << " non-numerical values" << endl;
        serr << "    " << s1 << " " << op << " " << s3 << endl;
        ierr = 2;
        wres.set_value(false);
        return;
    }

    // Compare two numbers.
    if ( (wq[i1].is_number()) && (wq[i3].is_number()) ) {
        double d1 = wq[i1].get_double();
        double d3 = wq[i3].get_double();
        if (op == ".gt.") result = d1 >  d3;
        if (op == ".ge.") result = d1 >= d3;
        if (op == ".lt.") result = d1 <  d3;
        if (op == ".le.") result = d1 <= d3;
        if (op == ".eq.") result = d1 == d3;
        if (op == ".ne.") result = d1 != d3;
        //cout << "&&&&&cw relational result = " << result << endl;
        wres.set_value(result);
        return;
    }

    if ( (wq[i1].is_bool()) && (wq[i3].is_bool()) ) {
        bool b1 = wq[i1].get_bool(serr, ierr);
        bool b3 = wq[i3].get_bool(serr, ierr);
        if (op == ".eq.") result = b1 == b3;
        if (op == ".ne.") result = b1 != b3;
        //cout << "&&&&&cw relational result = " << result << endl;
        wres.set_value(result);
        return;
    }

    // Compare two strings.
    if (op == ".gt.") result = s1 >  s3;
    if (op == ".ge.") result = s1 >= s3;
    if (op == ".lt.") result = s1 <  s3;
    if (op == ".le.") result = s1 <= s3;
    if (op == ".eq.") result = s1 == s3;
    if (op == ".ne.") result = s1 != s3;
    wres.set_value(result);
    return;
}

// ===========================================================================
// Do the .not. operation, this is different from all the others in that
// .not. is a unary operator, the others are binary ops.
// The result is either true or false and is put in word wres.
// ===========================================================================
void Parser_math::do_op_not(int i2, int i3, deque <Word> &wq,
                            Word &wres, stringstream &serr, int &ierr)
{
    string op = wq[i2].get_string();
    string s3 = wq[i3].get_string();
    bool result = false;

    if (!wq[i3].is_bool()) {
        wq[i2].fatal_error(serr, ierr);
        serr << "The word following the .not. operator must be"
             " true or false." << endl;
        serr << "Instead the word following .not. is " << s3 << endl;
        ierr = 2;
        wres.set_value(false);
        return;
    }

    result = true;
    if (wq[i3].get_bool(serr, ierr) == true) result = false;
    wres.set_value(result);
    return;
}


// ===========================================================================
// Do a single logical binary operation involving 3 words.
// The binary logical operators are .and. and .or.
// The result is either true or false and is put in word wres.
// ===========================================================================
void Parser_math::do_op_logical(int i1, int i2, int i3, deque <Word> &wq,
                                Word &wres, stringstream &serr, int &ierr)
{
    string s1 = wq[i1].get_string();
    string op = wq[i2].get_string();
    string s3 = wq[i3].get_string();
    bool result = false;

    //cout << "&&&&&cw logical = " << s1 << op << s3 << endl;

    // For .and. and .or., both operands must be boolean.
    if ((!wq[i1].is_bool()) || (!wq[i3].is_bool())) {
        wq[i2].fatal_error(serr, ierr);
        serr << "The operator is " << op << endl;
        serr << "The two operands (on the left and right of the operator)" << endl;
        serr << "must be logical values (true or false)." << endl;
        serr << "    " << s1 << " " << op << " " << s3 << endl;
        ierr = 2;
        wres.set_value(false);
        return;
    }

    bool b1 = wq[i1].get_bool(serr, ierr);
    bool b3 = wq[i3].get_bool(serr, ierr);
    if (op == ".and.") result = b1 && b3;
    if (op == ".or.")  result = b1 || b3;
    //cout << "&&&&&cw logical result = " << result << endl;
    wres.set_value(result);
    return;
}



} // End of the PP namespace
