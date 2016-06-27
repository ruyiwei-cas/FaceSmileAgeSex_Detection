/**
*** Copyright (C) 1985-2011 Intel Corporation.  All rights reserved.
***
*** The information and source code contained herein is the exclusive
*** property of Intel Corporation and may not be disclosed, examined
*** or reproduced in whole or in part without explicit written authorization
*** from the company.
***
*** Embedded Application Lab, Intel Labs China.
**/

#ifndef _CX_OPTIONS_HPP_
#define _CX_OPTIONS_HPP_

#include <string.h>

//
// This class simulates UNIX getopt() function.
//
// see: http://www.koders.com/c/fid034963469B932D9D87F91C86680EB08DB4DE9AA3.aspx
//
class CxOptions
{
public:
    CxOptions( int _argc, char* _argv[], const char* _opts )
    {
        sp = 1;
        argc = _argc;
        argv = _argv;
        opts = _opts;
        optind = 1;
        optopt = '\0';
        optarg = NULL;
    }

    int ind() const { return optind; }

    int opt() const { return optopt; }

    const char* arg() const { return optarg; }

    int get()
    {
        int c;
        const char *cp;

        if( sp == 1 )
        {
            if( optind >= argc
                    || argv[optind][0] != '-' || argv[optind][1] == '\0' )
                return -1;
            else if( strcmp(argv[optind], "--") == 0 ) 
            {
                optind++;
                return -1;
            }
        }
        optopt = c = argv[optind][sp];
        if( c == ':' || (cp=strchr(opts, c)) == NULL ) 
        {
//            ERR(": illegal option -- ", c);
            if( argv[optind][++sp] == '\0' ) 
            {
                optind++;
                sp = 1;
            }
            return '?';
        }
        if( *++cp == ':' ) 
        {
            if( argv[optind][sp+1] != '\0' )
                optarg = &argv[optind++][sp+1];
            else if( ++optind >= argc ) 
            {
//                ERR(": option requires an argument -- ", c);
                sp = 1;
                return '?';
            } 
            else
                optarg = argv[optind++];
            sp = 1;
        } 
        else 
        {
            if( argv[optind][++sp] == '\0' ) 
            {
                sp = 1;
                optind++;
            }
            optarg = NULL;
        }
        return c;
    }

protected:
    int         sp;
    int         argc;
    char**      argv;
    const char* opts;
    int         optind;
    int         optopt;
    const char* optarg;

};

#endif // _CX_OPTIONS_HPP_
