/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*                                                                  */
/* Global Data Header File                                          */
/*                                                                  */
/* Generated by KPP-2.2.3 symbolic chemistry Kinetics PreProcessor  */
/*       (http://www.cs.vt.edu/~asandu/Software/KPP)                */
/* KPP is distributed under GPL, the general public licence         */
/*       (http://www.gnu.org/copyleft/gpl.html)                     */
/* (C) 1995-1997, V. Damian & A. Sandu, CGRER, Univ. Iowa           */
/* (C) 1997-2005, A. Sandu, Michigan Tech, Virginia Tech            */
/*     With important contributions from:                           */
/*        M. Damian, Villanova University, USA                      */
/*        R. Sander, Max-Planck Institute for Chemistry, Mainz, Germany */
/*                                                                  */
/* File                 : KPP_Global.h                              */
/* Equation file        : KPP.kpp                                   */
/* Output root filename : KPP                                       */
/*                                                                  */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#ifndef KPP_GLOBAL_H_INCLUDED
#define KPP_GLOBAL_H_INCLUDED

/* Declaration of global variables                                  */

extern double C[NSPEC];                         /* Concentration of all species */
extern double * VAR;
extern double * FIX;
extern double RCONST[NREACT];                   /* Rate constants (global) */
extern double TIME;                             /* Current integration time */
extern double RTOLS;                            /* (scalar) Relative tolerance */
extern double TSTART;                           /* Integration start time */
extern double TEND;                             /* Final integration time */
extern double ATOL[NVAR];                       /* Absolute tolerance */
extern double RTOL[NVAR];                       /* Relative tolerance */
extern double STEPMIN;                          /* Lower bound for integration step */
extern double STEPMAX;                          /* Upper bound for integration step */
extern int LOOKAT[NLOOKAT];                     /* Indexes of species to look at */
extern const char * SPC_NAMES[NSPEC];           /* Names of chemical species */
extern char * SMASS[NMASS];                     /* Names of atoms for mass balance */
extern const char * EQN_NAMES[NREACT];          /* Equation names */
extern char * EQN_TAGS[NREACT];                 /* Equation tags */

/* INLINED global variable declarations                             */

extern double PHOTOL[NPHOTOL];                  /* Photolysis rates */
extern double HET[NSPEC][3];                    /* Heterogeneous reaction rates */
extern double SZA_CST[3];                       /* Constants to compute cosSZA */

/* INLINED global variable declarations                             */

#endif /* KPP_GLOBAL_H_INCLUDED */
