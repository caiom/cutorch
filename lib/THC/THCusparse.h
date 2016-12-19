#ifndef THC_CUSPARSE_INC
#define THC_CUSPARSE_INC

#include "THCGeneral.h"

/* Level 3 */
THC_API void THCusparse_Dcsrmm2(THCState *state, char transa, char transb, int m, int n, int k, int nnz, double alpha, double *csrValA, int * csrRowPtrA, int * csrColIndA, double * B, int ldb, double beta, double *C, int ldc);
THC_API void THCusparse_Scsrmm2(THCState *state, char transa, char transb, int m, int n, int k, int nnz, float alpha, float *csrValA, int * csrRowPtrA, int * csrColIndA, float * B, int ldb, float beta, float *C, int ldc);

THC_API void THCusparse_Dcsrmv(THCState *state, char transa, int m, int n, int nnz, double alpha, double *csrValA, int * csrRowPtrA, int * csrColIndA, double * B, double beta, double *C);
THC_API void THCusparse_Scsrmv(THCState *state, char transa, int m, int n, int nnz, float alpha, float *csrValA, int * csrRowPtrA, int * csrColIndA, float * B, float beta, float *C);

#endif
