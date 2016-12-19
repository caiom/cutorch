#include "THCusparse.h"
#include "THCGeneral.h"

struct cusparseMatDescr;
typedef struct cusparseMatDescr *cusparseMatDescr_t;

static cusparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't') return CUSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
    THError("trans must be one of: t, n, c");
    return CUSPARSE_OPERATION_TRANSPOSE;
  }
}


/* Level 3 */
void THCusparse_Scsrmm2(THCState *state, char transa, char transb, int m, int n, int k, int nnz, float alpha, float *csrValA, int * csrRowPtrA, int * csrColIndA, float * B, int ldb, float beta, float *C, int ldc)
{
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX)  && (nnz <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_nnz = (int)nnz;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cusparseHandle_t handle = THCState_getCurrentCusparseHandle(state);
    cusparseSetStream(handle, THCState_getCurrentStream(state));

    cusparseMatDescr_t matDescr;

    THCusparseCheck(cusparseCreateMatDescr(&matDescr));

    cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);

    THCusparseCheck(cusparseScsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, matDescr, csrValA, csrRowPtrA, csrColIndA, B, i_ldb, &beta, C, i_ldc));
    THCusparseCheck(cusparseDestroyMatDescr(matDescr));
    return;
  }
  THError("Cusparse_csrmm2 only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

void THCusparse_Dcsrmm2(THCState *state, char transa, char transb, int m, int n, int k, int nnz, double alpha, double *csrValA, int * csrRowPtrA, int * csrColIndA, double * B, int ldb, double beta, double *C, int ldc)
{
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX)  && (nnz <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_nnz = (int)nnz;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cusparseMatDescr_t matDescr;

    THCusparseCheck(cusparseCreateMatDescr(&matDescr));

    cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseHandle_t handle = THCState_getCurrentCusparseHandle(state);
    cusparseSetStream(handle, THCState_getCurrentStream(state));

    THCusparseCheck(cusparseDcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, matDescr, csrValA, csrRowPtrA, csrColIndA, B, i_ldb, &beta, C, i_ldc));
    THCusparseCheck(cusparseDestroyMatDescr(matDescr));
    return;
  }
  THError("Cusparse_csrmm2 only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

void THCusparse_Scsrmv(THCState *state, char transa, int m, int n, int nnz, float alpha, float *csrValA, int * csrRowPtrA, int * csrColIndA, float * B, float beta, float *C)
{
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);

  if( (m <= INT_MAX) && (n <= INT_MAX)  && (nnz <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_nnz = (int)nnz;

    cusparseHandle_t handle = THCState_getCurrentCusparseHandle(state);
    cusparseSetStream(handle, THCState_getCurrentStream(state));

    cusparseMatDescr_t matDescr;

    THCusparseCheck(cusparseCreateMatDescr(&matDescr));

    cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);

    THCusparseCheck(cusparseScsrmv(THCState_getCurrentCusparseHandle(state), opa, i_m, i_n, i_nnz, &alpha, matDescr, csrValA, csrRowPtrA, csrColIndA, B, &beta, C));
    THCusparseCheck(cusparseDestroyMatDescr(matDescr));
    return;
  }
  THError("Cusparse_csrmv only supports m, n"
          "with the bound [val] <= %d", INT_MAX);
}

void THCusparse_Dcsrmv(THCState *state, char transa, int m, int n, int nnz, double alpha, double *csrValA, int * csrRowPtrA, int * csrColIndA, double * B, double beta, double *C)
{
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);

  if( (m <= INT_MAX) && (n <= INT_MAX)  && (nnz <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_nnz = (int)nnz;

    cusparseHandle_t handle = THCState_getCurrentCusparseHandle(state);
    cusparseSetStream(handle, THCState_getCurrentStream(state));
 
    cusparseMatDescr_t matDescr;
   
    THCusparseCheck(cusparseCreateMatDescr(&matDescr));

    cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);

    THCusparseCheck(cusparseDcsrmv(THCState_getCurrentCusparseHandle(state), opa, i_m, i_n, i_nnz, &alpha, matDescr, csrValA, csrRowPtrA, csrColIndA, B, &beta, C));
    THCusparseCheck(cusparseDestroyMatDescr(matDescr));
    return;
  }
  THError("Cusparse_csrmv only supports m, n"
          "with the bound [val] <= %d", INT_MAX);
}
