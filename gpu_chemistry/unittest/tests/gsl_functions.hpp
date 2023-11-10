#pragma once

#include "gsl/gsl_linalg.h"


namespace GSL{

inline void LUDecompose(std::vector<gScalar>& matrix, gsl_permutation* p_view)
{
    gLabel size = std::sqrt(matrix.size());
    gsl_matrix_view m_view = gsl_matrix_view_array(matrix.data(), size, size);
    int s;
    gsl_linalg_LU_decomp(&m_view.matrix, p_view, &s);
}

inline void LUDecompose(std::vector<gScalar>& matrix, std::vector<gLabel>& pivot)
{
    gLabel size = pivot.size();
    gsl_permutation *p_view = gsl_permutation_alloc(size);
    LUDecompose(matrix, p_view);

    std::copy(p_view->data, p_view->data + size, pivot.begin());
    gsl_permutation_free(p_view);
}

inline void LUBacksubstitute(std::vector<gScalar>& matrix, gsl_permutation* p_view, std::vector<gScalar>& source, std::vector<gScalar>& x){

    gLabel size = source.size();
    gsl_matrix_view m_view = gsl_matrix_view_array(matrix.data(), size, size);
    gsl_vector_view b = gsl_vector_view_array(source.data(), size);
    gsl_vector_view x_view = gsl_vector_view_array(x.data(), size);


    gsl_linalg_LU_solve(&m_view.matrix, p_view, &b.vector, &x_view.vector);

}

inline void LUBacksubstitute(std::vector<gScalar>& matrix, std::vector<gLabel>& pivot, std::vector<gScalar>& source)
{

    gLabel size = pivot.size();
    gsl_permutation *p_view = gsl_permutation_alloc(size);
    std::copy(pivot.begin(), pivot.end(), p_view->data);

    std::vector<gScalar> x(size);

    LUBacksubstitute(matrix, p_view, source, x);

    source = x;

    gsl_permutation_free(p_view);

}

inline void LUSolve(std::vector<gScalar>& matrix, gsl_permutation* p_view, std::vector<gScalar>& source, std::vector<gScalar>& x){

    LUDecompose(matrix, p_view);
    LUBacksubstitute(matrix, p_view, source, x);
}

}

inline auto call_lu_gsl(const std::vector<gScalar>& m_vals, const std::vector<gScalar>& s_vals)
{


    gLabel size = std::sqrt(m_vals.size());

    std::vector<gScalar> matrix(m_vals.begin(), m_vals.end());
    std::vector<gLabel> pivot(size, 0);
    std::vector<gScalar> v(size, 0);
    std::vector<gScalar> source(s_vals.begin(), s_vals.end());


    GSL::LUDecompose(matrix, pivot);
    GSL::LUBacksubstitute(matrix, pivot, source);


    return std::make_tuple(matrix, pivot, source);


}
