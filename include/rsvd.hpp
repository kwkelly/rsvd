#include "El.hpp"

#ifndef _RSVD_HPP_
#define _RSVD_HPP_

using namespace El;

namespace rsvd{
template<typename F1, typename F2, typename T>
void rsvd(DistMatrix<T,El::VC,El::STAR> &U, DistMatrix<T,El::VC,El::STAR> &S, DistMatrix<T,El::VC,El::STAR> &V, F1 A, F2 At, const int m, const int n, int r);
}
#include "rsvd.txx"

#endif
