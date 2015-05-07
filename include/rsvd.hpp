#include "El.hpp"

#ifndef _RSVD_HPP_
#define _RSVD_HPP_

using namespace El;

template<typename F1, typename F2, typename T>
void rsvd2(DistMatrix<T,El::VR,El::STAR> &U, DistMatrix<T,El::VR,El::STAR> &S, DistMatrix<T,El::VR,El::STAR> &V, F1 A, F2 At,const int m, const int n, int r);

template<typename F1, typename F2, typename T>
void rsvd(DistMatrix<T,El::VR,El::STAR> &U, DistMatrix<T,El::VR,El::STAR> &S, DistMatrix<T,El::VR,El::STAR> &V, F1 A, F2 At, const int m, const int n, int r);

#include "rsvd.txx"

#endif
