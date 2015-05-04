#include "El.hpp"

#ifndef _RSVD_HPP_
#define _RSVD_HPP_

using namespace El;

template<typename F, typename T>
void rsvd2(DistMatrix<T,El::VR,El::STAR> &U, DistMatrix<T,El::VR,El::STAR> &S, DistMatrix<T,El::VR,El::STAR> &Vt, F A, F At,int m, int n, int r);

template<typename F, typename T>
void rsvd(DistMatrix<T,El::VR,El::STAR> &U, DistMatrix<T,El::VR,El::STAR> &S, DistMatrix<T,El::VR,El::STAR> &Vt, F A, F At, int m, int n, int r);

#include "rsvd.txx"

#endif
