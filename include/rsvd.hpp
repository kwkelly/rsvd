#include "El.hpp"

#ifndef _RSVD_HPP_
#define _RSVD_HPP_

using namespace El;

namespace rsvd{
	enum Adaptive{ // whether to use an adaptive or a fixed rank method
		FIXED,
		ADAP
	};

	enum Orientation{ // use the Adjoint or Normal operator to construct the svd.
		ADJOINT,
		NORMAL
	};

	struct RSVDCtrl
	{
		int m=0; // number of rows
		int n=0; // number of cols
		int r=0; // desired rank for for fixed rank. If adaptive, then the min rank to use.
		int l=0; // oversampling parameter
		int q=0; // power iteration parameter
		int max_sz = 0; // power iteration parameter
		double tol=0.05;
		Adaptive adap=FIXED;
		Orientation orientation=NORMAL;
	};

	template<typename F1, typename F2, typename T> 
	void rsvd(DistMatrix<T,El::VC,El::STAR> &U, DistMatrix<T,El::VC,El::STAR> &s, DistMatrix<T,El::VC,El::STAR> &V, F1 &A, F2 &At, RSVDCtrl &ctrl);
}
#include "rsvd.txx"

#endif
