#include "El.hpp"
#include <numeric>
#include "rsvd.hpp"

using namespace El;

template<class T>
T make_one()
{
	return 1.0;
}

template<>
Complex<double> make_one()
{
	Complex<double> alpha;
	SetRealPart(alpha,1.0);
	SetImagPart(alpha,0.0);
	return alpha;
}

template<>
double make_one()
{
	return 1.0;
}


template<class T>
T make_zero()
{
	return 0.0;
}

template<>
Complex<double> make_zero()
{
	Complex<double> alpha;
	SetRealPart(alpha,0.0);
	SetImagPart(alpha,0.0);
	return alpha;
}

template<>
double make_zero()
{
	return 0.0;
}

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
void rsvd(DistMatrix<T,El::VR,El::STAR> &U, DistMatrix<T,El::VR,El::STAR> &s, DistMatrix<T,El::VR,El::STAR> &V, F1 A, F2 At, RSVDCtrl &ctrl)
{
	// rsvd ctrl is the ctrl structure governing the RSVD behavior
	// extract the ctrl data
	int m=ctrl.m;
	int n=ctrl.n;
	int l=ctrl.l;
	int r=ctrl.r;
	int q=ctrl.q;
	int max_sz = ctrl.max_sz;
	double tol=ctrl.tol;
	Orientation orientation=ctrl.orientation;
	Adaptive adap=ctrl.adap;


	// some checks
	if(m==0 or n==0) std::logic_error("The matrix sizes in the ctrl struct are set to 0. Please set the sizes appropriately");
	
	// intilize some needed data
	const Grid& g = U.Grid();
	mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	auto alpha = T(1.0);
	auto beta =  T(0.0);
	double err_est = 0;
	int R = r;
	int R_old = 0;
	//int inc_size = std::ceil(0.05*std::min(m,n));
	int inc_size = r;
	int test_sz = 10;

	/////////////////////////////
	// ORIENTATION ADJOINT
	/////////////////////////////
	if(orientation == ADJOINT){
		DistMatrix<T,El::VR,El::STAR> rw(g);
		DistMatrix<T,El::VR,El::STAR> Q(g); // the projection of omega through A
		DistMatrix<T,El::VR,El::STAR> W(g); // the projection of omega through A
		W.Resize(n,max_sz+l);

		do{
			auto Y = View(W, 0, 0, n, R+l);
			for(int i=R_old;i<R+l;i++){ //project A*\Omega
				Gaussian(rw, m, 1);
				auto Y_i = View(Y, 0, i, n, 1);
				At(rw,Y_i);
			}
			for(int p=0;p<q;p++){ // power iterate
				DistMatrix<T,El::VR,El::STAR> temp(m,1,g);
				for(int i=R_old;i<R+l;i++){ // Apply A*
					auto Y_i = View(Y, 0, i, n, 1);
					A(Y_i,temp);
					At(temp,Y_i);
				}
			}

			if(adap == rsvd::FIXED){
				//Copy(Y,Q);
				Q = View(Y,0,0,n,R+l);
				DistMatrix<Base<T>,El::VR,El::STAR> temp1(g);
				DistMatrix<T,El::VR,El::STAR> temp2(g);
				SVD(Q, temp1, temp2);
				//qr::ExplicitUnitary(Q);
				Q.Resize(n,R);
				// Estimate the error if adaptivity is on, else we are done
			}
			else{
				//Copy(Y,Q);
				Q = View(Y,0,0,n,R+l);
				//DistMatrix<T,El::VR,El::STAR> Y = Q;
				DistMatrix<Base<T>,El::VR,El::STAR> temp1(g);
				DistMatrix<T,El::VR,El::STAR> temp2(g);
				SVD(Q, temp1, temp2);
				//qr::ExplicitUnitary(Q);
				Q.Resize(n,R);
				// compute an error estimate
				//Base<T> s_1 = TwoNormEstimate(Y);
				Base<T> s_1 = temp1.Get(0,0);
				// compute  ||Y - QQ*Y||
				DistMatrix<T,El::VR,El::STAR> C(R,R+l,g);
				DistMatrix<T,El::VR,El::STAR> D(n,R+l,g);
				Zeros(C,R,R+l);
				Zeros(D,n,R+l);

				std::cout << "YH: " << Y.Height() << std::endl;
				std::cout << "Yw: " << Y.Width() << std::endl;
				std::cout << "QH: " << Q.Height() << std::endl;
				std::cout << "Qw: " << Q.Width() << std::endl;
				Gemm(El::ADJOINT,El::NORMAL,alpha,Q,Y,beta,C);
				Gemm(El::NORMAL,El::NORMAL,alpha,Q,C,beta,D);
				Axpy(-1.0,D,Y);
				Base<T> s_diff = TwoNormEstimate(Y);
				err_est = std::pow(s_diff/s_1,1.0/(q+1));
				//std::cout << err_est << std::endl;
			}
			R_old = R;
			if(err_est > tol){

				if(R + inc_size < max_sz){
					R += inc_size;
				}
				else{
					R = max_sz;
				}
			}

		std::cout << "err: " << err_est <<std::endl;
		}
		while(adap == rsvd::ADAP and err_est > tol);

		// Now Y is such that G* \approx QQ*G*.Thus we can compute GQ
		// Compute it's SVD and then multiply by Q*, thus giving the approx 
		// SVD of G!!!
		Zeros(U,m,R);
		for(int i=0;i<R;i++){
			DistMatrix<T,El::VR,El::STAR> Q_i = View(Q, 0, i, n, 1);
			DistMatrix<T,El::VR,El::STAR> U_i = View(U, 0, i, m, 1);
			A(Q_i,U_i);
		}


		DistMatrix<Base<T>,El::VR,El::STAR> s_real(g);
		Zeros(s_real,R,1);
		Zeros(s,R,1);

		DistMatrix<T,El::VR,El::STAR> V_tilde(g);
		//Zeros(V,r+l,r+l);

		SVD(U, s_real, V_tilde, SVDCtrl<double>());


		int h = s.LocalHeight();
		int w = s.LocalWidth();	
		auto s_real_buffer = s_real.Buffer();
		T* s_buffer = s.Buffer();

		#pragma omp parallel for // cast the diagonal matrix to type T
		for(int i=0;i<w*h;i++){
			s_buffer[i] = T(s_real_buffer[i]);
		}

		// G \approx GQQ* = U\Sigma\V*Q*
		// So We take V* and multiply by Q*
		Zeros(V,n,R);
		Gemm(El::NORMAL,El::NORMAL,alpha,Q,V_tilde,beta,V);
	}

	/////////////////////////////
	// ORIENTATION NORMAL
	/////////////////////////////
	if(orientation == NORMAL){ 
		DistMatrix<T,El::VR,El::STAR> rw(g);
		DistMatrix<T,El::VR,El::STAR> Q(g); // the projection of omega through A
		DistMatrix<T,El::VR,El::STAR> W(g); // the projection of omega through A
		W.Resize(m,max_sz+l);

		do{
			auto Y = View(W, 0, 0, m, R+l);

			for(int i=R_old;i<R+l;i++){// apply A\Omega
				Gaussian(rw, n, 1);
				auto Y_i = View(Y, 0, i, m, 1);
				A(rw,Y_i);
			}
			for(int p=0;p<q;p++){ // power iterate
				DistMatrix<T,El::VR,El::STAR> temp(n,1,g);
				for(int i=R_old;i<R+l;i++){ // Apply A*
					auto Y_i = View(Y, 0, i, m, 1);
					At(Y_i,temp);
					A(temp,Y_i);
				}
			}

			if(adap == rsvd::FIXED){
				//Copy(Y,Q);
				Q = View(Y,0,0,m,R+l);
				DistMatrix<Base<T>,El::VR,El::STAR> temp1(g);
				DistMatrix<T,El::VR,El::STAR> temp2(g);
				SVD(Q, temp1, temp2);
				//qr::ExplicitUnitary(Q);
				Q.Resize(m,R);
			}
			else{
				//Copy(Y,Q);
				Q = View(Y,0,0,m,R+l);
				DistMatrix<Base<T>,El::VR,El::STAR> temp1(g);
				DistMatrix<T,El::VR,El::STAR> temp2(g);
				SVD(Q, temp1, temp2);
				Q.Resize(m,R);

				// compute an error estimate
				Base<T> s_1 = temp1.Get(0,0);
				// compute  ||Y - QQ*Y||
				DistMatrix<T,El::VR,El::STAR> C(R,R+l,g);
				DistMatrix<T,El::VR,El::STAR> D(m,R+l,g);
				Zeros(C,R,R+l);
				Zeros(D,m,R+l);

				Gemm(El::ADJOINT,El::NORMAL,alpha,Q,Y,beta,C);
				Gemm(El::NORMAL,El::NORMAL,alpha,Q,C,beta,D);
				Axpy(-1.0,Y,D);
				Base<T> s_diff = TwoNormEstimate(D);
				err_est = std::pow(s_diff/s_1,1.0/(q+1));
			}
			R_old = R;
			if(err_est > tol){
				if(R + inc_size < max_sz){
					R += inc_size;
				}
				else{
					R = max_sz;
				}
			}
		if(!rank) std::cout << "err: " << err_est <<std::endl;
		}
		while(adap == rsvd::ADAP and err_est > tol and R_old < max_sz);

		Zeros(V,n,R);
		for(int i=0;i<R;i++){
			DistMatrix<T,El::VR,El::STAR> Q_i = View(Q, 0, i, m, 1);
			DistMatrix<T,El::VR,El::STAR> V_i = View(V, 0, i, n, 1); // we call AtQ V because it becomes V in the SVD
			At(Q_i,V_i);
		}

		DistMatrix<Base<T>,El::VR,El::STAR> s_real(g);
		Zeros(s_real,R,1);
		Zeros(s,R,1);

		DistMatrix<T,El::VR,El::STAR> U_tilde(g);

		SVD(V, s_real, U_tilde, SVDCtrl<double>());

		int h = s.LocalHeight();
		int w = s.LocalWidth();
		auto s_real_buffer = s_real.Buffer();
		T* s_buffer = s.Buffer();

		#pragma omp parallel for
		for(int i=0;i<w*h;i++){
			s_buffer[i] = T(s_real_buffer[i]);
		}

		Zeros(U,m,R);
		Gemm(El::NORMAL,El::NORMAL,alpha,Q,U_tilde,beta,U);
	}

	ctrl.r = R;

	U.Resize(m,R);
	s.Resize(R,1);
	V.Resize(n,R);

	return;
}
} // end namespace
