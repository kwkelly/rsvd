#include "rsvd.hpp"
#include <functional>
#include <iostream>
#include <cmath>
#include <functional>
#include <mpi.h>

using namespace El;
using namespace std::placeholders;


template<typename T>
int ApplySVD(const DistMatrix<T,VR,STAR> &U, const DistMatrix<T,VR,STAR> &S, const DistMatrix<T,VR,STAR> &V,const DistMatrix<T,VR,STAR> &x,DistMatrix<T,VR,STAR> &y){
	// Apply the SVD
	// y = USV*x

	T alpha = T(1.0);
	T beta  = T(0.0);

	const Grid& g = U.Grid();
	int N =	U.Height();
	Zeros(y,N,1);

	DistMatrix<T,VR,STAR> temp(g);
	Zeros(temp,V.Width(),1);

	Gemv(ADJOINT,alpha,V,x,temp);
	DiagonalScale(LEFT,NORMAL,S,temp);
	Gemv(NORMAL,alpha,U,temp,y);

	return 0;
}


template<typename T>
int ApplySVDt(const DistMatrix<T,VR,STAR> &U, const DistMatrix<T,VR,STAR> &S, const DistMatrix<T,VR,STAR> &V,const DistMatrix<T,VR,STAR> &x,DistMatrix<T,VR,STAR> &y){
	// Apply the SVD
	// y = USV*x

	T alpha = T(1.0);
	T beta  = T(0.0);

	const Grid& g = U.Grid();
	int N = V.Height(),
	Zeros(y,N,1);

	DistMatrix<T,VR,STAR> temp(g);
	Zeros(temp,U.Width(),1);

	Gemv(ADJOINT,alpha,U,x,temp);
	DiagonalScale(LEFT,ADJOINT,S,temp);
	Gemv(NORMAL,alpha,V,temp,y);

	return 0;
}



//template<class double>
int rsvd_test_func2(const DistMatrix<double,VR,STAR> &x, DistMatrix<double,VR,STAR> &y, DistMatrix<double,VR,STAR> *L,DistMatrix<double,VR,STAR> *D, DistMatrix<double,VR,STAR> *R){

	auto alpha = make_one<double>();
	auto beta  = make_zero<double>();

	const Grid& g = x.Grid();

	int N = L->Height();
	Zeros(y,N,1);

	DistMatrix<double,VR,STAR> temp(g);
	Zeros(temp,R->Width(),1);

	Gemv(ADJOINT,alpha,*R,x,temp);
	DiagonalScale(LEFT,NORMAL,*D,temp);
	Gemv(NORMAL,alpha,*L,temp,y);

	return 0;
}

//template<class double>
int rsvd_test_t_func2(const DistMatrix<double,VR,STAR> &x, DistMatrix<double,VR,STAR> &y, DistMatrix<double,VR,STAR> *L,DistMatrix<double,VR,STAR> *D, DistMatrix<double,VR,STAR> *R){

	auto alpha = make_one<double>();
	auto beta  = make_zero<double>();
	const Grid& g = x.Grid();

	int N = R->Height();
	Zeros(y,N,1);

	DistMatrix<double,VR,STAR> temp(g);
	Zeros(temp,L->Width(),1);

	Gemv(ADJOINT,alpha,*L,x,temp);
	DiagonalScale(LEFT,NORMAL,*D,temp);
	Gemv(NORMAL,alpha,*R,temp,y);

	return 0;
}


//template<class double>
int rsvd_test_func(DistMatrix<double,VR,STAR> &x, DistMatrix<double,VR,STAR> &y, DistMatrix<double,VR,STAR> *A){

	const Grid& g = x.Grid();
	mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int rank;
	MPI_Comm_rank(comm,&rank);
	//if(!rank) std::cout << "A" << std::endl;
	auto alpha = make_one<double>();
	auto beta  = make_zero<double>();

	int N = A->Height();
	Zeros(y,N,1);

	Gemv(NORMAL,alpha,*A,x,y);
	return 0;
}

//template<class double>
int rsvd_test_t_func(DistMatrix<double,VR,STAR> &x, DistMatrix<double,VR,STAR> &y, DistMatrix<double,VR,STAR> *A){
	const Grid& g = x.Grid();
	mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int rank;
	MPI_Comm_rank(comm,&rank);
	//if(!rank) std::cout << "At" << std::endl;

	auto alpha = make_one<double>();
	auto beta  = make_zero<double>();

	int N = A->Width();
	Zeros(y,N,1);

	Gemv(ADJOINT,alpha,*A,x,y);

	return 0;
}


void test(const int m, const int n, const int k, int r, const int l, const int q, const double tol, const double d, rsvd::Adaptive adap, rsvd::Orientation orientation){

	double alpha =1.0;
	double beta =0.0;
	const int min_mn = std::min(m,n);
	int exact_rank = (k<=min_mn?k:min_mn);

	mpi::Comm comm = mpi::COMM_WORLD;
	Grid g;
	DistMatrix<double,VR,STAR> A(g);
	DistMatrix<double,VR,STAR> D(g);
	DistMatrix<double,VR,STAR> L(g);
	DistMatrix<double,VR,STAR> R(g);
	DistMatrix<double,VR,STAR> T(g);


	if(!mpi::Rank(comm)){
		std::cout << "m: " << m << std::endl;
		std::cout << "n: " << n << std::endl;
		std::cout << "r: " << r << std::endl;
		std::cout << "l: " << l << std::endl;
		std::cout << "q: " << q << std::endl;
		std::cout << "k: " << exact_rank << std::endl;
		std::cout << "tol: " << tol << std::endl;
		std::cout << "d: " << d << std::endl;
	};
	int r_orig = r;

	D.Resize(exact_rank,1);
	auto expfill = [&d]( Int i, Int j )->double{ return Pow(d,double(i)); };
	IndexDependentFill( D, std::function<double(Int,Int)>(expfill) );

	Gaussian(L,m,exact_rank);
	qr::ExplicitUnitary(L);

	Gaussian(R,n,exact_rank);
	qr::ExplicitUnitary(R);

	DistMatrix<double,VR,STAR> L_copy = L;
	DiagonalScale(RIGHT,NORMAL,D,L_copy);
	Zeros(A,m,n);
	Gemm(NORMAL,ADJOINT,alpha,L_copy,R,beta,A);

	//auto A_sf  = std::bind(rsvd_test_func,_1,_2,&A);
	//auto At_sf = std::bind(rsvd_test_t_func,_1,_2,&A);

	auto A_sf  = std::bind(rsvd_test_func2,_1,_2,&L,&D,&R);
	auto At_sf = std::bind(rsvd_test_t_func2,_1,_2,&L,&D,&R);


	DistMatrix<double,VR,STAR> U(g);
	DistMatrix<double,VR,STAR> S(g);
	DistMatrix<double,VR,STAR> V(g);

	{ // compute the rsvd factorization
		rsvd::RSVDCtrl ctrl;
		ctrl.m=m;
		ctrl.n=n;
		ctrl.r=r;
		ctrl.l=l;
		ctrl.q=q;
		ctrl.tol=tol;

		ctrl.adap=adap;
		ctrl.orientation=orientation;

		for(int i=0;i<10;i++){
			if(!mpi::Rank(comm)) std::cout << "rsvd" << std::endl;
			ctrl.r = r_orig;
			U.Empty();
			S.Empty();
			V.Empty();

			double rsvd_start = mpi::Time();
			rsvd::rsvd(U,S,V,A_sf,At_sf,ctrl);
			double rsvd_time = mpi::Time() - rsvd_start;

			double rsvd_time_max;
			mpi::Reduce(&rsvd_time,&rsvd_time_max,1,mpi::MAX,0,comm);
			if(!mpi::Rank(comm)) std::cout << "rsvd time" << rsvd_time_max << std::endl;
			r = ctrl.r;
			if(!mpi::Rank(comm)) std::cout << "r="<< r << std::endl;
		}


		r = ctrl.r;
		if(!mpi::Rank(comm)) std::cout << "r="<< r << std::endl;
		
		{// check to see if the right and left singular matrices are orthogonal
			DistMatrix<double,VR,STAR> I(g);
			DistMatrix<double,VR,STAR> UtU(g);
			Zeros(UtU,r,r);
			Gemm(ADJOINT,NORMAL,alpha,U,U,beta,UtU);
			Identity(I,r,r);
			Axpy(-1.0,I,UtU);
			double ortho_diff = FrobeniusNorm(UtU)/FrobeniusNorm(I);
			if(!mpi::Rank(comm)) std::cout << "||U*U - I||_F=" << ortho_diff << std::endl;

			DistMatrix<double,VR,STAR> VtV(g);
			Zeros(VtV,r,r);
			Gemm(ADJOINT,NORMAL,alpha,V,V,beta,VtV);
			Axpy(-1.0,I,VtV);
			ortho_diff = FrobeniusNorm(VtV)/FrobeniusNorm(I);
			if(!mpi::Rank(comm)) std::cout << "||V*V - I||_F=" << ortho_diff << std::endl;
		}

		for(int i=0;i<10;i++){ // test on some random inputs
			DistMatrix<double,VR,STAR> x(g);
			Gaussian(x,n,1);
			DistMatrix<double,VR,STAR> y_ex(m,1,g);
			DistMatrix<double,VR,STAR> y_svd(m,1,g);

			//ApplySVD(L,D,R,x,y_ex);
			Zeros(y_ex,m,1);
			Gemm(El::NORMAL, El::NORMAL,alpha, A,x,beta,y_ex);
			ApplySVD(U,S,V,x,y_svd);
			Axpy(-1.0,y_ex,y_svd);
			double ndiff = TwoNorm(y_svd)/TwoNorm(y_ex);
			if(!mpi::Rank(comm)) std::cout << "||y_ex - y_svd||/||y_ex||=" << ndiff << std::endl;
		}

		// report the times

		// compare the singular values
		D.Resize((r<exact_rank)?r:exact_rank,1);
		S.Resize((r<exact_rank)?r:exact_rank,1);
		double sing_r1 = (k > r) ? D.Get(r-1,0) : 0;
		if(!mpi::Rank(comm)) std::cout << "s_(r+1)=" << sing_r1 << std::endl;
		Axpy(-1.0,D,S);
		double sing_diff = TwoNorm(S)/TwoNorm(D);
		if(!mpi::Rank(comm)) std::cout << "||D-S||/||D||=" << sing_diff << std::endl;


		{// compare to full svd
			DistMatrix<double,VR,STAR> S_ex(g);
			DistMatrix<double,VR,STAR> V_ex(g);

			// compute the exact svd and then report that time
			double svd_start = mpi::Time();
			SVD(A,S_ex,V_ex,SVDCtrl<double>());
			double svd_time = mpi::Time() - svd_start;
			double svd_time_max;
			mpi::Reduce(&svd_time,&svd_time_max,1,mpi::MAX,0,comm);
			if(!mpi::Rank(comm)) std::cout << "svd time" << svd_time_max << std::endl;
		}

	}
	return;

}


int main(int argc, char* argv[]){
	Initialize(argc,argv);

	const int m = Input("--m","number of rows",100);
	const int n = Input("--n","number of cols",200);
	const int r = Input("--r","reduced rank",55);
	const int l = Input("--l","oversampling parameter",20);
	const int q = Input("--q","number of power iterations",0);
	const int k = Input("--k","exact rank",100);
	const std::string adap_s = Input("--adap","ADAP or FIXED","ADAP");
	const std::string orientation_s = Input("--orient","NORMAL or ADJOINT","NORMAL");
	const double d = Input("--d","decay",0.75);
	const double tol = Input("--tol","rsvd tolerance",0.05);


	rsvd::Adaptive adap;
	if(adap_s == "ADAP"){
		adap = rsvd::ADAP;
	}else if (adap_s == "FIXED"){
		adap = rsvd::FIXED;
	} else{
		adap = rsvd::ADAP;
	}

	rsvd::Orientation orientation;
	if(orientation_s == "NORMAL"){
		orientation = rsvd::NORMAL;
	}else if (orientation_s == "ADJOINT"){
		orientation = rsvd::ADJOINT;
	}else{
		orientation = rsvd::NORMAL;
	}





	ProcessInput();
	PrintInputReport();

	test(m,n,k,r,l,q,tol,d,adap,orientation);
	Finalize();

}
