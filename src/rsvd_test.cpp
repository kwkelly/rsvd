#include "rsvd.hpp"
#include <functional>
#include <iostream>
#include <cmath>
#include <functional>

using namespace El;
using namespace std::placeholders;


//template<class double>
int rsvd_test_func(DistMatrix<double,VR,STAR> &x, DistMatrix<double,VR,STAR> &y, DistMatrix<double,VR,STAR> *A){

	auto alpha = make_one<double>();
	auto beta  = make_zero<double>();

	int N = A->Height();
	Zeros(y,N,1);

	Gemm(NORMAL,NORMAL,alpha,*A,x,beta,y);

}

//template<class double>
int rsvd_test_t_func(DistMatrix<double,VR,STAR> &x, DistMatrix<double,VR,STAR> &y, DistMatrix<double,VR,STAR> *At){

	auto alpha = make_one<double>();
	auto beta  = make_zero<double>();

	int N = At->Width();
	Zeros(y,N,1);

	Gemm(ADJOINT,NORMAL,alpha,*At,x,beta,y);

	return 0;
}


void test(const int m, const int n, const int r, const int l, const double d){

	double alpha =1.0;
	double beta =0.0;
	const int min_mn = std::min(m,n);

	mpi::Comm comm = mpi::COMM_WORLD;
	Grid g;
	DistMatrix<double,VR,STAR> A(g);
	DistMatrix<double,VR,STAR> D(g);
	DistMatrix<double,VR,STAR> L(g);
	DistMatrix<double,VR,STAR> Rt(g);
	DistMatrix<double,VR,STAR> T(g);


	if(!mpi::Rank(comm)){
		std::cout << "m: " << m << std::endl;
		std::cout << "n: " << n << std::endl;
		std::cout << "r: " << r << std::endl;
		std::cout << "d: " << d << std::endl;
	};

	D.Resize(min_mn,1);
	auto expfill = [&d]( Int i, Int j )->double{ return Pow(d,double(i)); };
	IndexDependentFill( D, function<double(Int,Int)>(expfill) );

	Gaussian(L,m,min_mn);
	qr::ExplicitUnitary(L);

	Gaussian(Rt,n,min_mn);
	qr::ExplicitUnitary(Rt);

	//Zeros(T,m,min_mn);

	//Gemm(NORMAL,NORMAL,alpha,L,D,beta,T);
	DiagonalScale(RIGHT,NORMAL,D,L);
	Zeros(A,m,n);
	Gemm(NORMAL,TRANSPOSE,alpha,L,Rt,beta,A);

	auto A_sf  = std::bind(rsvd_test_func,_1,_2,&A);
	auto At_sf = std::bind(rsvd_test_t_func,_1,_2,&A);

	DistMatrix<double,VR,STAR> U(g);
	DistMatrix<double,VR,STAR> S(g);
	DistMatrix<double,VR,STAR> V(g);

	{
		if(!mpi::Rank(comm)) std::cout << "rsvd2" << std::endl;
		rsvd2(U,S,V,A_sf,At_sf,m,n,r,l);


		// check to see if the right and left singular matrices are orthogonal
		
		DistMatrix<double,VR,STAR> I(g);
		DistMatrix<double,VR,STAR> UtU(g);
		Zeros(UtU,r,r);
		Gemm(ADJOINT,NORMAL,alpha,U,U,beta,UtU);
		Identity(I,r,r);
		Axpy(-1.0,I,UtU);
		double ortho_diff = FrobeniusNorm(UtU)/FrobeniusNorm(I);
		if(!mpi::Rank(comm)) std::cout << "Ortho diff: " << ortho_diff << std::endl;

		DistMatrix<double,VR,STAR> VtV(g);
		Zeros(VtV,r,r);
		Gemm(ADJOINT,NORMAL,alpha,V,V,beta,VtV);
		Axpy(-1.0,I,VtV);
		ortho_diff = FrobeniusNorm(VtV)/FrobeniusNorm(I);
		if(!mpi::Rank(comm)) std::cout << "Ortho diff: " << ortho_diff << std::endl;



		// test that the factorization is good	
		DistMatrix<double,VR,STAR> USVt(g);

		DiagonalScale(RIGHT,NORMAL,S,U);

		Zeros(USVt,m,n);
		Gemm(NORMAL,ADJOINT,alpha,U,V,beta,USVt);

		Axpy(-1.0,A,USVt);

		double ndiff = FrobeniusNorm(USVt)/FrobeniusNorm(A);
		if(!mpi::Rank(comm)) std::cout << "Norm diff: " << ndiff << std::endl;



	}

	{
		if(!mpi::Rank(comm)) std::cout << "rsvd" << std::endl;
		rsvd(U,S,V,A_sf,At_sf,m,n,r,l);
		

		// check to see if the right and left singular matrices are orthogonal
		DistMatrix<double,VR,STAR> I(g);
		DistMatrix<double,VR,STAR> UtU(g);
		Zeros(UtU,r,r);
		Gemm(ADJOINT,NORMAL,alpha,U,U,beta,UtU);
		Identity(I,r,r);
		Axpy(-1.0,I,UtU);
		double ortho_diff = FrobeniusNorm(UtU)/FrobeniusNorm(I);
		if(!mpi::Rank(comm)) std::cout << "Ortho diff: " << ortho_diff << std::endl;

		DistMatrix<double,VR,STAR> VtV(g);
		Zeros(VtV,r,r);
		Gemm(ADJOINT,NORMAL,alpha,V,V,beta,VtV);
		Axpy(-1.0,I,VtV);
		ortho_diff = FrobeniusNorm(VtV)/FrobeniusNorm(I);
		if(!mpi::Rank(comm)) std::cout << "Ortho diff: " << ortho_diff << std::endl;


		// test that the factorization is good	
		DistMatrix<double,VR,STAR> US(g);
		DistMatrix<double,VR,STAR>	USVt(g);

		//Zeros(US,m,r);
		//Gemm(NORMAL,NORMAL,alpha,U,S,beta,US);
		DiagonalScale(RIGHT,NORMAL,S,U);

		Zeros(USVt,m,n);
		Gemm(NORMAL,ADJOINT,alpha,U,V,beta,USVt);

		Axpy(-1.0,A,USVt);

		double ndiff = FrobeniusNorm(USVt)/FrobeniusNorm(A);
		if(!mpi::Rank(comm)) std::cout << "Norm diff: " << ndiff << std::endl;
	}

	return;

}

int main(int argc, char* argv[]){
	Initialize(argc,argv);

	const int m = Input("--m","number of rows",100);
	const int n = Input("--n","number of cols",200);
	const int r = Input("--r","reduced rank",55);
	const int l = Input("--l","oversampling parameter",20);
	const double d = Input("--d","decay",0.75);
	const double tol = Input("--tol","rsvd tolerance",0.05);

	ProcessInput();
	PrintInputReport();

	test(m,n,r,l,d);
	Finalize();

}
