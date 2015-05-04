#include "rsvd.hpp"
#include <functional>
#include <iostream>
#include <cmath>

using namespace El;
using namespace std::placeholders;


template<class T>
int rsvd_test_func(DistMatrix<T> &x, DistMatrix<T> &y, DistMatrix<T> *A){

	auto alpha = make_one<T>();
	auto beta  = make_zero<T>();

	int N = A->Height();
	Zeros(y,N,1);

	Gemm(NORMAL,NORMAL,alpha,*A,x,beta,y);

}

template<class T>
int rsvd_test_t_func(DistMatrix<T> &x, DistMatrix<T> &y, DistMatrix<T> *At){

	auto alpha = make_one<T>();
	auto beta  = make_zero<T>();

	int N = At->Width();
	Zeros(y,N,1);

	Gemm(ADJOINT,NORMAL,alpha,*At,x,beta,y);

	return 0;
}


void test(const int m, const int n, const int r){

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
	std::vector<double> d(min_mn);
	for(int i=0;i<min_mn;i++){
		d[i] = std::pow(.99,i);
	}
	Diagonal(D,d);

	Gaussian(L,m,min_mn);
	qr::ExplicitUnitary(L);

	Gaussian(Rt,n,min_mn);
	qr::ExplicitUnitary(Rt);

	Zeros(T,m,min_mn);

	Gemm(NORMAL,NORMAL,alpha,L,D,beta,T);
	Zeros(A,m,n);
	Gemm(NORMAL,TRANSPOSE,alpha,T,Rt,beta,A);

	auto A_sf  = std::bind(rsvd_test_func,_1,_2,&A);
	auto At_sf = std::bind(rsvd_test_t_func,_1,_2,&A);

	DistMatrix<double,VR,STAR> U(g);
	DistMatrix<double,VR,STAR> S(g);
	DistMatrix<double,VR,STAR> Vt(g);

	{
		rsvd2(U,S,Vt,A_sf,At_sf,m,n,r);
		if(!mpi::Rank(comm)) std::cout << "rsvd2" << std::endl;

		// test that the factorization is good	
		DistMatrix<double,VR,STAR> US(g);
		DistMatrix<double,VR,STAR>	USVt(g);

		DistMatrix<double,VR,STAR> VSt(g);
		DistMatrix<double,VR,STAR>	VStUt(g);

		Zeros(US,m,r);
		Gemm(NORMAL,NORMAL,alpha,U,S,beta,US);

		Zeros(USVt,m,n);
		Gemm(NORMAL,NORMAL,alpha,US,Vt,beta,USVt);

		Axpy(-1.0,A,USVt);

		double ndiff = FrobeniusNorm(USVt)/FrobeniusNorm(A);
		if(!mpi::Rank(comm)) std::cout << "Norm diff: " << ndiff << std::endl;
	}

	{
		rsvd(U,S,Vt,A_sf,At_sf,m,n,r);
		if(!mpi::Rank(comm)) std::cout << "rsvd" << std::endl;

		// test that the factorization is good	
		DistMatrix<double,VR,STAR> US(g);
		DistMatrix<double,VR,STAR>	USVt(g);

		DistMatrix<double,VR,STAR> VSt(g);
		DistMatrix<double,VR,STAR>	VStUt(g);

		Zeros(US,m,r);
		Gemm(NORMAL,NORMAL,alpha,U,S,beta,US);

		Zeros(USVt,m,n);
		Gemm(NORMAL,NORMAL,alpha,US,Vt,beta,USVt);

		Axpy(-1.0,A,USVt);

		double ndiff = FrobeniusNorm(USVt)/FrobeniusNorm(A);
		if(!mpi::Rank(comm)) std::cout << "Norm diff: " << ndiff << std::endl;
	}

	return;

}

void test2(){

	Grid g;
	DistMatrix<double,VR,STAR> A(g);
	DistMatrix<double,STAR,STAR> A2(g);
	Gaussian(A,10,5);
	Display(A,"rect");
	A2 = A;
	
	A2.Resize(10*5,1);
	A=A2;
	Display(A,"vec");

	return;
}


int main(int argc, char* argv[]){
	Initialize(argc,argv);

	const int m = Input("--m","number of rows",100);
	const int n = Input("--n","number of cols",200);
	const int r = Input("--r","reduced rank",55);
	const double tol = Input("--tol","rsvd tolerance",0.05);

	ProcessInput();
	PrintInputReport();

	test(m,n,r);
	//test2();
	Finalize();

}
