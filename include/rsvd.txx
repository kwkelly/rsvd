#include "El.hpp"
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


template<typename F, typename T>
void rsvd2(DistMatrix<T,El::VR,El::STAR> &U, DistMatrix<T,El::VR,El::STAR> &s, DistMatrix<T,El::VR,El::STAR> &Vt, F A, F At, int m, int n, int r)
{

	const Grid& g = U.Grid();
	mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	auto alpha = make_one<T>();
	auto beta =  make_zero<T>();

	DistMatrix<T,El::VR,El::STAR> rw(g);
	DistMatrix<T,El::VR,El::STAR> Q(g); // the projection of omega through A*
	Zeros(Q,m,r);


	for(int i=0;i<r;i++){
		Gaussian(rw, n, 1);
		DistMatrix<T,El::VR,El::STAR> Y_i = View(Q, 0, i, m, 1);
		A(rw,Y_i);
	}

	DistMatrix<T,El::VR,El::STAR> R(g); // the projection of omaga through G*
	qr::Explicit(Q, R, QRCtrl<double>());

	DistMatrix<T,El::VR,El::STAR> GtQ(g);
	Zeros(GtQ,n,r);
	for(int i=0;i<r;i++){
		//if(!rank) std::cout << "ChebFMM " << i << std::endl; 
		DistMatrix<T,El::VR,El::STAR> Q_i = View(Q, 0, i, m, 1);
		DistMatrix<T,El::VR,El::STAR> GtQ_i = View(GtQ, 0, i, n, 1);
		At(Q_i,GtQ_i);
		//Display(Q_i,"Q_i");
		//Display(GQ_i,"GQ_i");
	}

	DistMatrix<double> s_real(g);
	Zeros(s_real,r,1);
	Zeros(s,r,1);

	DistMatrix<T,El::VR,El::STAR> V(g);
	Zeros(V,n,r);

	DistMatrix<T,El::VR,El::STAR> GQ(g);
	Adjoint(GtQ,GQ);
	SVD(GQ, s_real, V, SVDCtrl<double>());


	int h = s.LocalHeight();
	int w = s.LocalWidth();
	double * s_real_buffer = s_real.Buffer();
	El::Complex<double> * s_buffer = s.Buffer();

	#pragma omp parallel for
	for(int i=0;i<w*h;i++){
		s_buffer[i] = El::Complex<double>(s_real_buffer[i]);
	}

	Adjoint(V,Vt);

	//std::vector<double> d(r);
	//DistMatrix<double,STAR,STAR> s_star = s;
	//d.assign(s_star.Buffer(),s_star.Buffer()+r);

	//Zeros(S,r,r);
	//Diagonal(S, d);

	Zeros(U,m,r);
	Gemm(NORMAL,NORMAL,alpha,Q,GQ,beta,U);

	return;

}


template<typename F, typename T> 
void rsvd(DistMatrix<T,El::VR,El::STAR> &U, DistMatrix<T,El::VR,El::STAR> &s, DistMatrix<T,El::VR,El::STAR> &Vt, F A, F At, int m, int n, int r)
{

	const Grid& g = U.Grid();
	mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	auto alpha = make_one<T>();
	auto beta = make_zero<T>();
	//Complex<double> alpha;
	//SetRealPart(alpha,1.0);
	//SetImagPart(alpha,0.0);

	//Complex<double> beta;
	//SetRealPart(beta,0.0);
	//SetImagPart(beta,0.0);

	DistMatrix<T,El::VR,El::STAR> rw(g);
	DistMatrix<T,El::VR,El::STAR> Q(g); // the projection of omega through A*
	Zeros(Q,n,r);


	for(int i=0;i<r;i++){
		Gaussian(rw, m, 1);
		DistMatrix<T,El::VR,El::STAR> Q_i = View(Q, 0, i, n, 1);
		At(rw,Q_i);
	}

	//Display(Q,"Q");

	DistMatrix<T,El::VR,El::STAR> R(g); // the projection of omaga through G*
	qr::Explicit(Q, R, QRCtrl<double>());


	// Now Y is such that G* \approx QQ*G*.Thus we can compute GQ
	// Compute it's SVD and then multiply by Q*, thus giving the approx 
	// SVD of G!!!
	Zeros(U,m,r);
	for(int i=0;i<r;i++){
		DistMatrix<T,El::VR,El::STAR> Q_i = View(Q, 0, i, n, 1);
		DistMatrix<T,El::VR,El::STAR> U_i = View(U, 0, i, m, 1);
		A(Q_i,U_i);
	}


	DistMatrix<double> s_real(g);
	Zeros(s_real,r,1);
	Zeros(s,r,1);

	DistMatrix<T,El::VR,El::STAR> V(g);
	Zeros(V,r,r);

	SVD(U, s_real, V, SVDCtrl<double>());
	int h = s.LocalHeight();
	int w = s.LocalWidth();
	double * s_real_buffer = s_real.Buffer();
	El::Complex<double> * s_buffer = s.Buffer();

	#pragma omp parallel for
	for(int i=0;i<w*h;i++){
		s_buffer[i] = El::Complex<double>(s_real_buffer[i]);
	}


	//std::vector<double> d(r);
	//DistMatrix<double,STAR,STAR> s_star = s;
	//d.assign(s_star.Buffer(),s_star.Buffer()+r);

	//Zeros(S,r,r);
	//Diagonal(S, d);

	// G \approx GQQ* = U\Sigma\V*Q*
	// So We take V* and multiply by Q*
	Zeros(Vt,r,n);
	Gemm(ADJOINT,ADJOINT,alpha,V,Q,beta,Vt);

	return;
}
