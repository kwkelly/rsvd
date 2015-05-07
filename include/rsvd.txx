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


template<typename F1, typename F2, typename T>
void rsvd2(DistMatrix<T,El::VR,El::STAR> &U, DistMatrix<T,El::VR,El::STAR> &s, DistMatrix<T,El::VR,El::STAR> &V, F1 A, F2 At, const int m, const int n, const int r, int l)
{
	// m - number of rows in input operator A
	// n - number of cols in input operator A
	// r - desired rank of output
	// l - oversampling parameter
	
	if(l+r > std::min(m,n)){
		l = std::min(m,n)-r;
	}

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
	Zeros(Q,m,r+l);


	for(int i=0;i<r+l;i++){
		Gaussian(rw, n, 1);
		DistMatrix<T,El::VR,El::STAR> Y_i = View(Q, 0, i, m, 1);
		A(rw,Y_i);
	}

	DistMatrix<T,El::VR,El::STAR> R(g); // the projection of omaga through G*
	qr::Explicit(Q, R, QRCtrl<double>());

	DistMatrix<T,El::VR,El::STAR> AtQ(g);
	//Zeros(AtQ,n,r);
	Zeros(V,n,r+l);
	for(int i=0;i<r+l;i++){
		//if(!rank) std::cout << "ChebFMM " << i << std::endl; 
		DistMatrix<T,El::VR,El::STAR> Q_i = View(Q, 0, i, m, 1);
		DistMatrix<T,El::VR,El::STAR> V_i = View(V, 0, i, n, 1); // we call AtQ V because it becomes V in the SVD
		At(Q_i,V_i);
	}

	DistMatrix<Base<T>,El::VR,El::STAR> s_real(g);
	Zeros(s_real,r+l,1);
	Zeros(s,r+l,1);

	DistMatrix<T,El::VR,El::STAR> U_tilde(g);

	//DistMatrix<T,El::VR,El::STAR> AQ(g);
	//Adjoint(AtQ,AQ);
	SVD(V, s_real, U_tilde, SVDCtrl<double>());

	int h = s.LocalHeight();
	int w = s.LocalWidth();
	auto s_real_buffer = s_real.Buffer();
	T* s_buffer = s.Buffer();

	#pragma omp parallel for
	for(int i=0;i<w*h;i++){
		s_buffer[i] = T(s_real_buffer[i]);
	}

	//Adjoint(V,Vt);

	Zeros(U,m,r+l);
	Gemm(NORMAL,NORMAL,alpha,Q,U_tilde,beta,U);

	// Now truncate out the oversampling parameter
	U.Resize(m,r);
	s.Resize(r,1);
	V.Resize(n,r);

	return;

}


template<typename F1, typename F2, typename T> 
void rsvd(DistMatrix<T,El::VR,El::STAR> &U, DistMatrix<T,El::VR,El::STAR> &s, DistMatrix<T,El::VR,El::STAR> &V, F1 A, F2 At, const int m, const int n, const int r, int l)
{
	// m - number of rows in input operator A
	// n - number of cols in input operator A
	// r - desired rank of output
	// l - oversampling parameter
	
	if(l+r > std::min(m,n)){
		l = std::min(m,n)-r;
	}
	


	const Grid& g = U.Grid();
	mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	auto alpha = T(1.0);
	auto beta =  T(0.0);

	DistMatrix<T,El::VR,El::STAR> rw(g);
	DistMatrix<T,El::VR,El::STAR> Q(g); // the projection of omega through A*
	Zeros(Q,n,r+l);


	for(int i=0;i<r+l;i++){
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
	Zeros(U,m,r+l);
	for(int i=0;i<r+l;i++){
		DistMatrix<T,El::VR,El::STAR> Q_i = View(Q, 0, i, n, 1);
		DistMatrix<T,El::VR,El::STAR> U_i = View(U, 0, i, m, 1);
		A(Q_i,U_i);
	}


	DistMatrix<Base<T>,El::VR,El::STAR> s_real(g);
	Zeros(s_real,r+l,1);
	Zeros(s,r+l,1);

	DistMatrix<T,El::VR,El::STAR> V_tilde(g);
	//Zeros(V,r+l,r+l);

	SVD(U, s_real, V_tilde, SVDCtrl<double>());


	int h = s.LocalHeight();
	int w = s.LocalWidth();	
	auto s_real_buffer = s_real.Buffer();
	T* s_buffer = s.Buffer();

	#pragma omp parallel for
	for(int i=0;i<w*h;i++){
		s_buffer[i] = T(s_real_buffer[i]);
	}


	//std::vector<double> d(r);
	//DistMatrix<double,STAR,STAR> s_star = s;
	//d.assign(s_star.Buffer(),s_star.Buffer()+r);

	//Zeros(S,r,r);
	//Diagonal(S, d);

	// G \approx GQQ* = U\Sigma\V*Q*
	// So We take V* and multiply by Q*
	Zeros(V,n,r+l);
	Gemm(NORMAL,NORMAL,alpha,Q,V_tilde,beta,V);

	U.Resize(m,r);
	s.Resize(r,1);
	V.Resize(n,r);

	return;
}
