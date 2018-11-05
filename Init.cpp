

template <class T> void Allocate1GPU(int nx, int ny, T *&zb_g)
{
	CUDA_CHECK(cudaMalloc((void **)&zb_g, nx*ny * sizeof(T)));
}
template <class T> void Allocate4GPU(int nx, int ny, T *&zs_g, T *&hh_g, T *&uu_g, T *&vv_g)
{
	CUDA_CHECK(cudaMalloc((void **)&zs_g, nx*ny * sizeof(T)));
	CUDA_CHECK(cudaMalloc((void **)&hh_g, nx*ny * sizeof(T)));
	CUDA_CHECK(cudaMalloc((void **)&uu_g, nx*ny * sizeof(T)));
	CUDA_CHECK(cudaMalloc((void **)&vv_g, nx*ny * sizeof(T)));
}
template <class T> void Allocate1CPU(int nx, int ny, T *&zb)
{
	zb = (T *)malloc(nx*ny * sizeof(T));
	if (!zb)
	{
		fprintf(stderr, "Memory allocation failure\n");

		exit(EXIT_FAILURE);
	}
}

template <class T> void Allocate4CPU(int nx, int ny, T *&zs, T *&hh, T *&uu, T *&vv)
{

	zs = (T *)malloc(nx*ny * sizeof(T));
	hh = (T *)malloc(nx*ny * sizeof(T));
	uu = (T *)malloc(nx*ny * sizeof(T));
	vv = (T *)malloc(nx*ny * sizeof(T));

	if (!zs || !hh || !uu || !vv)
	{
		fprintf(stderr, "Memory allocation failure\n");

		exit(EXIT_FAILURE);
	}
}