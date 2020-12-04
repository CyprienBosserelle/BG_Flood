#include "Halo.h"

template <class T> void fillHaloD(Param XParam, int ib, BlockP<T> XBlock, T* z)
{
	

	fillLeft(XParam, ib, XBlock, z);
	fillRight(XParam, ib, XBlock, z);
	fillTop(XParam, ib, XBlock, z);
	fillBot(XParam, ib, XBlock, z);
	//fill bot
	//fill top
	

}
template void fillHaloD<double>(Param XParam, int ib, BlockP<double> XBlock, double* z);
template void fillHaloD<float>(Param XParam, int ib, BlockP<float> XBlock, float* z);

template <class T> void fillHaloC(Param XParam, BlockP<T> XBlock, T* z)
{
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		fillHaloD(XParam, ib, XBlock, z);
	}
}
template void fillHaloC<float>(Param XParam, BlockP<float> XBlock, float* z);
template void fillHaloC<double>(Param XParam, BlockP<double> XBlock, double* z);

template <class T> void fillHaloF(Param XParam, bool doProlongation, BlockP<T> XBlock, T* z)
{
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		fillLeftFlux(XParam, doProlongation, ib, XBlock, z);
		fillBotFlux(XParam, doProlongation, ib, XBlock, z);
		fillRightFlux(XParam, doProlongation, ib, XBlock, z);
		fillTopFlux(XParam, doProlongation, ib, XBlock, z);
	
	}
}
template void fillHaloF<float>(Param XParam, bool doProlongation, BlockP<float> XBlock, float* z);
template void fillHaloF<double>(Param XParam, bool doProlongation, BlockP<double> XBlock, double* z);


template <class T> void fillHaloGPU(Param XParam, BlockP<T> XBlock, cudaStream_t stream, T* z)
{

	dim3 blockDimHaloLR(1, 16, 1);
	dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	fillLeft << <gridDim, blockDimHaloLR, 0 , stream >> > (XParam.halowidth, XBlock.active, XBlock.level, XBlock.LeftBot, XBlock.LeftTop, XBlock.RightBot, XBlock.BotRight, XBlock.TopRight, z);
	//fillLeft << <gridDim, blockDimHaloLR, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, XBlock.LeftBot, XBlock.LeftTop, XBlock.RightBot, XBlock.BotRight, XBlock.TopRight, z);
	//CUDA_CHECK(cudaDeviceSynchronize());
	fillRight << <gridDim, blockDimHaloLR, 0, stream >> > (XParam.halowidth, XBlock.active, XBlock.level, XBlock.RightBot, XBlock.RightTop, XBlock.LeftBot, XBlock.BotLeft, XBlock.TopLeft, z);
	//fillRight << <gridDim, blockDimHaloLR, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, XBlock.RightBot, XBlock.RightTop, XBlock.LeftBot, XBlock.BotLeft, XBlock.TopLeft, z);
	//CUDA_CHECK(cudaDeviceSynchronize());
	fillBot << <gridDim, blockDimHaloBT, 0, stream >> > (XParam.halowidth, XBlock.active, XBlock.level, XBlock.BotLeft, XBlock.BotRight, XBlock.TopLeft, XBlock.LeftTop, XBlock.RightTop, z);
	//fillBot << <gridDim, blockDimHaloBT, 0>> > (XParam.halowidth, XBlock.active, XBlock.level, XBlock.BotLeft, XBlock.BotRight, XBlock.TopLeft, XBlock.LeftTop, XBlock.RightTop, z);
	//CUDA_CHECK(cudaDeviceSynchronize());
	fillTop << <gridDim, blockDimHaloBT, 0, stream >> > (XParam.halowidth, XBlock.active, XBlock.level, XBlock.TopLeft, XBlock.TopRight, XBlock.BotLeft, XBlock.LeftBot, XBlock.RightBot, z);
	//fillTop << <gridDim, blockDimHaloBT, 0>> > (XParam.halowidth, XBlock.active, XBlock.level, XBlock.TopLeft, XBlock.TopRight, XBlock.BotLeft, XBlock.LeftBot, XBlock.RightBot, z);
	//CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaStreamSynchronize(stream));

}
template void fillHaloGPU<double>(Param XParam, BlockP<double> XBlock, cudaStream_t stream, double* z);
template void fillHaloGPU<float>(Param XParam, BlockP<float> XBlock, cudaStream_t stream, float* z);


template <class T> void fillHaloTopRightC(Param XParam, BlockP<T> XBlock, T* z)
{
	// for flux term and actually most terms, only top and right neighbours are needed!

	//fillLeft(XParam, ib, XBlock, z);
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		fillRightFlux(XParam,false, ib, XBlock, z);
		fillTopFlux(XParam,false, ib, XBlock, z);
	}
	//fillBot(XParam, ib, XBlock, z);
	//fill bot
	//fill top


}
template void fillHaloTopRightC<double>(Param XParam, BlockP<double> XBlock, double* z);
template void fillHaloTopRightC<float>(Param XParam, BlockP<float> XBlock, float* z);


template <class T> void fillHaloTopRightGPU(Param XParam, BlockP<T> XBlock, cudaStream_t stream, T* z)
{

	dim3 blockDimHaloLR(1, 16, 1);
	dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	//fillLeft << <gridDim, blockDimHaloLR, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, XBlock.LeftBot, XBlock.LeftTop, XBlock.RightBot, XBlock.BotRight, XBlock.TopRight, a);
	fillRightFlux << <gridDim, blockDimHaloLR, 0, stream >> > (XParam.halowidth,false, XBlock.active, XBlock.level, XBlock.RightBot, XBlock.RightTop, XBlock.LeftBot, XBlock.BotLeft, XBlock.TopLeft, z);
	//fillBot << <gridDim, blockDimHaloBT, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, XBlock.BotLeft, XBlock.BotRight, XBlock.TopLeft, XBlock.LeftTop, XBlock.RightTop, a);
	fillTopFlux << <gridDim, blockDimHaloBT, 0, stream >> > (XParam.halowidth,false, XBlock.active, XBlock.level, XBlock.TopLeft, XBlock.TopRight, XBlock.BotLeft, XBlock.LeftBot, XBlock.RightBot, z);

	CUDA_CHECK(cudaStreamSynchronize(stream));

}
template void fillHaloTopRightGPU<double>(Param XParam, BlockP<double> XBlock, cudaStream_t stream, double* z);
template void fillHaloTopRightGPU<float>(Param XParam, BlockP<float> XBlock, cudaStream_t stream, float* z);


template <class T> void fillHalo(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T*zb)
{
	
		std::thread t0(fillHaloC<T>,XParam, XBlock, Xev.h);
		std::thread t1(fillHaloC<T>,XParam, XBlock, Xev.zs);
		std::thread t2(fillHaloF<T>,XParam,true, XBlock, Xev.u);
		std::thread t3(fillHaloF<T>,XParam,true, XBlock, Xev.v);

		t0.join();
		t1.join();
		t2.join();
		t3.join();

		conserveElevation(XParam, XBlock, Xev, zb);


		maskbnd(XParam, XBlock, Xev, zb);
	
}
template void fillHalo<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> Xev, float *zb);
template void fillHalo<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> Xev,double * zb);

template <class T> void fillHalo(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev)
{

	std::thread t0(fillHaloC<T>, XParam, XBlock, Xev.h);
	std::thread t1(fillHaloC<T>, XParam, XBlock, Xev.zs);
	std::thread t2(fillHaloF<T>, XParam, true, XBlock, Xev.u);
	std::thread t3(fillHaloF<T>, XParam, true, XBlock, Xev.v);

	t0.join();
	t1.join();
	t2.join();
	t3.join();

	
	//maskbnd(XParam, XBlock, Xev, zb);

}
template void fillHalo<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> Xev);
template void fillHalo<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> Xev);


template <class T> void fillHaloGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev)
{
	const int num_streams = 4;

	cudaStream_t streams[num_streams];

	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}


	fillHaloGPU(XParam, XBlock, streams[0], Xev.h);
	fillHaloGPU(XParam, XBlock, streams[1], Xev.zs);
	fillHaloGPU(XParam, XBlock, streams[2], Xev.u);
	fillHaloGPU(XParam, XBlock, streams[3], Xev.v);

	for (int i = 0; i < num_streams; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
}
template void fillHaloGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> Xev);
template void fillHaloGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> Xev);

template <class T> void fillHaloGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev,T * zb)
{
	const int num_streams = 4;
	dim3 blockDimHalo(16, 1, 1);
	dim3 gridDim(XBlock.mask.nblk, 1, 1);
	cudaStream_t streams[num_streams];

	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}
	

	fillHaloGPU(XParam, XBlock, streams[0], Xev.h);
	fillHaloGPU(XParam, XBlock, streams[1], Xev.zs);
	fillHaloGPU(XParam, XBlock, streams[2], Xev.u);
	fillHaloGPU(XParam, XBlock, streams[3], Xev.v);
	CUDA_CHECK(cudaDeviceSynchronize());


	conserveElevationGPU(XParam, XBlock, Xev, zb);


	maskbndGPUleft << <gridDim, blockDimHalo, 0 , streams[0] >> > (XParam, XBlock,  Xev, zb);
	maskbndGPUtop << <gridDim, blockDimHalo, 0, streams[1] >> > (XParam, XBlock, Xev, zb);
	maskbndGPUright << <gridDim, blockDimHalo, 0, streams[2] >> > (XParam, XBlock, Xev, zb);
	maskbndGPUtop << <gridDim, blockDimHalo, 0, streams[3] >> > (XParam, XBlock, Xev, zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	for (int i = 0; i < num_streams; i++)
	{
		cudaStreamDestroy(streams[i]);
	}

	
}
template void fillHaloGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> Xev,float *zb);
template void fillHaloGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> Xev,double* zb);

template <class T> void fillHalo(Param XParam, BlockP<T> XBlock, GradientsP<T> Grad)
{
	
	/*
	std::thread t0(fillHaloF<T>,XParam, true, XBlock, Grad.dhdx);
	std::thread t1(fillHaloF<T>,XParam, true, XBlock, Grad.dudx);
	std::thread t2(fillHaloF<T>,XParam, true, XBlock, Grad.dvdx);
	std::thread t3(fillHaloF<T>,XParam, true, XBlock, Grad.dzsdx);

	std::thread t4(fillHaloF<T>,XParam, true, XBlock, Grad.dhdy);
	std::thread t5(fillHaloF<T>,XParam, true, XBlock, Grad.dudy);
	std::thread t6(fillHaloF<T>,XParam, true, XBlock, Grad.dvdy);
	std::thread t7(fillHaloF<T>,XParam, true, XBlock, Grad.dzsdy);

	*/
	std::thread t0(fillHaloC<T>, XParam, XBlock, Grad.dhdx);
	std::thread t1(fillHaloC<T>, XParam, XBlock, Grad.dudx);
	std::thread t2(fillHaloC<T>, XParam, XBlock, Grad.dvdx);
	std::thread t3(fillHaloC<T>, XParam, XBlock, Grad.dzsdx);

	std::thread t4(fillHaloC<T>, XParam, XBlock, Grad.dhdy);
	std::thread t5(fillHaloC<T>, XParam, XBlock, Grad.dudy);
	std::thread t6(fillHaloC<T>, XParam, XBlock, Grad.dvdy);
	std::thread t7(fillHaloC<T>, XParam, XBlock, Grad.dzsdy);

	t0.join();
	t1.join();
	t2.join();
	t3.join();
	t4.join();
	t5.join();
	t6.join();
	t7.join();

	
}
template void fillHalo<float>(Param XParam, BlockP<float> XBlock, GradientsP<float> Grad);
template void fillHalo<double>(Param XParam, BlockP<double> XBlock, GradientsP<double> Grad);

template <class T> void fillHaloGPU(Param XParam, BlockP<T> XBlock, GradientsP<T> Grad)
{
	const int num_streams = 8;

	cudaStream_t streams[num_streams];

	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}
		
	fillHaloGPU(XParam, XBlock, streams[0], Grad.dhdx);
	fillHaloGPU(XParam, XBlock, streams[2], Grad.dudx);
	fillHaloGPU(XParam, XBlock, streams[3], Grad.dvdx);
	fillHaloGPU(XParam, XBlock, streams[4], Grad.dzsdx);

	fillHaloGPU(XParam, XBlock, streams[5], Grad.dhdy);
	fillHaloGPU(XParam, XBlock, streams[6], Grad.dudy);
	fillHaloGPU(XParam, XBlock, streams[7], Grad.dvdy);
	fillHaloGPU(XParam, XBlock, streams[1], Grad.dzsdy);

	for (int i = 0; i < num_streams; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
	
}
template void fillHaloGPU<float>(Param XParam, BlockP<float> XBlock, GradientsP<float> Grad);
template void fillHaloGPU<double>(Param XParam, BlockP<double> XBlock, GradientsP<double> Grad);


template <class T> void fillHalo(Param XParam, BlockP<T> XBlock, FluxP<T> Flux)
{
	
		
	std::thread t0(fillHaloTopRightC<T>,XParam, XBlock, Flux.Fhu);
	std::thread t1(fillHaloTopRightC<T>,XParam, XBlock, Flux.Fhv);
	std::thread t2(fillHaloTopRightC<T>,XParam, XBlock, Flux.Fqux);
	std::thread t3(fillHaloTopRightC<T>, XParam, XBlock, Flux.Fquy);

	std::thread t4(fillHaloTopRightC<T>, XParam, XBlock, Flux.Fqvx);
	std::thread t5(fillHaloTopRightC<T>, XParam, XBlock, Flux.Fqvy);
	std::thread t6(fillHaloTopRightC<T>, XParam, XBlock, Flux.Su);
	std::thread t7(fillHaloTopRightC<T>, XParam, XBlock, Flux.Sv);

	t0.join();
	t1.join();
	t2.join();
	t3.join();
	t4.join();
	t5.join();
	t6.join();
	t7.join();
	
}
template void fillHalo<float>(Param XParam, BlockP<float> XBlock, FluxP<float> Flux);
template void fillHalo<double>(Param XParam, BlockP<double> XBlock, FluxP<double> Flux);

template <class T> void fillHaloGPU(Param XParam, BlockP<T> XBlock, FluxP<T> Flux)
{
	const int num_streams = 8;

	cudaStream_t streams[num_streams];

	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}

	fillHaloTopRightGPU(XParam, XBlock, streams[0], Flux.Fhu);
	fillHaloTopRightGPU(XParam, XBlock, streams[1], Flux.Fhv);
	fillHaloTopRightGPU(XParam, XBlock, streams[2], Flux.Fqux);
	fillHaloTopRightGPU(XParam, XBlock, streams[3], Flux.Fquy);

	fillHaloTopRightGPU(XParam, XBlock, streams[4], Flux.Fqvx);
	fillHaloTopRightGPU(XParam, XBlock, streams[5], Flux.Fqvy);
	fillHaloTopRightGPU(XParam, XBlock, streams[6], Flux.Su);
	fillHaloTopRightGPU(XParam, XBlock, streams[7], Flux.Sv);

	for (int i = 0; i < num_streams; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
	
}
template void fillHaloGPU<float>(Param XParam, BlockP<float> XBlock, FluxP<float> Flux);
template void fillHaloGPU<double>(Param XParam, BlockP<double> XBlock, FluxP<double> Flux);




template <class T> void fillLeft(Param XParam, int ib, BlockP<T> XBlock, T* &z)
{
	int jj,bb;
	int read, write;
	int ii, ir, it, itr;


	if (XBlock.LeftBot[ib] == ib)//The lower half is a boundary 
	{
		for (int j = 0; j < (XParam.blkwidth / 2); j++)
		{

			read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			z[write] = z[read];
		}

		if (XBlock.LeftTop[ib] == ib) // boundary on the top half too
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				write = memloc(XParam, -1, j, ib);
				jj = (j - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, (XParam.blkwidth - 1), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam, (XParam.blkwidth - 2), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam, (XParam.blkwidth - 1), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam, (XParam.blkwidth - 2), jj + 1, XBlock.LeftTop[ib]);

				z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[ XBlock.LeftBot[ib] ]) // LeftTop block does not exist
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//

			write = memloc(XParam, -1, j, ib);
			read = memloc(XParam, (XParam.blkwidth - 1), j, XBlock.LeftBot[ib]);
			z[write] = z[read];
		}
	}
	else if (XBlock.level[XBlock.LeftBot[ib] ]> XBlock.level[ib])
	{

		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{

			write = memloc(XParam, -1, j, ib);

			jj = j * 2;
			bb = XBlock.LeftBot[ib];

			ii = memloc(XParam, (XParam.blkwidth - 1), jj, bb);
			ir = memloc(XParam, (XParam.blkwidth - 2), jj, bb);
			it = memloc(XParam, (XParam.blkwidth - 1), jj + 1, bb);
			itr = memloc(XParam, (XParam.blkwidth - 2), jj + 1, bb);

			z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.LeftTop[ib] == ib)
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//
				jj = (j - 8) * 2;
				bb = XBlock.LeftTop[ib];

				//read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				//z[write] = z[read];
				ii = memloc(XParam, (XParam.blkwidth - 1), jj, bb);
				ir = memloc(XParam, (XParam.blkwidth - 2), jj, bb);
				it = memloc(XParam, (XParam.blkwidth - 1), jj + 1, bb);
				itr = memloc(XParam, (XParam.blkwidth - 2), jj + 1, bb);

				z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			write = memloc(XParam, -1, j, ib);

			T w1, w2, w3;
			

			int jj = XBlock.RightBot[XBlock.LeftBot[ib]] == ib?ceil(j * (T)0.5): ceil(j * (T)0.5)+ XParam.blkwidth/2;
			w1 = T(1.0 / 3.0);
			w2 = ceil(j * (T)0.5) * 2 > j ? T(1.0 / 6.0) : T(0.5);
			w3 = ceil(j * (T)0.5) * 2 > j ? T(0.5) : T(1.0 / 6.0);
						
			ii= memloc(XParam, 0, j, ib);
			ir= memloc(XParam, XParam.blkwidth-1, jj, XBlock.LeftBot[ib]);
			it = memloc(XParam, XParam.blkwidth-1, jj - 1, XBlock.LeftBot[ib]);
			//2 scenarios here ib is the rightbot neighbour of the leftbot block or ib is the righttop neighbour
			if (XBlock.RightBot[XBlock.LeftBot[ib]] == ib)
			{
				if (j == 0)
				{
					if (XBlock.BotRight[XBlock.LeftBot[ib]] == XBlock.LeftBot[ib]) // no botom of leftbot block
					{
						w3 = (T)0.5 * (1.0 - w1);
						w2 = w3;
						it = ir;

					}
					else if (XBlock.level[XBlock.BotRight[XBlock.LeftBot[ib]]] < XBlock.level[XBlock.LeftBot[ib]]) // exists but is coarser
					{
						w1 = T(4.0 / 10.0);
						w2 = T(5.0 / 10.0);
						w3 = T(1.0 / 10.0);
						it = memloc(XParam, XParam.blkwidth-1, XParam.blkwidth - 1, XBlock.BotRight[XBlock.LeftBot[ib]]);
					}
					else if (XBlock.level[XBlock.BotRight[XBlock.LeftBot[ib]]] == XBlock.level[XBlock.LeftBot[ib]]) // exists with same level
					{
						it = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.BotRight[XBlock.LeftBot[ib]]);
					}
					else if (XBlock.level[XBlock.BotRight[XBlock.LeftBot[ib]]] > XBlock.level[XBlock.LeftBot[ib]]) // exists with higher level
					{
						w1 = T(1.0 / 4.0);
						w2 = T(1.0 / 2.0);
						w3 = T(1.0 / 4.0);
						it = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.BotRight[XBlock.LeftBot[ib]]);
					}
					
					
				}
									
				
			}
			else//righttopleftif == ib
			{
				if (j == (XParam.blkwidth - 1))
				{
					if (XBlock.TopRight[XBlock.LeftTop[ib]] == XBlock.LeftTop[ib]) // no botom of leftbot block
					{
						w3 = 0.5*(1.0-w1);
						w2 = w3;
						ir = it;

					}
					else if (XBlock.level[XBlock.TopRight[XBlock.LeftTop[ib]]] < XBlock.level[XBlock.LeftTop[ib]]) // exists but is coarser
					{
						w1 = 4.0 / 10.0;
						w2 = 1.0 / 10.0;
						w3 = 5.0 / 10.0;
						ir = memloc(XParam, XParam.blkwidth - 1,0, XBlock.TopRight[XBlock.LeftTop[ib]]);
					}
					else if (XBlock.level[XBlock.TopRight[XBlock.LeftTop[ib]]] == XBlock.level[XBlock.LeftTop[ib]]) // exists with same level
					{
						ir = memloc(XParam, XParam.blkwidth - 1, 0, XBlock.TopRight[XBlock.LeftTop[ib]]);
					}
					else if (XBlock.level[XBlock.TopRight[XBlock.LeftTop[ib]]] > XBlock.level[XBlock.LeftTop[ib]]) // exists with higher level
					{
						w1 = 1.0 / 4.0;
						w2 = 1.0 / 2.0;
						w3 = 1.0 / 4.0;
						ir = memloc(XParam, XParam.blkwidth - 1, 0, XBlock.TopRight[XBlock.LeftTop[ib]]);
					}
				}
				//
			}


			z[write] = w1 * z[ii] + w2 * z[ir] + w3 * z[it];
		}
	}
	


}


template <class T> __global__ void fillLeft(int halowidth, int* active, int * level, int* leftbot, int * lefttop, int * rightbot, int* botright,int * topright, T * a)
{
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = 0;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = active[ibl];

	int lev = level[ib];
	int LB = leftbot[ib];
	int LT = lefttop[ib];

	int RBLB = rightbot[LB];
	int BRLB = botright[LB];
	int TRLT = topright[LT];

	int levBRLB = level[BRLB];
	int levTRLT = level[TRLT];
	int levLB = level[LB];
	int levLT = level[LT];
	int write = memloc(halowidth, blkmemwidth, -1, iy, ib);
	int read;
	int jj, ii, ir, it, itr;
	T a_read;
	T w1, w2, w3;

	if (LB == ib)
	{
		if (iy < (blockDim.y / 2))
		{
			read = memloc(halowidth, blkmemwidth, 0, iy, ib);
			a_read = a[read];
		}
		else
		{
			if (LT == ib)
			{
				read = memloc(halowidth, blkmemwidth, 0, iy, ib);
				a_read = a[read];

			}
			else
			{
				
				jj = (iy - (blockDim.y / 2)) * 2;
				ii = memloc(halowidth, blkmemwidth, (blockDim.y - 1), jj, LT);
				ir = memloc(halowidth, blkmemwidth, (blockDim.y - 2), jj, LT);
				it = memloc(halowidth, blkmemwidth, (blockDim.y - 1), jj + 1, LT);
				itr = memloc(halowidth, blkmemwidth, (blockDim.y - 2), jj + 1, LT);

				a_read = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}
		
	}
	else if (levLB == lev )
	{
		read = memloc(halowidth, blkmemwidth, (blockDim.y - 1), iy, LB);
		a_read = a[read];
	}
	else if (levLB > lev)
	{
		if (iy < (blockDim.y / 2))
		{
			jj = iy * 2;
			ii = memloc(halowidth, blkmemwidth, (blockDim.y - 1), jj, LB);
			ir = memloc(halowidth, blkmemwidth, (blockDim.y - 2), jj, LB);
			it = memloc(halowidth, blkmemwidth, (blockDim.y - 1), jj + 1, LB);
			itr = memloc(halowidth, blkmemwidth, (blockDim.y - 2), jj + 1, LB);
			a_read= T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		else
		{
			if (LT == ib)
			{
				read = memloc(halowidth, blkmemwidth, 0, iy, ib);
				a_read = a[read];
			}
			else
			{
				jj = (iy - (blockDim.y / 2)) * 2;
								
				ii = memloc(halowidth, blkmemwidth, (blockDim.y - 1), jj, LT);
				ir = memloc(halowidth, blkmemwidth, (blockDim.y - 2), jj, LT);
				it = memloc(halowidth, blkmemwidth, (blockDim.y - 1), jj + 1, LT);
				itr = memloc(halowidth, blkmemwidth, (blockDim.y - 2), jj + 1, LT);

				a_read = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}
	}
	else if (levLB < lev)
	{
		jj = RBLB==ib? ceil(iy * (T)0.5): ceil(iy * (T)0.5) + blockDim.y / 2;
		w1 = (T)1.0 / (T)3.0;
		w2 = ceil(iy * (T)0.5) * 2 > iy ? T(1.0 / 6.0) : T(0.5);
		w3 = ceil(iy * (T)0.5) * 2 > iy ? T(0.5) : T(1.0 / 6.0);

		ii = memloc(halowidth, blkmemwidth, 0, iy, ib);
		ir = memloc(halowidth, blkmemwidth, blockDim.y - 1, jj, LB);
		it = memloc(halowidth, blkmemwidth, blockDim.y - 1, jj - 1, LB);
		if (RBLB == ib)
		{
			if (iy == 0)
			{
				if (BRLB == LB)
				{
					w3 = (T)0.5 * (1.0 - w1);
					w2 = w3;
					it = ir;
				}
				else if (levBRLB < levLB)
				{
					w1 = T(4.0 / 10.0);
					w2 = T(5.0 / 10.0);
					w3 = T(1.0 / 10.0);
					it = memloc(halowidth, blkmemwidth, blockDim.y - 1, blockDim.y - 1, BRLB);

				}
				else if (levBRLB == levLB)
				{
					it = memloc(halowidth, blkmemwidth, blockDim.y - 1, blockDim.y - 1, BRLB);
				}
				else if (levBRLB > levLB)
				{
					w1 = T(1.0 / 4.0);
					w2 = T(1.0 / 2.0);
					w3 = T(1.0 / 4.0);
					it = memloc(halowidth, blkmemwidth, blockDim.y - 1, blockDim.y - 1, BRLB);
				}
			}
		}
		else
		{
			if (iy == (blockDim.y - 1))
			{
				if (TRLT == LT)
				{
					w3 = 0.5 * (1.0 - w1);
					w2 = w3;
					ir = it;
				}
				else if (levTRLT < levLT)
				{
					w1 = 4.0 / 10.0;
					w2 = 1.0 / 10.0;
					w3 = 5.0 / 10.0;
					ir = memloc(halowidth, blkmemwidth, blockDim.y - 1, 0, TRLT);
				}
				else if (levTRLT == levLT)
				{
					ir = memloc(halowidth, blkmemwidth, blockDim.y - 1, 0, TRLT);
				}
				else if (levTRLT > levLT)
				{
					w1 = 1.0 / 4.0;
					w2 = 1.0 / 2.0;
					w3 = 1.0 / 4.0;
					ir = memloc(halowidth, blkmemwidth, blockDim.y - 1, 0, TRLT);
					
				}
			}
		}
		a_read = w1 * a[ii] + w2 * a[ir] + w3 * a[it];
	}

	a[write] = a_read;
}
template __global__ void fillLeft<float>(int halowidth, int* active, int* level, int* leftbot, int* lefttop, int* rightbot, int* botright, int* topright, float* a);
template __global__ void fillLeft<double>(int halowidth, int* active, int* level, int* leftbot, int* lefttop, int* rightbot, int* botright, int* topright, double* a);


template <class T> void fillLeftFlux(Param XParam, bool doProlongation, int ib, BlockP<T> XBlock, T*& z)
{
	int jj, bb;
	int read, write;
	int ii, ir, it, itr;


	if (XBlock.LeftBot[ib] == ib)//The lower half is a boundary 
	{
		for (int j = 0; j < (XParam.blkwidth / 2); j++)
		{

			read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			z[write] = z[read];
		}

		if (XBlock.LeftTop[ib] == ib) // boundary on the top half too
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				write = memloc(XParam, -1, j, ib);
				jj = (j - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, (XParam.blkwidth - 1), jj, XBlock.LeftTop[ib]);
				//ir = memloc(XParam, (XParam.blkwidth - 2), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam, (XParam.blkwidth - 1), jj + 1, XBlock.LeftTop[ib]);
				//itr = memloc(XParam, (XParam.blkwidth - 2), jj + 1, XBlock.LeftTop[ib]);

				z[write] = T(0.5) * (z[ii]  + z[it]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.LeftBot[ib]]) // LeftTop block does not exist
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//

			write = memloc(XParam, -1, j, ib);
			read = memloc(XParam, (XParam.blkwidth - 1), j, XBlock.LeftBot[ib]);
			z[write] = z[read];
		}
	}
	else if (XBlock.level[XBlock.LeftBot[ib]] > XBlock.level[ib])
	{

		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{

			write = memloc(XParam, -1, j, ib);

			jj = j * 2;
			bb = XBlock.LeftBot[ib];

			ii = memloc(XParam, (XParam.blkwidth - 1), jj, bb);
			//ir = memloc(XParam, (XParam.blkwidth - 2), jj, bb);
			it = memloc(XParam, (XParam.blkwidth - 1), jj + 1, bb);
			//itr = memloc(XParam, (XParam.blkwidth - 2), jj + 1, bb);

			z[write] = T(0.5) * (z[ii] + z[it]);
		}
		//now find out aboy lefttop block
		if (XBlock.LeftTop[ib] == ib)
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//
				jj = (j - 8) * 2;
				bb = XBlock.LeftTop[ib];

				//read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				//z[write] = z[read];
				ii = memloc(XParam, (XParam.blkwidth - 1), jj, bb);
				//ir = memloc(XParam, (XParam.blkwidth - 2), jj, bb);
				it = memloc(XParam, (XParam.blkwidth - 1), jj + 1, bb);
				//itr = memloc(XParam, (XParam.blkwidth - 2), jj + 1, bb);

				z[write] = T(0.5) * (z[ii] + z[it]);
			}
		}

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			write = memloc(XParam, -1, j, ib);

			T w1, w2, w3;


			int jj = XBlock.RightBot[XBlock.LeftBot[ib]] == ib ? ceil(j * (T)0.5) : ceil(j * (T)0.5) + XParam.blkwidth / 2;
			

			ii = memloc(XParam, XParam.blkwidth - 1, jj, XBlock.LeftBot[ib]);
			if (doProlongation)
				z[write] = z[ii];


			
		}
	}



}



template <class T> void fillRight(Param XParam, int ib, BlockP<T> XBlock, T*& z)
{
	int jj, bb;
	int read, write;
	int ii, ir, it, itr;


	if (XBlock.RightBot[ib] == ib)//The lower half is a boundary 
	{
		for (int j = 0; j < (XParam.blkwidth / 2); j++)
		{

			read = memloc(XParam, XParam.blkwidth-1, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			write = memloc(XParam, XParam.blkwidth, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			z[write] = z[read];
		}

		if (XBlock.RightTop[ib] == ib) // boundary on the top half too
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, XParam.blkwidth - 1, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, XParam.blkwidth, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				write = memloc(XParam, XParam.blkwidth, j, ib);
				jj = (j - 8) * 2;
				ii = memloc(XParam, 0, jj, XBlock.RightTop[ib]);
				ir = memloc(XParam, 1, jj, XBlock.RightTop[ib]);
				it = memloc(XParam, 0, jj + 1, XBlock.RightTop[ib]);
				itr = memloc(XParam, 1, jj + 1, XBlock.RightTop[ib]);

				z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.RightBot[ib]]) // LeftTop block does not exist
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//

			write = memloc(XParam, XParam.blkwidth, j, ib);
			read = memloc(XParam, 0, j, XBlock.RightBot[ib]);
			z[write] = z[read];
		}
	}
	else if (XBlock.level[XBlock.RightBot[ib]] > XBlock.level[ib])
	{

		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{

			write = memloc(XParam, XParam.blkwidth, j, ib);

			jj = j * 2;
			bb = XBlock.RightBot[ib];

			ii = memloc(XParam, 0, jj, bb);
			ir = memloc(XParam, 1, jj, bb);
			it = memloc(XParam, 0, jj + 1, bb);
			itr = memloc(XParam, 1, jj + 1, bb);

			z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.RightTop[ib] == ib)
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, XParam.blkwidth-1, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, XParam.blkwidth, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//
				jj = (j - 8) * 2;
				bb = XBlock.RightTop[ib];

				//read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, XParam.blkwidth, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				//z[write] = z[read];
				ii = memloc(XParam, 0, jj, bb);
				ir = memloc(XParam, 1, jj, bb);
				it = memloc(XParam, 0, jj + 1, bb);
				itr = memloc(XParam, 1, jj + 1, bb);

				z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.RightBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			write = memloc(XParam, XParam.blkwidth, j, ib);

			T w1, w2, w3;
			

			int jj = XBlock.LeftBot[XBlock.RightBot[ib]] == ib ? ceil(j * (T)0.5) : ceil(j * (T)0.5) + XParam.blkwidth / 2;
			w1 = 1.0 / 3.0;
			w2 = ceil(j * (T)0.5) * 2 > j ? T(1.0 / 6.0) : T(0.5);
			w3 = ceil(j * (T)0.5) * 2 > j ? T(0.5) : T(1.0 / 6.0);

			ii = memloc(XParam, XParam.blkwidth-1, j, ib);
			ir = memloc(XParam, 0, jj, XBlock.RightBot[ib]);
			it = memloc(XParam, 0, jj - 1, XBlock.RightBot[ib]);
			//2 scenarios here ib is the leftbot neighbour of the rightbot block or ib is the lefttop neighbour
			if (XBlock.LeftBot[XBlock.RightBot[ib]] == ib)
			{
				if (j == 0)
				{
					if (XBlock.BotLeft[XBlock.RightBot[ib]] == XBlock.RightBot[ib]) // no botom of leftbot block
					{
						w3 = 0.5 * (1.0 - w1);
						w2 = w3;
						it = ir;

					}
					else if (XBlock.level[XBlock.BotLeft[XBlock.RightBot[ib]]] < XBlock.level[XBlock.RightBot[ib]]) // exists but is coarser
					{
						w1 = 4.0 / 10.0;
						w2 = 5.0 / 10.0;
						w3 = 1.0 / 10.0;
						it = memloc(XParam, 0, XParam.blkwidth - 1, XBlock.BotLeft[XBlock.RightBot[ib]]);
					}
					else if (XBlock.level[XBlock.BotLeft[XBlock.RightBot[ib]]] == XBlock.level[XBlock.RightBot[ib]]) // exists with same level
					{
						it = memloc(XParam, 0, XParam.blkwidth - 1, XBlock.BotLeft[XBlock.RightBot[ib]]);
					}
					else if (XBlock.level[XBlock.BotLeft[XBlock.RightBot[ib]]] > XBlock.level[XBlock.RightBot[ib]]) // exists with higher level
					{
						w1 = 1.0 / 4.0;
						w2 = 1.0 / 2.0;
						w3 = 1.0 / 4.0;
						it = memloc(XParam, 0, XParam.blkwidth - 1, XBlock.BotLeft[XBlock.RightBot[ib]]);
					}


				}


			}
			else//
			{
				if (j == (XParam.blkwidth - 1))
				{
					if (XBlock.TopLeft[XBlock.RightTop[ib]] == XBlock.RightTop[ib]) // no botom of leftbot block
					{
						w3 = 0.5 * (1.0 - w1);
						w2 = w3;
						ir = it;

					}
					else if (XBlock.level[XBlock.TopLeft[XBlock.RightTop[ib]]] < XBlock.level[XBlock.RightTop[ib]]) // exists but is coarser
					{
						w1 = 4.0 / 10.0;
						w2 = 1.0 / 10.0;
						w3 = 5.0 / 10.0;
						ir = memloc(XParam, 0, 0, XBlock.TopLeft[XBlock.RightTop[ib]]);
					}
					else if (XBlock.level[XBlock.TopLeft[XBlock.RightTop[ib]]] == XBlock.level[XBlock.RightTop[ib]]) // exists with same level
					{
						ir = memloc(XParam, 0, 0, XBlock.TopLeft[XBlock.RightTop[ib]]);
					}
					else if (XBlock.level[XBlock.TopLeft[XBlock.RightTop[ib]]] > XBlock.level[XBlock.RightTop[ib]]) // exists with higher level
					{
						w1 = 1.0 / 4.0;
						w2 = 1.0 / 2.0;
						w3 = 1.0 / 4.0;
						ir = memloc(XParam, 0, 0, XBlock.TopLeft[XBlock.RightTop[ib]]);
					}
				}
				//
			}


			z[write] = w1 * z[ii] + w2 * z[ir] + w3 * z[it];
		}
	}



}



template <class T> __global__ void fillRight(int halowidth, int* active, int* level, int * rightbot,int* righttop,int * leftbot,int*botleft,int* topleft, T* a)
{
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = blockDim.y - 1;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = active[ibl];

	int RB = rightbot[ib];
	int RT = righttop[ib];
	int LB = leftbot[ib];
	int BL = botleft[ib];
	int LBRB = leftbot[RB];
	int TLRT = topleft[RT];
	int BLRB = botleft[RB];


	int lev = level[ib];
	int levRB = level[RB];
	int levRT = level[RT];
	int levBLRB = level[BLRB];
	int levTLRT = level[TLRT];

	int write = memloc(halowidth, blkmemwidth, blockDim.y, iy, ib);
	int read;
	int jj, ii, ir, it, itr;
	T a_read;
	T w1, w2, w3;


	if (RB == ib)
	{
		if (iy < (blockDim.y / 2))
		{
			read = memloc(halowidth, blkmemwidth, blockDim.y - 1, iy, ib);
			a_read = a[read];
		}
		else
		{
			if (RT == ib)
			{
				read = memloc(halowidth, blkmemwidth, blockDim.y - 1, iy, ib);
				a_read = a[read];
			}
			else
			{
				jj = (iy - (blockDim.y / 2)) * 2;
				ii = memloc(halowidth, blkmemwidth, 0, jj, RT);
				ir = memloc(halowidth, blkmemwidth, 1, jj, RT);
				it = memloc(halowidth, blkmemwidth, 0, jj + 1, RT);
				itr = memloc(halowidth, blkmemwidth, 1, jj + 1, RT);

				a_read = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}
	}
	else if (levRB == lev)
	{
		read = memloc(halowidth, blkmemwidth, 0, iy, RB);
		a_read = a[read];
	}
	else if (levRB > lev)
	{
		if (iy < (blockDim.y / 2))
		{
			jj = iy * 2;


			ii = memloc(halowidth, blkmemwidth, 0, jj, RB);
			ir = memloc(halowidth, blkmemwidth, 1, jj, RB);
			it = memloc(halowidth, blkmemwidth, 0, jj + 1, RB);
			itr = memloc(halowidth, blkmemwidth, 1, jj + 1, RB);

			a_read = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		else
		{
			if (RT == ib)
			{
				read = memloc(halowidth, blkmemwidth, blockDim.y - 1, iy, ib);
				a_read = a[read];
			}
			else
			{
				jj = (iy - (blockDim.y / 2)) * 2;
				
				ii = memloc(halowidth, blkmemwidth, 0, jj, RT);
				ir = memloc(halowidth, blkmemwidth, 1, jj, RT);
				it = memloc(halowidth, blkmemwidth, 0, jj + 1, RT);
				itr = memloc(halowidth, blkmemwidth, 1, jj + 1, RT);

				a_read = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}
	}
	else if (levRB < lev)
	{
		//
		jj = LBRB == ib ? ceil(iy * (T)0.5) : ceil(iy * (T)0.5) + blockDim.y / 2;
		w1 = 1.0 / 3.0;
		w2 = ceil(iy * (T)0.5) * 2 > iy ? T(1.0 / 6.0) : T(0.5);
		w3 = ceil(iy * (T)0.5) * 2 > iy ? T(0.5) : T(1.0 / 6.0);
		ii = memloc(halowidth, blkmemwidth, blockDim.y - 1, iy, ib);
		ir = memloc(halowidth, blkmemwidth, 0, jj, RB);
		it = memloc(halowidth, blkmemwidth, 0, jj - 1, RB);
		if (LBRB == ib)
		{
			if (iy == 0)
			{
				if (BLRB == RB)
				{
					w3 = 0.5 * (1.0 - w1);
					w2 = w3;
					it = ir;
				}
				else if (levBLRB < levRB)
				{
					w1 = 4.0 / 10.0;
					w2 = 5.0 / 10.0;
					w3 = 1.0 / 10.0;
					it = memloc(halowidth, blkmemwidth, 0, blockDim.y - 1, BLRB);
				}
				else if (levBLRB == levRB)
				{
					it = memloc(halowidth, blkmemwidth, 0, blockDim.y - 1, BLRB);
				}
				else if (levBLRB > levRB)
				{
					w1 = 1.0 / 4.0;
					w2 = 1.0 / 2.0;
					w3 = 1.0 / 4.0;
					it = memloc(halowidth, blkmemwidth, 0, blockDim.y - 1, BLRB);
				}
			}
		}
		else
		{
			if (iy == (blockDim.y - 1))
			{
				if (TLRT == RT)
				{
					w3 = 0.5 * (1.0 - w1);
					w2 = w3;
					ir = it;
				}
				else if (levTLRT < levRT)
				{
					w1 = 4.0 / 10.0;
					w2 = 1.0 / 10.0;
					w3 = 5.0 / 10.0;
					ir = memloc(halowidth, blkmemwidth, 0, 0, TLRT);
				}
				else if (levTLRT == levRT)
				{
					ir = memloc(halowidth, blkmemwidth, 0, 0, TLRT);
				}
				else if (levTLRT > levRT)
				{
					w1 = 1.0 / 4.0;
					w2 = 1.0 / 2.0;
					w3 = 1.0 / 4.0;
					ir = memloc(halowidth, blkmemwidth, 0, 0, TLRT);
				}
			}
		}

		a_read= w1 * a[ii] + w2 * a[ir] + w3 * a[it];
	}
	a[write] = a_read;
}

template __global__ void fillRight<float>(int halowidth, int* active, int* level, int* rightbot, int* righttop, int* leftbot, int* botleft, int* topleft, float* a);
template __global__ void fillRight<double>(int halowidth, int* active, int* level, int* rightbot, int* righttop, int* leftbot, int* botleft, int* topleft, double* a);

template <class T> void fillRightFlux(Param XParam, bool doProlongation, int ib, BlockP<T> XBlock, T*& z)
{
	int jj, bb;
	int read, write;
	int ii, ir, it, itr;


	if (XBlock.RightBot[ib] == ib)//The lower half is a boundary 
	{
		for (int j = 0; j < (XParam.blkwidth / 2); j++)
		{

			read = memloc(XParam, XParam.blkwidth - 1, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			write = memloc(XParam, XParam.blkwidth, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			z[write] = z[read];
		}

		if (XBlock.RightTop[ib] == ib) // boundary on the top half too
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, XParam.blkwidth - 1, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, XParam.blkwidth, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				write = memloc(XParam, XParam.blkwidth, j, ib);
				jj = (j - 8) * 2;
				ii = memloc(XParam, 0, jj, XBlock.RightTop[ib]);
				//ir = memloc(XParam, 1, jj, XBlock.RightTop[ib]);
				it = memloc(XParam, 0, jj + 1, XBlock.RightTop[ib]);
				//itr = memloc(XParam, 1, jj + 1, XBlock.RightTop[ib]);

				z[write] = T(0.5) * (z[ii] + z[it]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.RightBot[ib]]) // LeftTop block does not exist
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//

			write = memloc(XParam, XParam.blkwidth, j, ib);
			read = memloc(XParam, 0, j, XBlock.RightBot[ib]);
			z[write] = z[read];
		}
	}
	else if (XBlock.level[XBlock.RightBot[ib]] > XBlock.level[ib])
	{

		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{

			write = memloc(XParam, XParam.blkwidth, j, ib);

			jj = j * 2;
			bb = XBlock.RightBot[ib];

			ii = memloc(XParam, 0, jj, bb);
			//ir = memloc(XParam, 1, jj, bb);
			it = memloc(XParam, 0, jj + 1, bb);
			//itr = memloc(XParam, 1, jj + 1, bb);

			//z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
			z[write] = T(0.5) * (z[ii] + z[it]);
		}
		//now find out aboy lefttop block
		if (XBlock.RightTop[ib] == ib)
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, XParam.blkwidth - 1, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, XParam.blkwidth, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//
				jj = (j - 8) * 2;
				bb = XBlock.RightTop[ib];

				//read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, XParam.blkwidth, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				//z[write] = z[read];
				ii = memloc(XParam, 0, jj, bb);
				//ir = memloc(XParam, 1, jj, bb);
				it = memloc(XParam, 0, jj + 1, bb);
				//itr = memloc(XParam, 1, jj + 1, bb);
				z[write] = T(0.5) * (z[ii] + z[it]);
				//z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.RightBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			write = memloc(XParam, XParam.blkwidth, j, ib);


			int jj = XBlock.LeftBot[XBlock.RightBot[ib]] == ib ? floor(j * (T)0.5) : floor(j * (T)0.5) + XParam.blkwidth / 2;

			ii = memloc(XParam, 0, jj, XBlock.RightBot[ib]);
			if (doProlongation)
				z[write] = z[ii];
		}
	}



}

template void fillRightFlux<float>(Param XParam, bool doProlongation, int ib, BlockP<float> XBlock, float*& z);
template void fillRightFlux<double>(Param XParam, bool doProlongation, int ib, BlockP<double> XBlock, double*& z);



template <class T> __global__ void fillRightFlux(int halowidth, bool doProlongation, int* active, int* level, int* rightbot, int* righttop, int* leftbot, int* botleft, int* topleft, T* a)
{
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = blockDim.y - 1;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = active[ibl];

	int RB = rightbot[ib];
	int RT = righttop[ib];
	int LB = leftbot[ib];
	int BL = botleft[ib];
	int LBRB = leftbot[RB];
	int TLRT = topleft[RT];
	int BLRB = botleft[RB];


	int lev = level[ib];
	int levRB = level[RB];
	int levRT = level[RT];
	int levBLRB = level[BLRB];
	int levTLRT = level[TLRT];

	int write = memloc(halowidth, blkmemwidth, blockDim.y, iy, ib);
	int read;
	int jj, ii, ir, it, itr;
	T a_read;
	T w1, w2, w3;


	if (RB == ib)
	{
		if (iy < (blockDim.y / 2))
		{
			read = memloc(halowidth, blkmemwidth, blockDim.y - 1, iy, ib);
			a_read = a[read];
		}
		else
		{
			if (RT == ib)
			{
				read = memloc(halowidth, blkmemwidth, blockDim.y - 1, iy, ib);
				a_read = a[read];
			}
			else
			{
				jj = (iy - (blockDim.y / 2)) * 2;
				ii = memloc(halowidth, blkmemwidth, 0, jj, RT);
				//ir = memloc(halowidth, blkmemwidth, 1, jj, RT);
				it = memloc(halowidth, blkmemwidth, 0, jj + 1, RT);
				//itr = memloc(halowidth, blkmemwidth, 1, jj + 1, RT);

				a_read = T(0.5) * (a[ii] + a[it]);
			}
		}
	}
	else if (levRB == lev)
	{
		read = memloc(halowidth, blkmemwidth, 0, iy, RB);
		a_read = a[read];
	}
	else if (levRB > lev)
	{
		if (iy < (blockDim.y / 2))
		{
			jj = iy * 2;


			ii = memloc(halowidth, blkmemwidth, 0, jj, RB);
			//ir = memloc(halowidth, blkmemwidth, 1, jj, RB);
			it = memloc(halowidth, blkmemwidth, 0, jj + 1, RB);
			//itr = memloc(halowidth, blkmemwidth, 1, jj + 1, RB);

			a_read = T(0.5) * (a[ii] + a[it]);
		}
		else
		{
			if (RT == ib)
			{
				read = memloc(halowidth, blkmemwidth, blockDim.y - 1, iy, ib);
				a_read = a[read];
			}
			else
			{
				jj = (iy - (blockDim.y / 2)) * 2;

				ii = memloc(halowidth, blkmemwidth, 0, jj, RT);
				//ir = memloc(halowidth, blkmemwidth, 1, jj, RT);
				it = memloc(halowidth, blkmemwidth, 0, jj + 1, RT);
				//itr = memloc(halowidth, blkmemwidth, 1, jj + 1, RT);

				a_read = T(0.5) * (a[ii] + a[it] );
			}
		}
	}
	else if (levRB < lev)
	{
		//
		jj = LBRB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + blockDim.y / 2;
		
		
		ir = memloc(halowidth, blkmemwidth, 0, jj, RB);
		
		if (doProlongation)
			a_read = a[ir];
		
		else
			a_read = a[write];
	}
	a[write] = a_read;
}
template __global__ void fillRightFlux<float>(int halowidth, bool doProlongation, int* active, int* level, int* rightbot, int* righttop, int* leftbot, int* botleft, int* topleft, float* a);
template __global__ void fillRightFlux<double>(int halowidth, bool doProlongation, int* active, int* level, int* rightbot, int* righttop, int* leftbot, int* botleft, int* topleft, double* a);



template <class T> void fillBot(Param XParam, int ib, BlockP<T> XBlock, T*& z)
{
	int jj, bb;
	int read, write;
	int ii, ir, it, itr;


	if (XBlock.BotLeft[ib] == ib)//The lower half is a boundary 
	{
		for (int j = 0; j < (XParam.blkwidth / 2); j++)
		{

			read = memloc(XParam, j, 0, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			write = memloc(XParam, j, -1, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			z[write] = z[read];
		}

		if (XBlock.BotRight[ib] == ib) // boundary on the top half too
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, j, 0, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, -1, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				write = memloc(XParam,j, -1, ib);
				jj = (j - 8) * 2;
				ii = memloc(XParam,jj, (XParam.blkwidth - 1), XBlock.BotRight[ib]);
				ir = memloc(XParam,jj, (XParam.blkwidth - 2), XBlock.BotRight[ib]);
				it = memloc(XParam,jj+1, (XParam.blkwidth - 1), XBlock.BotRight[ib]);
				itr = memloc(XParam,jj+1, (XParam.blkwidth - 2), XBlock.BotRight[ib]);

				z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.BotLeft[ib]]) // LeftTop block does not exist
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//

			write = memloc(XParam,j, -1, ib);
			read = memloc(XParam, j, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
			z[write] = z[read];
		}
	}
	else if (XBlock.level[XBlock.BotLeft[ib]] > XBlock.level[ib])
	{

		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{

			write = memloc(XParam, j, -1, ib);

			jj = j * 2;
			bb = XBlock.BotLeft[ib];

			ii = memloc(XParam, jj, (XParam.blkwidth - 1), bb);
			ir = memloc(XParam, jj, (XParam.blkwidth - 2), bb);
			it = memloc(XParam, jj + 1, (XParam.blkwidth - 1), bb);
			itr = memloc(XParam, jj + 1, (XParam.blkwidth - 2), bb);

			z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
		}
		//now find out aboy botright block
		if (XBlock.BotRight[ib] == ib)
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, j, 0, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, -1, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//
				jj = (j - 8) * 2;
				bb = XBlock.BotRight[ib];

				//read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, -1, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				//z[write] = z[read];
				ii = memloc(XParam, jj, (XParam.blkwidth - 1), bb);
				ir = memloc(XParam, jj, (XParam.blkwidth - 2), bb);
				it = memloc(XParam, jj + 1, (XParam.blkwidth - 1), bb);
				itr = memloc(XParam, jj + 1, (XParam.blkwidth - 2), bb);

				z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			write = memloc(XParam, j, -1, ib);

			T w1, w2, w3;
			

			int jj = XBlock.TopLeft[XBlock.BotLeft[ib]] == ib ? ceil(j * (T)0.5) : ceil(j * (T)0.5) + XParam.blkwidth / 2;
			w1 = 1.0 / 3.0;
			w2 = ceil(j * (T)0.5) * 2 > j ? T(1.0 / 6.0) : T(0.5);
			w3 = ceil(j * (T)0.5) * 2 > j ? T(0.5) : T(1.0 / 6.0);

			ii = memloc(XParam, j, 0, ib);
			ir = memloc(XParam, jj, XParam.blkwidth - 1, XBlock.BotLeft[ib]);
			it = memloc(XParam, jj -1, XParam.blkwidth - 1, XBlock.BotLeft[ib]);
			//2 scenarios here ib is the rightbot neighbour of the leftbot block or ib is the righttop neighbour
			if (XBlock.TopLeft[XBlock.BotLeft[ib]] == ib)
			{
				if (j == 0)
				{
					if (XBlock.LeftTop[XBlock.BotLeft[ib]] == XBlock.BotLeft[ib]) // no botom of leftbot block
					{
						w3 = 0.5 * (1.0 - w1);
						w2 = w3;
						it = ir;

					}
					else if (XBlock.level[XBlock.LeftTop[XBlock.BotLeft[ib]]] < XBlock.level[XBlock.BotLeft[ib]]) // exists but is coarser
					{
						w1 = 4.0 / 10.0;
						w2 = 5.0 / 10.0;
						w3 = 1.0 / 10.0;
						it = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.LeftTop[XBlock.BotLeft[ib]]);
					}
					else if (XBlock.level[XBlock.LeftTop[XBlock.BotLeft[ib]]] == XBlock.level[XBlock.BotLeft[ib]]) // exists with same level
					{
						it = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.LeftTop[XBlock.BotLeft[ib]]);
					}
					else if (XBlock.level[XBlock.LeftTop[XBlock.BotLeft[ib]]] > XBlock.level[XBlock.BotLeft[ib]]) // exists with higher level
					{
						w1 = 1.0 / 4.0;
						w2 = 1.0 / 2.0;
						w3 = 1.0 / 4.0;
						it = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.LeftTop[XBlock.BotLeft[ib]]);
					}


				}


			}
			else//righttopleftif == ib
			{
				if (j == (XParam.blkwidth - 1))
				{
					if (XBlock.RightTop[XBlock.BotRight[ib]] == XBlock.BotRight[ib]) // no botom of leftbot block
					{
						w3 = 0.5 * (1.0 - w1);
						w2 = w3;
						ir = it;

					}
					else if (XBlock.level[XBlock.RightTop[XBlock.BotRight[ib]]] < XBlock.level[XBlock.BotRight[ib]]) // exists but is coarser
					{
						w1 = 4.0 / 10.0;
						w2 = 1.0 / 10.0;
						w3 = 5.0 / 10.0;
						ir = memloc(XParam, 0,XParam.blkwidth - 1, XBlock.RightTop[XBlock.BotRight[ib]]);
					}
					else if (XBlock.level[XBlock.RightTop[XBlock.BotRight[ib]]] == XBlock.level[XBlock.BotRight[ib]]) // exists with same level
					{
						ir = memloc(XParam,0, XParam.blkwidth - 1, XBlock.RightTop[XBlock.BotRight[ib]]);
					}
					else if (XBlock.level[XBlock.RightTop[XBlock.BotRight[ib]]] > XBlock.level[XBlock.BotRight[ib]]) // exists with higher level
					{
						w1 = 1.0 / 4.0;
						w2 = 1.0 / 2.0;
						w3 = 1.0 / 4.0;
						ir = memloc(XParam,0,XParam.blkwidth - 1, XBlock.RightTop[XBlock.BotRight[ib]]);
					}
				}
				//
			}


			z[write] = w1 * z[ii] + w2 * z[ir] + w3 * z[it];
		}
	}



}


template <class T> __global__ void fillBot(int halowidth, int* active, int* level, int* botleft, int* botright, int* topleft, int* lefttop, int* righttop, T* a)
{
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = 0;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = active[ibl];

	int BL = botleft[ib];
	int BR = botright[ib];
	int TLBL = topleft[BL];
	int LTBL = lefttop[BL];
	int RTBR = righttop[BR];

	int lev = level[ib];
	int levBL = level[BL];
	int levBR = level[BR];
	int levLTBL = level[LTBL];
	int levRTBR = level[RTBR];

	int write = memloc(halowidth, blkmemwidth, ix, -1, ib);
	int read;
	int jj, ii, ir, it, itr;
	T a_read;
	T w1, w2, w3;
	if (BL == ib)
	{
		if (ix < (blockDim.x / 2))
		{
			read = memloc(halowidth, blkmemwidth, ix, 0, ib);
			a_read = a[read];
		}
		else
		{
			if (BR == ib)
			{
				read = memloc(halowidth, blkmemwidth, ix, 0, ib);
				a_read = a[read];
			}
			else
			{
				jj = (ix - (blockDim.x/2)) * 2;
				ii = memloc(halowidth, blkmemwidth, jj, (blockDim.x - 1), BR);
				ir = memloc(halowidth, blkmemwidth, jj, (blockDim.x - 2), BR);
				it = memloc(halowidth, blkmemwidth, jj + 1, (blockDim.x - 1), BR);
				itr = memloc(halowidth, blkmemwidth, jj + 1, (blockDim.x - 2), BR);
				a_read = T(0.25)* (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}
	}
	else if (levBL == lev)
	{
		read = memloc(halowidth, blkmemwidth, ix, (blockDim.x - 1), BL);
		a_read = a[read];
	}
	else if (levBL > lev)
	{
		if (ix < (blockDim.x / 2))
		{
			jj = ix * 2;
			ii = memloc(halowidth, blkmemwidth, jj, (blockDim.x - 1), BL);
			ir = memloc(halowidth, blkmemwidth, jj, (blockDim.x - 2), BL);
			it = memloc(halowidth, blkmemwidth, jj + 1, (blockDim.x - 1), BL);
			itr = memloc(halowidth, blkmemwidth, jj + 1, (blockDim.x - 2), BL);
			a_read = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		else
		{
			if (BR == ib)
			{
				read = memloc(halowidth, blkmemwidth, ix, 0, ib);
				a_read = a[read];
			}
			else
			{
				jj = (ix - (blockDim.x/2)) * 2;
				ii = memloc(halowidth, blkmemwidth, jj, (blockDim.x - 1), BR);
				ir = memloc(halowidth, blkmemwidth, jj, (blockDim.x - 2), BR);
				it = memloc(halowidth, blkmemwidth, jj + 1, (blockDim.x - 1), BR);
				itr = memloc(halowidth, blkmemwidth, jj + 1, (blockDim.x - 2), BR);
				a_read = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}
	}
	else if (levBL < lev)
	{
		jj = TLBL == ib ? ceil(ix * (T)0.5) : ceil(ix * (T)0.5) + blockDim.x / 2;
		w1 = 1.0 / 3.0;
		w2 = ceil(ix * (T)0.5) * 2 > ix ? T(1.0 / 6.0) : T(0.5);
		w3 = ceil(ix * (T)0.5) * 2 > ix ? T(0.5) : T(1.0 / 6.0);

		ii = memloc(halowidth, blkmemwidth, ix, 0, ib);
		ir = memloc(halowidth, blkmemwidth, jj, blockDim.x - 1, BL);
		it = memloc(halowidth, blkmemwidth, jj - 1, blockDim.x - 1, BL);

		if (TLBL == ib)
		{
			if (ix == 0)
			{
				if (LTBL == BL)
				{
					w3 = 0.5 * (1.0 - w1);
					w2 = w3;
					it = ir;
				}
				else if (levLTBL < levBL)
				{
					w1 = 4.0 / 10.0;
					w2 = 5.0 / 10.0;
					w3 = 1.0 / 10.0;
					it = memloc(halowidth, blkmemwidth, blockDim.x - 1, blockDim.x - 1, LTBL);
				}
				else if (levLTBL == levBL)
				{
					it = memloc(halowidth, blkmemwidth, blockDim.x - 1, blockDim.x - 1, LTBL);
				}
				else if (levLTBL > levBL)
				{
					w1 = 1.0 / 4.0;
					w2 = 1.0 / 2.0;
					w3 = 1.0 / 4.0;
					it = memloc(halowidth, blkmemwidth, blockDim.x - 1, blockDim.x - 1, LTBL);
				}
			}
		}
		else
		{
			if (ix == (blockDim.x - 1))
			{
				if (RTBR == BR)
				{
					w3 = 0.5 * (1.0 - w1);
					w2 = w3;
					ir = it;
				}
				else if (levRTBR < levBR)
				{
					w1 = 4.0 / 10.0;
					w2 = 1.0 / 10.0;
					w3 = 5.0 / 10.0;
					ir = memloc(halowidth, blkmemwidth,0, blockDim.x - 1, RTBR);
				}
				else if (levRTBR == levBR)
				{
					ir = memloc(halowidth, blkmemwidth,0, blockDim.x - 1, RTBR);
				}
				else if (levRTBR > levBR)
				{
					w1 = 1.0 / 4.0;
					w2 = 1.0 / 2.0;
					w3 = 1.0 / 4.0;
					ir = memloc(halowidth, blkmemwidth, 0, blockDim.x - 1, RTBR);
				}
			}
		}
		a_read = w1 * a[ii] + w2 * a[ir] + w3 * a[it];
	}
	a[write] = a_read;

}
template __global__ void fillBot<float>(int halowidth, int* active, int* level, int* botleft, int* botright, int* topleft, int* lefttop, int* righttop, float* a);
template __global__ void fillBot<double>(int halowidth, int* active, int* level, int* botleft, int* botright, int* topleft, int* lefttop, int* righttop, double* a);

template <class T> void fillBotFlux(Param XParam, bool doProlongation, int ib, BlockP<T> XBlock, T*& z)
{
	int jj, bb;
	int read, write;
	int ii, ir, it, itr;


	if (XBlock.BotLeft[ib] == ib)//The lower half is a boundary 
	{
		for (int j = 0; j < (XParam.blkwidth / 2); j++)
		{

			read = memloc(XParam, j, 0, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			write = memloc(XParam, j, -1, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			z[write] = z[read];
		}

		if (XBlock.BotRight[ib] == ib) // boundary on the top half too
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, j, 0, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, -1, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				write = memloc(XParam, j, -1, ib);
				jj = (j - 8) * 2;
				ii = memloc(XParam, jj, (XParam.blkwidth - 1), XBlock.BotRight[ib]);
				//ir = memloc(XParam, jj, (XParam.blkwidth - 2), XBlock.BotRight[ib]);
				it = memloc(XParam, jj + 1, (XParam.blkwidth - 1), XBlock.BotRight[ib]);
				//itr = memloc(XParam, jj + 1, (XParam.blkwidth - 2), XBlock.BotRight[ib]);

				z[write] = T(0.5) * (z[ii] + z[it] );

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.BotLeft[ib]]) // LeftTop block does not exist
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//

			write = memloc(XParam, j, -1, ib);
			read = memloc(XParam, j, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
			z[write] = z[read];
		}
	}
	else if (XBlock.level[XBlock.BotLeft[ib]] > XBlock.level[ib])
	{

		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{

			write = memloc(XParam, j, -1, ib);

			jj = j * 2;
			bb = XBlock.BotLeft[ib];

			ii = memloc(XParam, jj, (XParam.blkwidth - 1), bb);
			//ir = memloc(XParam, jj, (XParam.blkwidth - 2), bb);
			it = memloc(XParam, jj + 1, (XParam.blkwidth - 1), bb);
			//itr = memloc(XParam, jj + 1, (XParam.blkwidth - 2), bb);

			z[write] = T(0.5) * (z[ii] + z[it]);
		}
		//now find out aboy botright block
		if (XBlock.BotRight[ib] == ib)
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, j, 0, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, -1, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//
				jj = (j - 8) * 2;
				bb = XBlock.BotRight[ib];

				//read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, -1, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				//z[write] = z[read];
				ii = memloc(XParam, jj, (XParam.blkwidth - 1), bb);
				//ir = memloc(XParam, jj, (XParam.blkwidth - 2), bb);
				it = memloc(XParam, jj + 1, (XParam.blkwidth - 1), bb);
				//itr = memloc(XParam, jj + 1, (XParam.blkwidth - 2), bb);

				z[write] = T(0.5) * (z[ii]  + z[it] );
			}
		}

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			write = memloc(XParam, j, -1, ib);

			T w1, w2, w3;


			int jj = XBlock.TopLeft[XBlock.BotLeft[ib]] == ib ? ceil(j * (T)0.5) : ceil(j * (T)0.5) + XParam.blkwidth / 2;
			

			//ii = memloc(XParam, j, 0, ib);
			ir = memloc(XParam, jj, XParam.blkwidth - 1, XBlock.BotLeft[ib]);
			if(doProlongation)
				z[write] = z[ir];
		}
	}



}

template <class T> void fillTop(Param XParam, int ib, BlockP<T> XBlock, T*& z)
{
	int jj, bb;
	int read, write;
	int ii, ir, it, itr;


	if (XBlock.TopLeft[ib] == ib)//The lower half is a boundary 
	{
		for (int j = 0; j < (XParam.blkwidth / 2); j++)
		{

			read = memloc(XParam,j, XParam.blkwidth - 1, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			write = memloc(XParam,j, XParam.blkwidth, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			z[write] = z[read];
		}

		if (XBlock.TopRight[ib] == ib) // boundary on the top half too
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, j, XParam.blkwidth - 1, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, XParam.blkwidth, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				write = memloc(XParam, j, XParam.blkwidth, ib);
				jj = (j - (XParam.blkwidth / 2)) * 2;
				ii = memloc(XParam, jj, 0, XBlock.TopRight[ib]);
				ir = memloc(XParam, jj, 1, XBlock.TopRight[ib]);
				it = memloc(XParam, jj + 1, 0, XBlock.TopRight[ib]);
				itr = memloc(XParam, jj + 1, 1, XBlock.TopRight[ib]);

				z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.TopLeft[ib]]) // LeftTop block does not exist
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//

			write = memloc(XParam, j, XParam.blkwidth, ib);
			read = memloc(XParam, j, 0, XBlock.TopLeft[ib]);
			z[write] = z[read];
		}
	}
	else if (XBlock.level[XBlock.TopLeft[ib]] > XBlock.level[ib])
	{

		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{

			write = memloc(XParam, j, XParam.blkwidth, ib);

			jj = j * 2;
			bb = XBlock.TopLeft[ib];

			ii = memloc(XParam,jj, 0, bb);
			ir = memloc(XParam,jj, 1, bb);
			it = memloc(XParam,jj + 1, 0, bb);
			itr = memloc(XParam,jj + 1, 1, bb);

			z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.TopRight[ib] == ib)
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam,j, XParam.blkwidth - 1, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, XParam.blkwidth, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//
				jj = (j - (XParam.blkwidth / 2)) * 2;
				bb = XBlock.TopRight[ib];

				//read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j , XParam.blkwidth, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				//z[write] = z[read];
				ii = memloc(XParam,jj, 0, bb);
				ir = memloc(XParam,jj, 1, bb);
				it = memloc(XParam,jj + 1, 0, bb);
				itr = memloc(XParam,jj + 1, 1, bb);

				z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.TopLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			write = memloc(XParam,j, XParam.blkwidth, ib);

			T w1, w2, w3;
			

			int jj = XBlock.BotLeft[XBlock.TopLeft[ib]] == ib ? ceil(j * (T)0.5) : ceil(j * (T)0.5) + XParam.blkwidth / 2;
			w1 = 1.0 / 3.0;
			w2 = ceil(j * (T)0.5) * 2 > j ? T(1.0 / 6.0) : T(0.5);
			w3 = ceil(j * (T)0.5) * 2 > j ? T(0.5) : T(1.0 / 6.0);

			ii = memloc(XParam,j, XParam.blkwidth - 1, ib);
			ir = memloc(XParam,jj, 0,  XBlock.TopLeft[ib]);
			it = memloc(XParam,jj-1, 0, XBlock.TopLeft[ib]);
			//2 scenarios here ib is the leftbot neighbour of the rightbot block or ib is the lefttop neighbour
			if (XBlock.BotLeft[XBlock.TopLeft[ib]] == ib)
			{
				if (j == 0)
				{
					if (XBlock.LeftBot[XBlock.TopLeft[ib]] == XBlock.TopLeft[ib]) // no botom of leftbot block
					{
						w3 = 0.5 * (1.0 - w1);
						w2 = w3;
						it = ir;

					}
					else if (XBlock.level[XBlock.LeftBot[XBlock.TopLeft[ib]]] < XBlock.level[XBlock.TopLeft[ib]]) // exists but is coarser
					{
						w1 = 4.0 / 10.0;
						w2 = 5.0 / 10.0;
						w3 = 1.0 / 10.0;
						it = memloc(XParam, XParam.blkwidth - 1,0, XBlock.LeftBot[XBlock.TopLeft[ib]]);
					}
					else if (XBlock.level[XBlock.LeftBot[XBlock.TopLeft[ib]]] == XBlock.level[XBlock.TopLeft[ib]]) // exists with same level
					{
						it = memloc(XParam,  XParam.blkwidth - 1,0, XBlock.LeftBot[XBlock.TopLeft[ib]]);
					}
					else if (XBlock.level[XBlock.LeftBot[XBlock.TopLeft[ib]]] > XBlock.level[XBlock.TopLeft[ib]]) // exists with higher level
					{
						w1 = 1.0 / 4.0;
						w2 = 1.0 / 2.0;
						w3 = 1.0 / 4.0;
						it = memloc(XParam, XParam.blkwidth - 1, 0, XBlock.LeftBot[XBlock.TopLeft[ib]]);
					}


				}


			}
			else//
			{
				if (j == (XParam.blkwidth - 1))
				{
					if (XBlock.RightBot[XBlock.TopRight[ib]] == XBlock.TopRight[ib]) // no botom of leftbot block
					{
						w3 = 0.5 * (1.0 - w1);
						w2 = w3;
						ir = it;

					}
					else if (XBlock.level[XBlock.RightBot[XBlock.TopRight[ib]]] < XBlock.level[XBlock.TopRight[ib]]) // exists but is coarser
					{
						w1 = 4.0 / 10.0;
						w2 = 1.0 / 10.0;
						w3 = 5.0 / 10.0;
						ir = memloc(XParam, 0, 0, XBlock.RightBot[XBlock.TopRight[ib]]);
					}
					else if (XBlock.level[XBlock.RightBot[XBlock.TopRight[ib]]] == XBlock.level[XBlock.TopRight[ib]]) // exists with same level
					{
						ir = memloc(XParam, 0, 0, XBlock.RightBot[XBlock.TopRight[ib]]);
					}
					else if (XBlock.level[XBlock.RightBot[XBlock.TopRight[ib]]] > XBlock.level[XBlock.TopRight[ib]]) // exists with higher level
					{
						w1 = 1.0 / 4.0;
						w2 = 1.0 / 2.0;
						w3 = 1.0 / 4.0;
						ir = memloc(XParam, 0,0, XBlock.RightBot[XBlock.TopRight[ib]]);
					}
				}
				//
			}


			z[write] = w1 * z[ii] + w2 * z[ir] + w3 * z[it];
		}
	}



}

template <class T> __global__ void fillTop(int halowidth, int* active, int* level,int * topleft, int * topright,int * botleft, int* leftbot, int* rightbot,  T* a)
{
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = blockDim.x-1;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = active[ibl];

	int TL = topleft[ib];
	int TR = topright[ib];
	int LBTL = leftbot[TL];
	int BLTL = botleft[TL];
	int RBTR = rightbot[TR];


	int lev = level[ib];
	int levTL = level[TL];
	int levTR = level[TR];
	int levLBTL = level[LBTL];
	int levRBTR = level[RBTR];

	int write = memloc(halowidth, blkmemwidth, ix, blockDim.x, ib);
	int read;
	int jj, ii, ir, it, itr;
	T a_read;
	T w1, w2, w3;

	if (TL == ib)
	{
		if (ix < (blockDim.x / 2))
		{
			read = memloc(halowidth, blkmemwidth, ix, blockDim.x - 1, ib);
			a_read = a[read];
		}
		else
		{
			if (TR == ib)
			{
				read = memloc(halowidth, blkmemwidth, ix, blockDim.x - 1, ib);
				a_read = a[read];
			}
			else
			{
				jj = (ix - (blockDim.x / 2)) * 2;
				ii = memloc(halowidth, blkmemwidth, jj, 0, TR);
				ir = memloc(halowidth, blkmemwidth, jj, 1, TR);
				it = memloc(halowidth, blkmemwidth, jj + 1, 0, TR);
				itr = memloc(halowidth, blkmemwidth, jj + 1, 1, TR);

				a_read = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}
	}
	else if (levTL == lev)
	{
		read = memloc(halowidth, blkmemwidth, ix, 0, TL);
		a_read = a[read];
	}
	else if (levTL > lev)
	{
		if (ix < (blockDim.x / 2))
		{
			jj = ix * 2;
			

			ii = memloc(halowidth, blkmemwidth, jj, 0, TL);
			ir = memloc(halowidth, blkmemwidth, jj, 1, TL);
			it = memloc(halowidth, blkmemwidth, jj + 1, 0, TL);
			itr = memloc(halowidth, blkmemwidth, jj + 1, 1, TL);
			a_read = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		else
		{
			if (TR == ib)
			{
				read = memloc(halowidth, blkmemwidth, ix, blockDim.x-1, ib);
				a_read = a[read];
			}
			else
			{
				jj = (ix - (blockDim.x / 2)) * 2;
				ii = memloc(halowidth, blkmemwidth, jj, 0, TR);
				ir = memloc(halowidth, blkmemwidth, jj, 1, TR);
				it = memloc(halowidth, blkmemwidth, jj + 1, 0, TR);
				itr = memloc(halowidth, blkmemwidth, jj + 1, 1, TR);
				a_read = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}
	}
	else if (levTL < lev)
	{
		jj = BLTL == ib ? ceil(ix * (T)0.5) : ceil(ix * (T)0.5) + blockDim.x / 2;
		w1 = 1.0 / 3.0;
		w2 = ceil(ix * (T)0.5) * 2 > ix ? T(1.0 / 6.0) : T(0.5);
		w3 = ceil(ix * (T)0.5) * 2 > ix ? T(0.5) : T(1.0 / 6.0);
		ii = memloc(halowidth, blkmemwidth, ix, blockDim.x - 1, ib);
		ir = memloc(halowidth, blkmemwidth, jj, 0, TL);
		it = memloc(halowidth, blkmemwidth, jj - 1, 0, TL);
		if (BLTL == ib)
		{
			if (ix == 0)
			{
				if (LBTL == TL)
				{
					w3 = 0.5 * (1.0 - w1);
					w2 = w3;
					it = ir;
				}
				else if (levLBTL < levTL)
				{
					w1 = 4.0 / 10.0;
					w2 = 5.0 / 10.0;
					w3 = 1.0 / 10.0;
					it = memloc(halowidth, blkmemwidth, blockDim.x - 1, 0, LBTL);
				}
				else if (levLBTL == levTL)
				{
					it = memloc(halowidth, blkmemwidth, blockDim.x - 1, 0, LBTL);
				}
				else if (levLBTL > levTL)
				{
					w1 = 1.0 / 4.0;
					w2 = 1.0 / 2.0;
					w3 = 1.0 / 4.0;
					it = memloc(halowidth, blkmemwidth, blockDim.x - 1, 0, LBTL);
				}
			}
		}
		else
		{
			if (ix == blockDim.x - 1)
			{
				if (RBTR == TR)
				{
					w3 = 0.5 * (1.0 - w1);
					w2 = w3;
					ir = it;
				}
				else if (levRBTR < levTR)
				{
					w1 = 4.0 / 10.0;
					w2 = 1.0 / 10.0;
					w3 = 5.0 / 10.0;
					ir = memloc(halowidth, blkmemwidth, 0, 0, RBTR);
				}
				else if (levRBTR == levTR)
				{
					ir = memloc(halowidth, blkmemwidth, 0, 0, RBTR);
				}
				else if (levRBTR > levTR)
				{
					w1 = 1.0 / 4.0;
					w2 = 1.0 / 2.0;
					w3 = 1.0 / 4.0;
					ir = memloc(halowidth, blkmemwidth,0, 0, RBTR);
				}
			}
		}
		a_read= w1 * a[ii] + w2 * a[ir] + w3 * a[it];
	}
	a[write] = a_read;
}

template __global__ void fillTop<float>(int halowidth, int* active, int* level, int* topleft, int* topright, int* botleft, int* leftbot, int* rightbot, float* a);
template __global__ void fillTop<double>(int halowidth, int* active, int* level, int* topleft, int* topright, int* botleft, int* leftbot, int* rightbot, double* a);

template <class T> void fillTopFlux(Param XParam, bool doProlongation, int ib, BlockP<T> XBlock, T*& z)
{
	int jj, bb;
	int read, write;
	int ii, ir, it, itr;


	if (XBlock.TopLeft[ib] == ib)//The lower half is a boundary 
	{
		for (int j = 0; j < (XParam.blkwidth / 2); j++)
		{

			read = memloc(XParam, j, XParam.blkwidth - 1, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			write = memloc(XParam, j, XParam.blkwidth, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			z[write] = z[read];
		}

		if (XBlock.TopRight[ib] == ib) // boundary on the top half too
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, j, XParam.blkwidth - 1, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, XParam.blkwidth, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				write = memloc(XParam, j, XParam.blkwidth, ib);
				jj = (j - (XParam.blkwidth / 2)) * 2;
				ii = memloc(XParam, jj, 0, XBlock.TopRight[ib]);
				//ir = memloc(XParam, jj, 1, XBlock.TopRight[ib]);
				it = memloc(XParam, jj + 1, 0, XBlock.TopRight[ib]);
				//itr = memloc(XParam, jj + 1, 1, XBlock.TopRight[ib]);

				z[write] = T(0.5) * (z[ii] + z[it] );

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.TopLeft[ib]]) // LeftTop block does not exist
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//

			write = memloc(XParam, j, XParam.blkwidth, ib);
			read = memloc(XParam, j, 0, XBlock.TopLeft[ib]);
			z[write] = z[read];
		}
	}
	else if (XBlock.level[XBlock.TopLeft[ib]] > XBlock.level[ib])
	{

		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{

			write = memloc(XParam, j, XParam.blkwidth, ib);

			jj = j * 2;
			bb = XBlock.TopLeft[ib];

			ii = memloc(XParam, jj, 0, bb);
			//ir = memloc(XParam, jj, 1, bb);
			it = memloc(XParam, jj + 1, 0, bb);
			//itr = memloc(XParam, jj + 1, 1, bb);

			z[write] = T(0.5) * (z[ii]  + z[it]);
		}
		//now find out aboy lefttop block
		if (XBlock.TopRight[ib] == ib)
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, j, XParam.blkwidth - 1, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, XParam.blkwidth, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//
				jj = (j - (XParam.blkwidth / 2)) * 2;
				bb = XBlock.TopRight[ib];

				//read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, j, XParam.blkwidth, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				//z[write] = z[read];
				ii = memloc(XParam, jj, 0, bb);
				//ir = memloc(XParam, jj, 1, bb);
				it = memloc(XParam, jj + 1, 0, bb);
				//itr = memloc(XParam, jj + 1, 1, bb);

				z[write] = T(0.5) * (z[ii]  + z[it]);
			}
		}

	}
	else if (XBlock.level[XBlock.TopLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			write = memloc(XParam, j, XParam.blkwidth, ib);
			int jj = XBlock.BotLeft[XBlock.TopLeft[ib]] == ib ? floor(j * (T)0.5) : floor(j * (T)0.5) + XParam.blkwidth / 2;
						
			ir = memloc(XParam, jj, 0, XBlock.TopLeft[ib]);
			
			if (doProlongation)
				z[write] = z[ir];

		}
	}



}
template void fillTopFlux<float>(Param XParam, bool doProlongation, int ib, BlockP<float> XBlock, float*& z);
template void fillTopFlux<double>(Param XParam, bool doProlongation, int ib, BlockP<double> XBlock, double*& z);

template <class T> __global__ void fillTopFlux(int halowidth, bool doProlongation, int* active, int* level, int* topleft, int* topright, int* botleft, int* leftbot, int* rightbot, T* a)
{
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = blockDim.x - 1;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = active[ibl];

	int TL = topleft[ib];
	int TR = topright[ib];
	int LBTL = leftbot[TL];
	int BLTL = botleft[TL];
	int RBTR = rightbot[TR];


	int lev = level[ib];
	int levTL = level[TL];
	int levTR = level[TR];
	int levLBTL = level[LBTL];
	int levRBTR = level[RBTR];

	int write = memloc(halowidth, blkmemwidth, ix, blockDim.x, ib);
	int read;
	int jj, ii, ir, it, itr;
	T a_read;
	T w1, w2, w3;

	if (TL == ib)
	{
		if (ix < (blockDim.x / 2))
		{
			read = memloc(halowidth, blkmemwidth, ix, blockDim.x - 1, ib);
			a_read = a[read];
		}
		else
		{
			if (TR == ib)
			{
				read = memloc(halowidth, blkmemwidth, ix, blockDim.x - 1, ib);
				a_read = a[read];
			}
			else
			{
				jj = (ix - (blockDim.x / 2)) * 2;
				ii = memloc(halowidth, blkmemwidth, jj, 0, TR);
				//ir = memloc(halowidth, blkmemwidth, jj, 1, TR);
				it = memloc(halowidth, blkmemwidth, jj + 1, 0, TR);
				//itr = memloc(halowidth, blkmemwidth, jj + 1, 1, TR);

				a_read = T(0.5) * (a[ii]  + a[it] );
			}
		}
	}
	else if (levTL == lev)
	{
		read = memloc(halowidth, blkmemwidth, ix, 0, TL);
		a_read = a[read];
	}
	else if (levTL > lev)
	{
		if (ix < (blockDim.x / 2))
		{
			jj = ix * 2;


			ii = memloc(halowidth, blkmemwidth, jj, 0, TL);
			//ir = memloc(halowidth, blkmemwidth, jj, 1, TL);
			it = memloc(halowidth, blkmemwidth, jj + 1, 0, TL);
			//itr = memloc(halowidth, blkmemwidth, jj + 1, 1, TL);
			a_read = T(0.5) * (a[ii] + a[it]);
		}
		else
		{
			if (TR == ib)
			{
				read = memloc(halowidth, blkmemwidth, ix, blockDim.x - 1, ib);
				a_read = a[read];
			}
			else
			{
				jj = (ix - (blockDim.x / 2)) * 2;
				ii = memloc(halowidth, blkmemwidth, jj, 0, TR);
				//ir = memloc(halowidth, blkmemwidth, jj, 1, TR);
				it = memloc(halowidth, blkmemwidth, jj + 1, 0, TR);
				//itr = memloc(halowidth, blkmemwidth, jj + 1, 1, TR);
				a_read = T(0.5) * (a[ii] + a[it]);
			}
		}
	}
	else if (levTL < lev)
	{
		jj = BLTL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + blockDim.x / 2;
		
		ir = memloc(halowidth, blkmemwidth, jj, 0, TL);
		if (doProlongation)
			a_read = a[ir];
		else
			a_read =  a[write];
	}
	a[write] = a_read;
}

template __global__ void fillTopFlux<float>(int halowidth, bool doProlongation, int* active, int* level, int* topleft, int* topright, int* botleft, int* leftbot, int* rightbot, float* a);
template __global__ void fillTopFlux<double>(int halowidth, bool doProlongation, int* active, int* level, int* topleft, int* topright, int* botleft, int* leftbot, int* rightbot, double* a);



template <class T> void fillCorners(Param XParam, BlockP<T> XBlock, T*& z)
{
	int ib;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		fillCorners(XParam, ib, XBlock, z);
		
	}

}
template void fillCorners<float>(Param XParam, BlockP<float> XBlock, float*& z);
template void fillCorners<double>(Param XParam, BlockP<double> XBlock, double*& z);


template <class T> void fillCorners(Param XParam, BlockP<T> XBlock, EvolvingP<T>& Xev)
{
	int ib;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		fillCorners(XParam, ib, XBlock, Xev.h);
		fillCorners(XParam, ib, XBlock, Xev.zs);
		fillCorners(XParam, ib, XBlock, Xev.u);
		fillCorners(XParam, ib, XBlock, Xev.v);
	}

}
template void fillCorners<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float>& Xev);
template void fillCorners<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double>& Xev);



template <class T> void fillCorners(Param XParam, int ib, BlockP<T> XBlock, T*& z)
{
	// Run only this function after the filling the other bit of halo (i.e. fctn fillleft...)
	// Most of the time the cormers are not needed. they are when refining a cell! 

	
	T zz;
	int write;
	int ii, ir, it, itr;


	// Bottom left corner
	write = memloc(XParam, -1, -1, ib);
	//check that there is a block there and if there is calculate the value depending on the level of that block
	if (XBlock.LeftTop[XBlock.BotLeft[ib]] == XBlock.BotLeft[ib]) // There is no block
	{
		zz = T(0.5) * (z[memloc(XParam, -1, 0, ib)] + z[memloc(XParam, 0, -1, ib)]);
	}
	else if (XBlock.level[XBlock.LeftTop[XBlock.BotLeft[ib]]] == XBlock.level[ib])
	{
		zz = z[memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.LeftTop[XBlock.BotLeft[ib]])];
	}
	else if (XBlock.level[XBlock.LeftTop[XBlock.BotLeft[ib]]] > XBlock.level[ib])
	{
		ii = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.LeftTop[XBlock.BotLeft[ib]]);
		ir = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 2, XBlock.LeftTop[XBlock.BotLeft[ib]]);
		it = memloc(XParam, XParam.blkwidth - 2, XParam.blkwidth - 1, XBlock.LeftTop[XBlock.BotLeft[ib]]);
		itr = memloc(XParam, XParam.blkwidth - 2, XParam.blkwidth - 2, XBlock.LeftTop[XBlock.BotLeft[ib]]);

		zz = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
	}
	else if (XBlock.level[XBlock.LeftTop[XBlock.BotLeft[ib]]] < XBlock.level[ib])
	{
		ii = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.LeftTop[XBlock.BotLeft[ib]]);
		ir = memloc(XParam, - 1, 0, ib);
		it = memloc(XParam,0, - 1, ib);
		zz = T(0.5) * z[ii] + T(0.25) * (z[ir] + z[it]);
	}

	z[write] = zz;

	// Top Left corner
	write = memloc(XParam, -1, XParam.blkwidth, ib);
	//check that there is a block there and if there is calculate the value depending on the level of that block
	if (XBlock.LeftBot[XBlock.TopLeft[ib]] == XBlock.TopLeft[ib]) // There is no block
	{
		zz = T(0.5) * (z[memloc(XParam, -1, XParam.blkwidth-1, ib)] + z[memloc(XParam, 0, XParam.blkwidth, ib)]);
	}
	else if (XBlock.level[XBlock.LeftBot[XBlock.TopLeft[ib]]] == XBlock.level[ib])
	{
		zz = z[memloc(XParam, XParam.blkwidth - 1, 0, XBlock.LeftBot[XBlock.TopLeft[ib]])];
	}
	else if (XBlock.level[XBlock.LeftBot[XBlock.TopLeft[ib]]] > XBlock.level[ib])
	{
		ii = memloc(XParam, XParam.blkwidth - 1, 0, XBlock.LeftBot[XBlock.TopLeft[ib]]);
		ir = memloc(XParam, XParam.blkwidth - 1, 1, XBlock.LeftBot[XBlock.TopLeft[ib]]);
		it = memloc(XParam, XParam.blkwidth - 2, 0, XBlock.LeftBot[XBlock.TopLeft[ib]]);
		itr = memloc(XParam, XParam.blkwidth - 2, 1, XBlock.LeftBot[XBlock.TopLeft[ib]]);

		zz = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
	}
	else if (XBlock.level[XBlock.LeftBot[XBlock.TopLeft[ib]]] < XBlock.level[ib])
	{
		ii = memloc(XParam, XParam.blkwidth - 1, 0, XBlock.LeftBot[XBlock.TopLeft[ib]]);
		ir = memloc(XParam, -1, XParam.blkwidth - 1, ib);
		it = memloc(XParam, 0, XParam.blkwidth, ib);
		zz = T(0.5) * z[ii] + T(0.25) * (z[ir] + z[it]);
	}

	z[write] = zz;

	//Top Right corner
	write = memloc(XParam, XParam.blkwidth, XParam.blkwidth, ib);
	//check that there is a block there and if there is calculate the value depending on the level of that block
	if (XBlock.RightBot[XBlock.TopRight[ib]] == XBlock.TopRight[ib]) // There is no block
	{
		zz = T(0.5) * (z[memloc(XParam, XParam.blkwidth, XParam.blkwidth - 1, ib)] + z[memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth, ib)]);
	}
	else if (XBlock.level[XBlock.RightBot[XBlock.TopRight[ib]]] == XBlock.level[ib])
	{
		zz = z[memloc(XParam, 0, 0, XBlock.RightBot[XBlock.TopRight[ib]])];
	}
	else if (XBlock.level[XBlock.RightBot[XBlock.TopRight[ib]]] > XBlock.level[ib])
	{
		ii = memloc(XParam, 0, 0, XBlock.RightBot[XBlock.TopRight[ib]]);
		ir = memloc(XParam, 0, 1, XBlock.RightBot[XBlock.TopRight[ib]]);
		it = memloc(XParam, 1, 0, XBlock.RightBot[XBlock.TopRight[ib]]);
		itr = memloc(XParam, 1, 1, XBlock.RightBot[XBlock.TopRight[ib]]);

		zz = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
	}
	else if (XBlock.level[XBlock.LeftBot[XBlock.TopLeft[ib]]] < XBlock.level[ib])
	{
		ii = memloc(XParam, 0, 0, XBlock.LeftBot[XBlock.TopLeft[ib]]);
		ir = memloc(XParam, XParam.blkwidth, XParam.blkwidth - 1, ib);
		it = memloc(XParam, XParam.blkwidth-1, XParam.blkwidth, ib);
		zz = T(0.5) * z[ii] + T(0.25) * ( z[ir] +  z[it]);
	}

	z[write] = zz;

	//Bot Right corner
	write = memloc(XParam, XParam.blkwidth, -1, ib);
	//check that there is a block there and if there is calculate the value depending on the level of that block
	if (XBlock.RightBot[XBlock.BotRight[ib]] == XBlock.BotRight[ib]) // There is no block
	{
		zz = T(0.5) * (z[memloc(XParam, XParam.blkwidth-1, - 1, ib)] + z[memloc(XParam, XParam.blkwidth , 0, ib)]);
	}
	else if (XBlock.level[XBlock.RightBot[XBlock.BotRight[ib]]] == XBlock.level[ib])
	{
		zz = z[memloc(XParam, 0, XParam.blkwidth - 1, XBlock.RightBot[XBlock.BotRight[ib]])];
	}
	else if (XBlock.level[XBlock.RightBot[XBlock.BotRight[ib]]] > XBlock.level[ib])
	{
		ii = memloc(XParam, 0, XParam.blkwidth - 1, XBlock.RightBot[XBlock.BotRight[ib]]);
		ir = memloc(XParam, 0, XParam.blkwidth - 2, XBlock.RightBot[XBlock.BotRight[ib]]);
		it = memloc(XParam, 1, XParam.blkwidth - 1, XBlock.RightBot[XBlock.BotRight[ib]]);
		itr = memloc(XParam, 1, XParam.blkwidth - 2, XBlock.RightBot[XBlock.BotRight[ib]]);

		zz = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
	}
	else if (XBlock.level[XBlock.RightBot[XBlock.BotRight[ib]]] < XBlock.level[ib])
	{
		ii = memloc(XParam, 0, XParam.blkwidth - 1, XBlock.LeftBot[XBlock.TopLeft[ib]]);
		ir = memloc(XParam, XParam.blkwidth - 1, -1, ib);
		it = memloc(XParam, XParam.blkwidth, 0, ib);
		zz = T(0.5) * z[ii] + T(0.25) * (z[ir] + z[it]);
	}

	z[write] = zz;

}
template void fillCorners<float>(Param XParam, int ib, BlockP<float> XBlock, float*& z);
template void fillCorners<double>(Param XParam, int ib, BlockP<double> XBlock, double*& z);

