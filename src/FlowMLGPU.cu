#include "FlowMLGPU.h"

template <class T> void FlowMLGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel)
{
	//============================================
	// construct threads abnd block parameters
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	// for flux reconstruction the loop overlap the right(or top for the y direction) halo
	dim3 blockDimKX(XParam.blkwidth + XParam.halowidth, XParam.blkwidth, 1);
	dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + XParam.halowidth, 1);

	// For halo corners
	dim3 blockDimHC(4, 1, 1);

	// Fill halo for Fu and Fv
	dim3 blockDimHaloLR(1, XParam.blkwidth, 1);
	//dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDimHaloLR(XParam.nblk, 1, 1);

	dim3 blockDimHaloBT(XParam.blkwidth, 1, 1);
	dim3 gridDimHaloBT(XParam.nblk , 1, 1);

	if (XParam.atmpforcing)
	{
		//Update atm press forcing
		AddPatmforcingGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XForcing.Atmp, XModel);
		CUDA_CHECK(cudaDeviceSynchronize());

		//Fill atmp halo
		cudaStream_t atmpstreams[1];
		CUDA_CHECK(cudaStreamCreate(&atmpstreams[0]));
		fillHaloGPU(XParam, XModel.blocks, atmpstreams[0], XModel.Patm);
		CUDA_CHECK(cudaDeviceSynchronize());
		cudaStreamDestroy(atmpstreams[0]);
	}


	// fill halo for zs,h,u and v 

	//============================================
	//  Fill the halo for gradient reconstruction & Recalculate zs
	fillHaloGPU(XParam, XModel.blocks, XModel.evolv, XModel.zb);
	
	
	//CUDA_CHECK(cudaMemcpy(XModel.evolv_o.h, XModel.evolv.h, XParam.nblk * XParam.blksize * sizeof(T), cudaMemcpyDeviceToDevice));
	//============================================
	// Calculate gradient for evolving parameters for predictor step
	gradientGPUML(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb);
	//gradientSMC << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XModel.evolv.h, XModel.grad.dhdx, XModel.grad.dhdy);
	//CUDA_CHECK(cudaDeviceSynchronize());


	//============================================
	// Synchronise all ongoing streams
	CUDA_CHECK(cudaDeviceSynchronize());

	// Set max timestep
	reset_var <<< gridDim, blockDim, 0 >>> (XParam.halowidth, XModel.blocks.active, XLoop.hugeposval, XModel.time.dtmax);
	CUDA_CHECK(cudaDeviceSynchronize());

	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, T(0.0), XModel.adv.dh);
	CUDA_CHECK(cudaDeviceSynchronize());

	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, T(0.0), XModel.adv.dhu);
	CUDA_CHECK(cudaDeviceSynchronize());

	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, T(0.0), XModel.adv.dhv);

	CUDA_CHECK(cudaDeviceSynchronize());


	//CUDA_CHECK(cudaMemcpy(XModel.evolv_o.zs, XModel.evolv.zs, XParam.nblk * XParam.blksize * sizeof(T) , cudaMemcpyDeviceToDevice));
	
	//CUDA_CHECK(cudaMemcpy(XModel.evolv_o.u, XModel.evolv.u, XParam.nblk * XParam.blksize * sizeof(T), cudaMemcpyDeviceToDevice));
	//CUDA_CHECK(cudaMemcpy(XModel.evolv_o.v, XModel.evolv.v, XParam.nblk * XParam.blksize * sizeof(T), cudaMemcpyDeviceToDevice));

	// Compute face value
	CalcfaceValX << < gridDim, blockDim, 0 >> > (T(XLoop.dtmax), XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.fluxml, XModel.time.dtmax, XModel.zb, XModel.Patm);
	CUDA_CHECK(cudaDeviceSynchronize());

	CalcfaceValY << < gridDim, blockDim, 0 >> > (T(XLoop.dtmax), XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.fluxml, XModel.time.dtmax, XModel.zb, XModel.Patm);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Timestep reduction
	XLoop.dt = double(CalctimestepGPU(XParam, XLoop, XModel.blocks, XModel.time));
	XLoop.dtmax = XLoop.dt;

	

	//Fill flux Halo for ha and hf
	/*fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hfu);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hfv);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hau);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hav);

	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hu);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hv);*/

	HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfu);
	CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfu);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfv);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfv);
	CUDA_CHECK(cudaDeviceSynchronize());


	
	//HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfu);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUBMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfu);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfv);
	CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUBMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfv);
	//CUDA_CHECK(cudaDeviceSynchronize());
	




	HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hau);
	CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hau);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hav);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hav);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Half advection if implicit
	if (XParam.implicit)//&& (theta_H < 1.)
	{
		//
		int n = XParam.blksize * XParam.nblk;

		if(XParam.theta_H < 1.0)
		{
			T dt_thetaH = T((1 - XParam.theta_H) * XLoop.dt);
			AdvecML( XParam, XLoop, XForcing, XModel, dt_thetaH);
		}

		cudaMemcpy(XModel.fluximp.eta_r, XModel.evolv.zs, n * sizeof(T), cudaMemcpyDeviceToDevice);

		//cudaMemcpy(XModel.fluximp.r, XModel.evolv.h, n * sizeof(T), cudaMemcpyDeviceToDevice);

		//assemble_alpha_kernel<<<gridDim, blockDim, 0 >>>(Param, XModel.blocks,XModel.fluxml,dt);
		acceleration_facex<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluxml, XModel.fluximp, XModel.evolv, T(XLoop.dt));
		CUDA_CHECK(cudaDeviceSynchronize());

		acceleration_facey<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluxml, XModel.fluximp, XModel.evolv, T(XLoop.dt));
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluximp.su);
		//CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluximp.sv);
		CUDA_CHECK(cudaDeviceSynchronize());

		acceleration_rhs<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp, T(XLoop.dt));
		CUDA_CHECK(cudaDeviceSynchronize());

		//test_symetry(XParam, XModel, T(XLoop.dt));

		solveEtaPCG(XParam, XModel, T(XLoop.dt));

		//cudaMemcpy(XModel.fluximp.Ap, XModel.fluxml.hav, n * sizeof(T), cudaMemcpyDeviceToDevice);
		//cudaMemcpy(XModel.fluximp.z, XModel.fluxml.hu, n * sizeof(T), cudaMemcpyDeviceToDevice);
		

		

		// Update Halo for eta_r
		HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluximp.eta_r);
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPUBMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluximp.eta_r);
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluximp.eta_r);
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluximp.eta_r);
		CUDA_CHECK(cudaDeviceSynchronize());

		//cudaMemcpy(XModel.fluximp.p, XModel.evolv.h, n * sizeof(T), cudaMemcpyDeviceToDevice);

		pressure_flux_reconstruction_facex<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluxml, XModel.fluximp, XModel.evolv, T(XLoop.dt));
		CUDA_CHECK(cudaDeviceSynchronize());
		
		pressure_flux_reconstruction_facey<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluxml, XModel.fluximp, XModel.evolv, T(XLoop.dt));
		CUDA_CHECK(cudaDeviceSynchronize());


		
		

		
		HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hau);
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hau);
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hav);
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hav);
		CUDA_CHECK(cudaDeviceSynchronize());

		//cudaMemcpy(XModel.fluximp.z, XModel.fluxml.hav, n * sizeof(T), cudaMemcpyDeviceToDevice);

		cudaMemcpy(XModel.evolv.zs, XModel.fluximp.eta_r, n * sizeof(T), cudaMemcpyDeviceToDevice);

	}

	
	// Acceleration
	// Pressure
	
	pressureML << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, T(XLoop.dt), XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());
	

	AdvecML( XParam, XLoop, XForcing, XModel, T(XLoop.dt));


	bottomfrictionGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, T(XLoop.dt), XModel.cf, XModel.evolv);
	//XiafrictionGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XModel.time.dt, XModel.cf, XModel.evolv, XModel.evolv);
	CUDA_CHECK(cudaDeviceSynchronize());

	

	if (XForcing.rivers.size() > 0)
	{
		//Add River ML
		AddRiverForcing(XParam, XLoop, XForcing.rivers, XModel);
	}

	if (XForcing.culverts.size() > 0)
	{
		AddCulverts(XParam, XLoop.dt, XForcing.culverts, XModel);
		//CUDA_CHECK(cudaDeviceSynchronize());
	}

	if (!XForcing.UWind.inputfile.empty())//&& !XForcing.UWind.inputfile.empty()
	{
		AddwindforcingGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XForcing.UWind, XForcing.VWind, XModel.adv);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	if (XForcing.rivers.size() > 0 || !XForcing.UWind.inputfile.empty())
	{
		Updatewindandriver << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, T(XLoop.dt), XModel.evolv, XModel.adv);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	if (!XForcing.Rain.inputfile.empty())
	{
		AddrainforcingImplicitGPU << < gridDim, blockDim, 0 >> > (XParam, XLoop, XModel.blocks, XForcing.Rain, XModel.evolv);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	

	if (XParam.infiltration)
	{
		AddinfiltrationImplicitGPU << < gridDim, blockDim, 0 >> > (XParam, XLoop, XModel.blocks, XModel.il, XModel.cl, XModel.evolv, XModel.hgw);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	if (XParam.VelThreshold > 0.0)
	{
		TheresholdVelGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.evolv);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	// Recalculate zs based on h and zb
	CleanupML << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.evolv, XModel.zb);
	CUDA_CHECK(cudaDeviceSynchronize());

	


}
template void FlowMLGPU<float>(Param XParam, Loop<float>& XLoop, Forcing<float> XForcing, Model<float> XModel);
template void FlowMLGPU<double>(Param XParam, Loop<double>& XLoop, Forcing<float> XForcing, Model<double> XModel);

/*
//template <class T> void FlowMLGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel)
template <class T> void solveImplicit(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel)
{
	//ImplicitCtx& ctx, double tol = 1e-5, int maxIter = 100
	double tol = XParam.mg_tol;//1e-5;
	int maxIter = XParam.max_iter;
   	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	// for flux reconstruction the loop overlap the right(or top for the y direction) halo
	dim3 blockDimKX(XParam.blkwidth + XParam.halowidth, XParam.blkwidth, 1);
	dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + XParam.halowidth, 1);

	// Fill halo for Fu and Fv
	dim3 blockDimHaloLR(1, XParam.blkwidth, 1);
	//dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDimHaloLR(XParam.nblk, 1, 1);

	dim3 blockDimHaloBT(XParam.blkwidth, 1, 1);
	dim3 gridDimHaloBT(XParam.nblk , 1, 1);

	int n = XParam.nblk * XParam.blkmemsize;// 

    // --- Assemble coefficients & RHS (once per outer timestep) ---
	//assemble_alpha_kernel(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux,T dt)
    assemble_alpha_kernel<<<gridDim, blockDim, 0 >>>(Param, XModel.blocks,XModel.fluxml,dt);
	CUDA_CHECK(cudaDeviceSynchronize());

    // Fill halo
	HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.alpha_x);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUBMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.alpha_y);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.alpha_x);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.alpha_y);
	CUDA_CHECK(cudaDeviceSynchronize());

    assemble_rhs_kernel<<<gridDim, blockDim>>>(Param, XModel.blocks,XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

    jacobi_diag_kernel<<<gridDim, blockDim>>>(Param, XModel.blocks,XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

    // --- Initial guess: warm-start from eta_n (copy already done by caller) ---
     // Fill halo
	HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.eta_n);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.eta_n);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUBMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.eta_n);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.eta_n);
	CUDA_CHECK(cudaDeviceSynchronize());

    matvec_kernel<<<gridDim, blockDim>>>(XParam, XModel.blocks, XModel.fluxml.eta_n, XModel.fluxml.Ap, XModel.fluxml);   // Ap = A(eta0)
	CUDA_CHECK(cudaDeviceSynchronize());

    // r = rhs_eta - A(eta0)   (keep rhs_eta untouched, write into ctx.r)
    cudaMemcpy(XModel.fluxml.r, XModel.fluxml.rhs_eta, n * sizeof(double), cudaMemcpyDeviceToDevice);
    axpy_kernel<<<gridDim, blockDim>>>(XParam, XModel.blocks,XModel.fluxml.r, XModel.fluxml.Ap, -1.0);   // r += -1.0 * Ap

    jacobi_apply_kernel<<<gridDim, blockDim>>>(XParam, XModel.blocks,XModel.fluxml.r, XModel.fluxml.z, XModel.fluxml.diagInv);
    cudaMemcpy(XModel.fluxml.p, XModel.fluxml.z, n * sizeof(double), cudaMemcpyDeviceToDevice);

    double rz_old = reducedot(XParam, XModel.blocks, XModel.fluxml.r, XModel.fluxml.z, XModel.fluxml.store);//reduceDot(ctx.r, ctx.z, n);

    for (int iter = 0; iter < maxIter; ++iter)
    {
        //haloExchange(ctx.p, ctx);
        //matvec_kernel<<<blocks, threads>>>(ctx.p, ctx.Ap, ctx);

		HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.p);
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.p);
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPUBMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.p);
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.p);
		CUDA_CHECK(cudaDeviceSynchronize());

		matvec_kernel<<<gridDim, blockDim>>>(XParam, XModel.blocks, XModel.fluxml.p, XModel.fluxml.Ap, XModel.fluxml);   // Ap = A(eta0)
		CUDA_CHECK(cudaDeviceSynchronize());

        //double pAp  = reduceDot(ctx.p, ctx.Ap, n);
		double pAp = reducedot(XParam, XModel.blocks, XModel.fluxml.p, XModel.fluxml.Ap,XModel.fluxml.store)

        double alpha = rz_old / pAp;

		axpy_kernel<<<gridDim, blockDim>>>(XParam, XModel.blocks,XModel.fluxml.eta_n, XModel.fluxml.p, alpha);
		CUDA_CHECK(cudaDeviceSynchronize());
		axpy_kernel<<<gridDim, blockDim>>>(XParam, XModel.blocks,XModel.fluxml.r, XModel.fluxml.Ap, -alpha);
		CUDA_CHECK(cudaDeviceSynchronize());

        //axpy_kernel<<<blocks, threads>>>(ctx.eta_n, ctx.p, alpha, ctx);   // eta += alpha*p
        //axpy_kernel<<<blocks, threads>>>(ctx.r,     ctx.Ap, -alpha, ctx); // r   -= alpha*Ap

        double resNorm = reduceAbsMax(XParam, XModel.blocks,XModel.fluxml.r, ,XModel.fluxml.store);
        if (resNorm < tol) break;

        jacobi_apply_kernel<<<gridDim, blockDim>>>(XParam, XModel.blocks,XModel.fluxml.r, XModel.fluxml.z, XModel.fluxml.diagInv);
		CUDA_CHECK(cudaDeviceSynchronize());

        double rz_new = reduceDot(XParam, XModel.blocks, XModel.fluxml.r, XModel.fluxml.z, XModel.fluxml.store);
        double beta = rz_new / rz_old;

        xpby_kernel<<<blocks, threads>>>(XParam, XModel.blocks,XModel.fluxml.p, XModel.fluxml.z, beta);
		CUDA_CHECK(cudaDeviceSynchronize());

        rz_old = rz_new;
    }

    // ctx.eta_n now holds eta^{n+1}. Caller reconstructs theta_H*(hu)^{n+1}
    // exactly as in implicit.h's `pressure` event:
    //   hu.x[] = theta_H*(hf.x[]*uf.x[] + dt*ha.x[]) - dt*ha.x[]
    // using the freshly solved eta for ha.x[] = hf.x[]*a_baro(eta,0), then
    // pass that into your existing advect()-equivalent for the remaining
    // theta_H*dt portion (mirrors why implicit.h scales by theta_H there:
    // the (1-theta_H)*dt advection was already done in half_advection).
}
*/

template <class T> void AdvecML(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel,T dt)
{

	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	// for flux reconstruction the loop overlap the right(or top for the y direction) halo
	dim3 blockDimKX(XParam.blkwidth + XParam.halowidth, XParam.blkwidth, 1);
	dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + XParam.halowidth, 1);

	// For halo corners
	dim3 blockDimHC(4, 1, 1);

	// Fill halo for Fu and Fv
	dim3 blockDimHaloLR(1, XParam.blkwidth, 1);
	//dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDimHaloLR(XParam.nblk, 1, 1);

	dim3 blockDimHaloBT(XParam.blkwidth, 1, 1);
	dim3 gridDimHaloBT(XParam.nblk , 1, 1);


	// Check hu/hv
	CheckadvecMLY << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, dt, XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

	CheckadvecMLX << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, dt, XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Fill halo u and v calc grd for u and v and fill halo for hu and hv
	// 

	fillHaloGPU(XParam, XModel.blocks, XModel.evolv.u);
	fillHaloGPU(XParam, XModel.blocks, XModel.evolv.v);
	/*
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hu);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hv);

	HaloFluxGPULRnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hu);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUBTnew << <gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	CUDA_CHECK(cudaDeviceSynchronize());*/

	//HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	//HaloFluxGPUBMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hu);
	CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hu);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hu);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUBMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hu);
	CUDA_CHECK(cudaDeviceSynchronize());


	for (int iseg = 0; iseg < XForcing.bndseg.size(); iseg++)
	{
		FlowbndFluxML(XParam, XLoop.totaltime + XLoop.dt, XModel.blocks, XForcing.bndseg[iseg], XForcing.Atmp, XModel.evolv, XModel.fluxml);
	}
	//HaloFluxGPULRnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hu);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUBTnew << <gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPULRnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUBTnew << <gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hu);
	//CUDA_CHECK(cudaDeviceSynchronize());

	gradientSMC << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XModel.evolv.u, XModel.grad.dudx, XModel.grad.dudy);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientSMC << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XModel.evolv.v, XModel.grad.dvdx, XModel.grad.dvdy);
	CUDA_CHECK(cudaDeviceSynchronize());


	/*fillHaloGPU(XParam, XModel.blocks, XModel.grad.dudx);
	fillHaloGPU(XParam, XModel.blocks, XModel.grad.dudy);
	fillHaloGPU(XParam, XModel.blocks, XModel.grad.dvdx);
	fillHaloGPU(XParam, XModel.blocks, XModel.grad.dvdy);*/


	gradientHaloGPUnew(XParam, XModel.blocks, XModel.evolv.u, XModel.grad.dudx, XModel.grad.dudy);
	gradientHaloGPUnew(XParam, XModel.blocks, XModel.evolv.v, XModel.grad.dvdx, XModel.grad.dvdy);

	HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.grad.dvdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.grad.dudy);
	CUDA_CHECK(cudaDeviceSynchronize());


	HaloFluxGPUBMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.grad.dudx);
	CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.grad.dvdy);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.grad.dudy);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.grad.dudx);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.grad.dvdx);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUBMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.grad.dvdx);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	fillCornersGPU <<< gridDim, blockDimHC, 0 >>> (XParam, XModel.blocks, XModel.fluxml.hu);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillCornersGPU << < gridDim, blockDimHC, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillCornersGPU << < gridDim, blockDimHC, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfu);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillCornersGPU << < gridDim, blockDimHC, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfv);
	CUDA_CHECK(cudaDeviceSynchronize());

	/*fillCornersGPU << < gridDim, blockDimHC, 0 >> > (XParam, XModel.blocks, XModel.evolv.u);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillCornersGPU << < gridDim, blockDimHC, 0 >> > (XParam, XModel.blocks, XModel.evolv.v);
	CUDA_CHECK(cudaDeviceSynchronize());*/
	
	//hv hfv u hu hfu v


	
	// Advection
	AdvecFluxML << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, dt, XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

	

	/*fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.Fux);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.Fvx);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.Fuy);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.Fvy);*/

	HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.Fux);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.Fuy);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPURMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.Fvx);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew << < gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.Fvy);
	CUDA_CHECK(cudaDeviceSynchronize());

	
	/*
	refine_bilinearGPU(XParam, XModel.blocks, XModel.fluxml.Fux);
	refine_bilinearGPU(XParam, XModel.blocks, XModel.fluxml.Fvx);
	refine_bilinearGPU(XParam, XModel.blocks, XModel.fluxml.Fuy);
	refine_bilinearGPU(XParam, XModel.blocks, XModel.fluxml.Fvy);
	*/
	/*
	HaloFluxGPULRnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.Fux);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPULRnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.Fvx);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUBTnew << <gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.Fvy);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUBTnew << <gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.Fuy);
	CUDA_CHECK(cudaDeviceSynchronize());

	*/

	AdvecEv << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, dt, XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());


}
template void AdvecML<float>(Param XParam, Loop<float>& XLoop, Forcing<float> XForcing, Model<float> XModel,float dt);
template void AdvecML<double>(Param XParam, Loop<double>& XLoop, Forcing<float> XForcing, Model<double> XModel,double dt);


template <class T> void solveEtaPCG(Param XParam, Model<T> XModel,T dt)
{
    double tol = XParam.mg_tol;//1e-5;
	int maxIter = XParam.max_iter;//100

	int n = (XParam.blkwidth + XParam.halowidth*2)*(XParam.blkwidth + XParam.halowidth*2) * XParam.nblk;
   	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	// for flux reconstruction the loop overlap the right(or top for the y direction) halo
	dim3 blockDimKX(XParam.blkwidth + XParam.halowidth, XParam.blkwidth, 1);
	dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + XParam.halowidth, 1);

	// Fill halo for Fu and Fv
	dim3 blockDimHaloLR(1, XParam.blkwidth, 1);
	//dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDimHaloLR(XParam.nblk, 1, 1);

	dim3 blockDimHaloBT(XParam.blkwidth, 1, 1);
	dim3 gridDimHaloBT(XParam.nblk , 1, 1);


	T maxerror;

    HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >>> (XParam, XModel.blocks, XModel.fluximp.alpha_x);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUBMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >>> (XParam, XModel.blocks, XModel.fluximp.alpha_y);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPULMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >>> (XParam, XModel.blocks, XModel.fluximp.alpha_x);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >>> (XParam, XModel.blocks, XModel.fluximp.alpha_y);
	CUDA_CHECK(cudaDeviceSynchronize());

	// HaloFluxGPUTMLclamp<<< gridDimHaloBT, blockDimHaloBT, 0 >>>(XParam, XModel.blocks,XModel.fluximp.alpha_y,T(0.0));
	// CUDA_CHECK(cudaDeviceSynchronize());

	// HaloFluxGPURMLclamp<<< gridDimHaloLR, blockDimHaloLR, 0 >>> (XParam, XModel.blocks,XModel.fluximp.alpha_x,T(0.0));
	// CUDA_CHECK(cudaDeviceSynchronize());

	fillHaloGPU(XParam, XModel.blocks, XModel.fluximp.alpha_x);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluximp.alpha_y);

    jacobi_diag<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp);
	CUDA_CHECK(cudaDeviceSynchronize());

    // --- initial residual: r = rhs_eta - A(eta_r) ---
    //haloExchange(f.eta_r, g);

	matvec_facefieldx<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks,XModel.fluximp.eta_r,XModel.fluximp.g_x,XModel.fluximp.alpha_x);
	CUDA_CHECK(cudaDeviceSynchronize());

	matvec_facefieldy<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks,XModel.fluximp.eta_r,XModel.fluximp.g_y,XModel.fluximp.alpha_y);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluximp.g_x);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluximp.g_y);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLclamp<<< gridDimHaloBT, blockDimHaloBT, 0 >>>(XParam, XModel.blocks,XModel.fluximp.g_y,T(0.0));
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPURMLclamp<<< gridDimHaloLR, blockDimHaloLR, 0 >>> (XParam, XModel.blocks,XModel.fluximp.g_x,T(0.0));
	CUDA_CHECK(cudaDeviceSynchronize());

	// fillHaloGPU(XParam, XModel.blocks, XModel.fluximp.g_x);
	// fillHaloGPU(XParam, XModel.blocks, XModel.fluximp.g_y);

    //matvec_facefield<<<blocks, threads>>>(f.eta_r, f.g_x, f.alpha_eta_x, g);
    // matvec_facefield_y<<<...>>>(f.eta_r, f.g_y, f.alpha_eta_y, g);  (y-mirror)
    matvec_apply<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks,XModel.fluximp.eta_r, XModel.fluximp.Ap, XModel.fluximp.g_x, XModel.fluximp.g_y);
	CUDA_CHECK(cudaDeviceSynchronize());


    CUDA_CHECK(cudaMemcpy(XModel.fluximp.r, XModel.fluximp.rhs_eta, n * sizeof(T), cudaMemcpyDeviceToDevice));
	//x2_kernel<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp.rhs_eta, XModel.fluximp.r);
	//CUDA_CHECK(cudaDeviceSynchronize());

    //vec_axpy<<<blocks1d, threads1d>>>(f.r, f.Ap, -1.0, n);
	axpy_kernel<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp.r, XModel.fluximp.Ap,T(-1.0));
	CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaMemcpy(XModel.fluximp.z, XModel.fluximp.r, n * sizeof(T), cudaMemcpyDeviceToDevice));

    //vec_jacobi_apply<<<blocks1d, threads1d>>>(f.r, f.z, f.diagInv, n);
	jacobi_apply_kernel<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp.r,XModel.fluximp.z,XModel.fluximp.diagInv);
	CUDA_CHECK(cudaDeviceSynchronize());


	CUDA_CHECK(cudaMemcpy(XModel.fluximp.p, XModel.fluximp.z, n * sizeof(T), cudaMemcpyDeviceToDevice));
	//x2_kernel<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp.z, XModel.fluximp.p);
	//CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(XModel.time.arrmax, XModel.fluximp.z, n * sizeof(T), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(XModel.time.arrmin, XModel.fluximp.r, n * sizeof(T), cudaMemcpyDeviceToDevice));

    T rz_old = reducedot(XParam, XModel.blocks,XModel.fluximp.r, XModel.fluximp.z, XModel.fluximp.store);

	CUDA_CHECK(cudaMemcpy(XModel.fluximp.z, XModel.time.arrmax,  n * sizeof(T), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(XModel.fluximp.r, XModel.time.arrmin,  n * sizeof(T), cudaMemcpyDeviceToDevice));


	
	maxerror=reduceabsmax(XParam, XModel.blocks, XModel.fluximp.r, XModel.fluximp.store);

	CUDA_CHECK(cudaMemcpy(XModel.fluximp.r, XModel.time.arrmin,  n * sizeof(T), cudaMemcpyDeviceToDevice));

	if(maxerror > tol)
	{
    for (int iter = 0; iter < maxIter; ++iter)
    {

		//log("implicit Iteration " + std::to_string(iter));
	
		// Update Halo for eta_r
		// HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluximp.eta_r);
		// CUDA_CHECK(cudaDeviceSynchronize());

		// HaloFluxGPUBMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluximp.eta_r);
		// CUDA_CHECK(cudaDeviceSynchronize());

		// HaloFluxGPULMLnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluximp.eta_r);
		// CUDA_CHECK(cudaDeviceSynchronize());

		// HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluximp.eta_r);
		// CUDA_CHECK(cudaDeviceSynchronize());

		fillHaloGPU(XParam, XModel.blocks, XModel.fluximp.eta_r);

        HaloFluxGPULMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluximp.p);
		//CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPUBMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluximp.p);
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPUTMLclamp<<< gridDimHaloBT, blockDimHaloBT, 0 >>>(XParam, XModel.blocks,XModel.fluximp.p,T(0.0));
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPURMLclamp<<< gridDimHaloLR, blockDimHaloLR, 0 >>> (XParam, XModel.blocks,XModel.fluximp.p,T(0.0));
		CUDA_CHECK(cudaDeviceSynchronize());


		matvec_facefieldx<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp.p, XModel.fluximp.g_x, XModel.fluximp.alpha_x);
		CUDA_CHECK(cudaDeviceSynchronize());

		matvec_facefieldy<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp.p, XModel.fluximp.g_y, XModel.fluximp.alpha_y);
		CUDA_CHECK(cudaDeviceSynchronize());

		

        //matvec_facefield<<<blocks, threads>>>(f.p, f.g_x, f.alpha_eta_x, g);
        // matvec_facefield_y<<<...>>>(f.p, f.g_y, f.alpha_eta_y, g);

        //matvec_apply<<<blocks, threads>>>(f.p, f.Ap, f.g_x, f.g_y, g);

		HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluximp.g_x);
		//CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluximp.g_y);
		CUDA_CHECK(cudaDeviceSynchronize());

		// fillHaloGPU(XParam, XModel.blocks, XModel.fluximp.g_x);
		// fillHaloGPU(XParam, XModel.blocks, XModel.fluximp.g_y);

		HaloFluxGPUTMLclamp<<< gridDimHaloBT, blockDimHaloBT, 0 >>>(XParam, XModel.blocks,XModel.fluximp.g_y,T(0.0));
		CUDA_CHECK(cudaDeviceSynchronize());

		HaloFluxGPURMLclamp<<< gridDimHaloLR, blockDimHaloLR, 0 >>> (XParam, XModel.blocks,XModel.fluximp.g_x,T(0.0));
		CUDA_CHECK(cudaDeviceSynchronize());

		matvec_apply<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks,XModel.fluximp.p, XModel.fluximp.Ap, XModel.fluximp.g_x, XModel.fluximp.g_y);
		CUDA_CHECK(cudaDeviceSynchronize());


        //double pAp   = reduceDot(f.p, f.Ap, n);

		CUDA_CHECK(cudaMemcpy(XModel.time.arrmax, XModel.fluximp.p, n * sizeof(T), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(XModel.time.arrmin, XModel.fluximp.Ap, n * sizeof(T), cudaMemcpyDeviceToDevice));

		T pAp   = reducedot(XParam, XModel.blocks, XModel.fluximp.p, XModel.fluximp.Ap, XModel.fluximp.store);
        T alpha = rz_old / pAp;

		CUDA_CHECK(cudaMemcpy(XModel.fluximp.p, XModel.time.arrmax,  n * sizeof(T), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(XModel.fluximp.Ap, XModel.time.arrmin, n * sizeof(T), cudaMemcpyDeviceToDevice));

		//printf("rz_old = %f, pAp = %f, alpha = %f, rz_new = %f, beta = %f, maxerror = %f\n",rz_old,pAp,alpha,rz_new, beta,maxerror);
		//printf("rz_old = %f, pAp = %f, alpha = %f, \n",rz_old,pAp,alpha);

		

        //vec_axpy<<<blocks1d, threads1d>>>(f.eta_r, f.p,  alpha, n);
        //vec_axpy<<<blocks1d, threads1d>>>(f.r,     f.Ap, -alpha, n);

		axpy_kernel<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp.eta_r,XModel.fluximp.p,alpha);
		CUDA_CHECK(cudaDeviceSynchronize());

		axpy_kernel<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp.r,XModel.fluximp.Ap,-alpha);
		CUDA_CHECK(cudaDeviceSynchronize());

        //if (reduceAbsMax(f.r, n) < tolerance) break;

		CUDA_CHECK(cudaMemcpy(XModel.time.arrmax, XModel.fluximp.r, n * sizeof(T), cudaMemcpyDeviceToDevice));

		maxerror=reduceabsmax(XParam, XModel.blocks, XModel.fluximp.r, XModel.fluximp.store);
		
		CUDA_CHECK(cudaMemcpy(XModel.fluximp.r, XModel.time.arrmax, n * sizeof(T), cudaMemcpyDeviceToDevice));

		if (maxerror < tol) break;

       	// vec_jacobi_apply<<<blocks1d, threads1d>>>(f.r, f.z, f.diagInv, n);
	    jacobi_apply_kernel<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp.r,XModel.fluximp.z, XModel.fluximp.diagInv);
	    CUDA_CHECK(cudaDeviceSynchronize());

        //double rz_new = reducedot(f.r, f.z, n);

		CUDA_CHECK(cudaMemcpy(XModel.time.arrmax, XModel.fluximp.r, n * sizeof(T), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(XModel.time.arrmin, XModel.fluximp.z, n * sizeof(T), cudaMemcpyDeviceToDevice));

		T rz_new = reducedot(XParam, XModel.blocks,XModel.fluximp.r, XModel.fluximp.z, XModel.fluximp.store);
        T beta = rz_new / rz_old;

		CUDA_CHECK(cudaMemcpy(XModel.fluximp.r, XModel.time.arrmax, n * sizeof(T), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(XModel.fluximp.z, XModel.time.arrmin, n * sizeof(T), cudaMemcpyDeviceToDevice));

		///xpby_kernel(Param XParam, BlockP<T> XBlock, double* p, const double* z, double beta)
		xpby_kernel<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks, XModel.fluximp.p, XModel.fluximp.z,beta);
        //vec_xpby<<<blocks1d, threads1d>>>(f.p, f.z, beta, n);
		 CUDA_CHECK(cudaDeviceSynchronize());


		
        rz_old = rz_new;

		if (iter == maxIter - 1)
		{
			log("Warning! Implicit max iteration reached. residual = " + std::to_string(rz_old));
			//printf("rz_old = %f, pAp = %f, alpha = %f, \n",rz_old,pAp,alpha);
		}
		
    }
	}
	
    // f.eta_r now holds eta_r^{n+1} (== eta^{n+1} unless rigid lid).


}

template <class T> void test_symetry(Param XParam, Model<T> XModel,T dt)
{
    double tol = XParam.mg_tol;//1e-5;
	int maxIter = XParam.max_iter;//100

	int n = (XParam.blkwidth + XParam.halowidth*2)*(XParam.blkwidth + XParam.halowidth*2) * XParam.nblk;
   	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	// for flux reconstruction the loop overlap the right(or top for the y direction) halo
	dim3 blockDimKX(XParam.blkwidth + XParam.halowidth, XParam.blkwidth, 1);
	dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + XParam.halowidth, 1);

	// Fill halo for Fu and Fv
	dim3 blockDimHaloLR(1, XParam.blkwidth, 1);
	//dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDimHaloLR(XParam.nblk, 1, 1);

	dim3 blockDimHaloBT(XParam.blkwidth, 1, 1);
	dim3 gridDimHaloBT(XParam.nblk , 1, 1);


	// Do the first variable eta!

	//Second var is zb


	matvec_facefieldx<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks,XModel.fluximp.eta_r,XModel.fluximp.g_x,XModel.fluximp.alpha_x);
	CUDA_CHECK(cudaDeviceSynchronize());

	matvec_facefieldy<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks,XModel.fluximp.eta_r,XModel.fluximp.g_y,XModel.fluximp.alpha_y);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluximp.g_x);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluximp.g_y);
	CUDA_CHECK(cudaDeviceSynchronize());

    //matvec_facefield<<<blocks, threads>>>(f.eta_r, f.g_x, f.alpha_eta_x, g);
    // matvec_facefield_y<<<...>>>(f.eta_r, f.g_y, f.alpha_eta_y, g);  (y-mirror)
    matvec_apply<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks,XModel.fluximp.eta_r, XModel.fluximp.Ap, XModel.fluximp.g_x, XModel.fluximp.g_y);
	CUDA_CHECK(cudaDeviceSynchronize());



	CUDA_CHECK(cudaMemcpy(XModel.time.arrmax, XModel.zb, n * sizeof(T), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(XModel.time.arrmin, XModel.fluximp.Ap, n * sizeof(T), cudaMemcpyDeviceToDevice));

    T rz_A = reducedot(XParam, XModel.blocks,XModel.zb, XModel.fluximp.Ap, XModel.fluximp.store);

	CUDA_CHECK(cudaMemcpy(XModel.zb, XModel.time.arrmax,  n * sizeof(T), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(XModel.fluximp.Ap, XModel.time.arrmin,  n * sizeof(T), cudaMemcpyDeviceToDevice));




	matvec_facefieldx<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks,XModel.zb,XModel.fluximp.g_x,XModel.fluximp.alpha_x);
	CUDA_CHECK(cudaDeviceSynchronize());

	matvec_facefieldy<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks,XModel.zb,XModel.fluximp.g_y,XModel.fluximp.alpha_y);
	CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPURMLnew <<< gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluximp.g_x);
	//CUDA_CHECK(cudaDeviceSynchronize());

	HaloFluxGPUTMLnew <<< gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluximp.g_y);
	CUDA_CHECK(cudaDeviceSynchronize());

    //matvec_facefield<<<blocks, threads>>>(f.eta_r, f.g_x, f.alpha_eta_x, g);
    // matvec_facefield_y<<<...>>>(f.eta_r, f.g_y, f.alpha_eta_y, g);  (y-mirror)
    matvec_apply<<<gridDim, blockDim, 0 >>>(XParam, XModel.blocks,XModel.zb, XModel.fluximp.r, XModel.fluximp.g_x, XModel.fluximp.g_y);
	CUDA_CHECK(cudaDeviceSynchronize());



	CUDA_CHECK(cudaMemcpy(XModel.time.arrmax, XModel.fluximp.eta_r, n * sizeof(T), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(XModel.time.arrmin, XModel.fluximp.r, n * sizeof(T), cudaMemcpyDeviceToDevice));

    T rz_B = reducedot(XParam, XModel.blocks,XModel.fluximp.eta_r, XModel.fluximp.Ap, XModel.fluximp.store);

	CUDA_CHECK(cudaMemcpy(XModel.fluximp.eta_r, XModel.time.arrmax,  n * sizeof(T), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(XModel.fluximp.r, XModel.time.arrmin,  n * sizeof(T), cudaMemcpyDeviceToDevice));

	printf("symmetry check: %g vs %g (diff %g)\n", rz_A, rz_B, rz_A - rz_B);
}






