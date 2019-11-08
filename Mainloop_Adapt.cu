void mainloopGPUADA(Param XParam) // float, metric coordinate
{
	double nextoutputtime = XParam.outputtimestep;
	int nstep = 0;
	int nTSsteps = 0;

	int windstep = 1;
	std::vector<Pointout> zsout;

	std::vector< std::vector< Pointout > > zsAllout;

	Pointout stepread;

	FILE* fsSLTS;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
							 //dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);


	for (int o = 0; o < XParam.TSoutfile.size(); o++)
	{
		//Overwrite existing files
		fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "w");
		fprintf(fsSLTS, "# x=%f\ty=%f\ti=%d\tj=%d\t%s\n", XParam.TSnodesout[o].x, XParam.TSnodesout[o].y, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSoutfile[o].c_str());
		fclose(fsSLTS);

		// Add empty row for each output point
		zsAllout.push_back(std::vector<Pointout>());
	}
	// Reset GPU mean and max arrays
	ResetmeanvarGPU(XParam);
	ResetmaxvarGPU(XParam);


	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		LeftFlowBnd(XParam);
		RightFlowBnd(XParam);
		TopFlowBnd(XParam);
		BotFlowBnd(XParam);



		// Core engine
		XParam.dt = FlowGPU(XParam, nextoutputtime);

		// River
		if (XParam.Rivers.size() > 0)
		{
			RiverSource(XParam);
		}

		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;

		// Do Sum & Max variables Here
		meanmaxvarGPU(XParam);




		//Check for TSoutput
		if (XParam.TSnodesout.size() > 0)
		{
			pointoutputstep(XParam, gridDim, blockDim, nTSsteps, zsAllout);


		}

		if (nextoutputtime - XParam.totaltime <= XParam.dt * 0.00001f && XParam.outputtimestep > 0.0)
		{
			// Avg var sum here
			DivmeanvarGPU(XParam, nstep * 1.0f);

			if (XParam.outvort == 1)
			{
				CalcVorticity << <gridDim, blockDim, 0 >> > (vort_g, dvdx_g, dudy_g);
				CUDA_CHECK(cudaDeviceSynchronize());
			}

			if (!XParam.outvars.empty())
			{
				writenctimestep(XParam.outfile, XParam.totaltime);

				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					if (OutputVarMaplen[XParam.outvars[ivar]] > 0)
					{
						if (XParam.GPUDEVICE >= 0)
						{
							//Should be async
							CUDA_CHECK(cudaMemcpy(OutputVarMapCPU[XParam.outvars[ivar]], OutputVarMapGPU[XParam.outvars[ivar]], OutputVarMaplen[XParam.outvars[ivar]] * sizeof(float), cudaMemcpyDeviceToHost));

						}
						//Create definition for each variable and store it
						writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
					}
				}
			}

			nextoutputtime = min(nextoutputtime + XParam.outputtimestep, XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//.Reset Avg Variables
			ResetmeanvarGPU(XParam);
			if (XParam.resetmax == 1)
			{
				ResetmaxvarGPU(XParam);
			}




			//

			// Reset nstep
			nstep = 0;
		} // End of output part

	} //Main while loop
}