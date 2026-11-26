//////////////////////////////////////////////////////////////////////////////////
// BG_Flood Main function						                                //
// Copyright (C) 2018 Bosserelle                                                //
// This code contains an adaptation of the St Venant equation from Basilisk		//
// See																			//
// http://basilisk.fr/src/saint-venant.h and									//
// S. Popinet. Quadtree-adaptive tsunami modelling. Ocean Dynamics,				//
// doi: 61(9) : 1261 - 1285, 2011												//
//                                                                              //
// This program is free software: you can redistribute it and/or modify         //
// it under the terms of the GNU General Public License as published by         //
// the Free Software Foundation.                                                //
//                                                                              //
// This program is distributed in the hope that it will be useful,              //
// but WITHOUT ANY WARRANTY; without even the implied warranty of               //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                //
// GNU General Public License for more details.                                 //
//                                                                              //
// You should have received a copy of the GNU General Public License            //
// along with this program.  If not, see <http://www.gnu.org/licenses/>.        //
//////////////////////////////////////////////////////////////////////////////////

// includes, system

#include "BG_Flood.h"


/**
 * @brief Main entry point for the BG_Flood model executable.
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return Status code (0 for success)
 *
 * Main function
 * This function is the entry point to the software
 * The main function setups all the init of the model and then calls the mainloop to actually run the model
 *
 * There are 3 main class storing information about the model: XParam (class Param), XModel (class Model) and XForcing (class Forcing)
 * Leading X stands for eXecution and is to avoid confusion between the class variable and the class declaration
 * When running with the GPU there is also XModel_g
 * which is the same as XModel but with GPU specific pointers
 *
 * The (spectral) wave model follows the same structure: XWaveParam (class WaveParam) and XWave (class WaveModel)
 * and XWave_g for the GPU pointers. The wave model shares the mesh, bathymetry and forcing with the flow model.
 * The flow and/or the wave model can be turned on/off with the "flowmodel" and "wavemodel" keywords.
 *
 * This function does:
 * - Reads the inputs to the model
 * - Allocate memory on GPU and CPU
 * - Prepare and initialise memory and arrays on CPU and GPU
 * - Setup initial condition
 * - Adapt grid if require
 * - Prepare output file
 * - Run main loop
 * - Clean up and close
 *
 * It coordinates CPU and GPU resources and calls the mainwork routine for model execution.
 */
int main(int argc, char* argv[])
{
	//===========================================
	// Read model argument (filename). If one is not given use the default name
	std::string ParamFile;

	if (argc > 1)
	{
		ParamFile = argv[1];
	}
	else
	{
		ParamFile = "BG_param.txt";
	}

	//std::cout << ParamFile << '\n';

	//===========================================
	//  Define the main parameter controling the model (XModels class at produced later) 
	Param XParam;
	WaveParam XWaveParam; // Wave model parameters
	Forcing<float> XForcing; // for reading and storing forcing data (CPU only) // by default we read only float precision!
	// Start timer to keep track of time
	XParam.startcputime = clock();

	
	// Create/overwrite existing 
	create_logfile();

	//============================================
	// Read Operational file
	// Also check XParam sanity

	Readparamfile(XParam, XForcing, XWaveParam, ParamFile);


	//============================================
	// Create external forcing and model pointers
	// Before this is done we need to check
	// if the model will be double or float precision

	Model<double> XModel_d; // For CPU double pointers
	Model<double> XModel_gd; // For GPU double pointers

	Model<float> XModel_f; // For CPU float pointers
	Model<float> XModel_gf; // For GPU float pointers

	WaveModel<double> XWave_d; // For CPU double wave pointers
	WaveModel<double> XWave_gd; // For GPU double wave pointers

	WaveModel<float> XWave_f; // For CPU float wave pointers
	WaveModel<float> XWave_gf; // For GPU float wave pointers

	if (XParam.doubleprecision < 1)
	{
		// Call the Float precision run
		mainwork(XParam, XWaveParam, XForcing, XModel_f, XModel_gf, XWave_f, XWave_gf);
	}
	else
	{
		mainwork(XParam, XWaveParam, XForcing, XModel_d, XModel_gd, XWave_d, XWave_gd);
	}

}

/**
 * @brief Main model setup and execution routine for BG_Flood.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XWaveParam Wave model parameters
 * @param XForcing Forcing data
 * @param XModel Model structure (CPU)
 * @param XModel_g Model structure (GPU)
 * @param XWave Wave model structure (CPU)
 * @param XWave_g Wave model structure (GPU)
 * @return Status code (0 for success)
 *
 * This function performs the main setup and execution steps for the BG_Flood model:
 * - Reads and verifies input/forcing data
 * - Initializes mesh and initial conditions
 * - Performs initial adaptation
 * - Initializes the wave model (if requested)
 * - Sets up GPU resources
 * - Runs the main simulation loop (MainLoop)
 * - Handles test mode if specified
 *
 * Integrates all major model components and coordinates CPU/GPU execution.
 */
template < class T > int mainwork(Param XParam, WaveParam XWaveParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g, WaveModel<T> XWave, WaveModel<T> XWave_g)
{
	//============================================
	// Read the forcing data (Including bathymetry)
	readforcing(XParam, XForcing);

	//=============================================
	// Verify Compatibility of forcing and model Parameters
	checkparamsanity(XParam, XForcing);
	checkwaveparamsanity(XParam, XWaveParam, XForcing);

	//============================================
	// Prepare initial mesh layout
	InitMesh(XParam, XForcing, XModel);

	//============================================
	// Prepare initial conditions on CPU
	InitialConditions(XParam, XForcing, XModel);

	//============================================
	// Initial adaptation
	InitialAdaptation(XParam, XForcing, XModel);

	//============================================
	// Initialise the wave model on CPU (bypassed if the wave model is not used)
	// This is done after the adaptation so the wave model uses the final mesh
	InitWaveModel(XParam, XWaveParam, XForcing, XModel, XWave);

	//============================================
	// Setup GPU (bypassed within the function if no suitable GPU is available)
	SetupGPU(XParam, XModel,XForcing, XModel_g);

	// Setup wave model on GPU (bypassed if no GPU or no wave model, needs to be after SetupGPU)
	SetupWaveGPU(XParam, XWaveParam, XModel_g, XWave, XWave_g);



	//
	log("\nModel setup complete");
	log("#################################");
	//===========================================
	//   End of Initialisation time
	//===========================================
	XParam.setupcputime = clock();
	bool isfailed = false;

	if (XParam.test < 0)
	{
		//============================================
		// MainLoop
		//printf("XCulvertsF h1=%f\n", XModel.culvertsF.h1[0]);
		MainLoop(XParam, XWaveParam, XForcing, XModel, XModel_g, XWave, XWave_g);
	}
	else
	{
		//============================================
		// Testing
		//Gaussianhump(XParam, XModel, XModel_g);
		isfailed = Testing(XParam, XForcing, XModel, XModel_g);
	}

		

	//===========================================
	//   End of Model
	//===========================================
	XParam.endcputime = clock();

	//===========================================
	//   Log the timer
	//===========================================
	log("#################################");
	log("End Computation");
	log("#################################");
	log("Total runtime= " + std::to_string((XParam.endcputime - XParam.startcputime) / CLOCKS_PER_SEC) + " seconds");
	log("Model Setup time= " + std::to_string((XParam.setupcputime - XParam.startcputime) / CLOCKS_PER_SEC) + " seconds");
	log("Model runtime= " + std::to_string((XParam.endcputime - XParam.setupcputime) / CLOCKS_PER_SEC) + " seconds");


	if (XParam.GPUDEVICE >= 0)
	{
		size_t free_byte;

		size_t total_byte;

		CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

		XParam.GPU_totalmem_byte = (total_byte - free_byte) - XParam.GPU_initmem_byte;
		log("Model final memory usage= " + std::to_string((XParam.GPU_totalmem_byte) / 1024.0 / 1024.0) + " MB");

	}


	//============================================
	// Cleanup and free memory
	//
	if (XParam.test < 0)
	{
		exit(0);
	}
	else 
	{
		exit(isfailed);
	}
	
}
