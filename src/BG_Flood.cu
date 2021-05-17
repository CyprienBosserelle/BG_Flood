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




/*! \fn int main(int argc, char **argv)
* Main function 
* This function is the entry point to the software
*/
int main(int argc, char **argv)
{

	//The main function setups all the init of the model and then calls the mainloop to actually run the model

	// There are 3 main class storing information about the model: XParam (class Param), XModel (class Model) and XForcing (class Forcing)
	// Leading X stands for eXecution and is to avoid confusion between the class variable and the class declaration
	// When running with the GPU there is also XModel_g
	// which is the same as XModel but with GPU specific pointers

	//First part reads the inputs to the model
	//then allocate memory on GPU and CPU
	//Then prepare and initialise memory and arrays on CPU and GPU
	// Setup initial condition
	// Adapt grid if require
	// Prepare output file
	// Run main loop
	// Clean up and close

	//===========================================
	//  Define the main parameter controling the model (XModels class at produced later) 
	Param XParam;
	Forcing<float> XForcing; // for reading and storing forcing data (CPU only) // by default we read only float precision!
	// Start timer to keep track of time
	XParam.startcputime = clock();

	// Create/overight existing 
	create_logfile();

	//============================================
	// Read Operational file
	// Also check XParam sanity

	Readparamfile(XParam,XForcing);
	

	//============================================
	// Create external forcing and model pointers
	// Before this is done we need to check
	// if the model will be double or float precision
	

	auto modeltype = XParam.doubleprecision < 1 ? float() : double();
	Model<decltype(modeltype)> XModel; // For CPU pointers
	Model<decltype(modeltype)> XModel_g; // For GPU pointers

	//============================================
	// Read the forcing data (Including bathymetry)
	readforcing(XParam, XForcing);

	//=============================================
	// Verify Compatibility of forcing and model Parameters
	checkparamsanity(XParam, XForcing);

	//============================================
	// Prepare initial mesh layout
	InitMesh(XParam, XForcing, XModel);

	//============================================
	// Prepare initial conditions
	InitialConditions(XParam, XForcing, XModel);

	//============================================
	// Initial adaptation
	InitialAdaptation(XParam, XForcing, XModel);

	//============================================
	// Setup GPU (bypassed within the function if no suitable GPU is available)
	SetupGPU(XParam, XModel,XForcing, XModel_g);

	//
	log("\nModel setup complete");
	log("#################################");
	//===========================================
	//   End of Initialisation time
	//===========================================
	XParam.setupcputime = clock();

	if (XParam.test < 0)
	{
		//============================================
		// MainLoop
		MainLoop(XParam, XForcing, XModel, XModel_g);
	}
	else
	{
		//============================================
		// Testing
		//Gaussianhump(XParam, XModel, XModel_g);
		Testing(XParam, XForcing, XModel, XModel_g);

	}

	//log(std::to_string(XForcing.Bathy.val[50]));
	//TestingOutput(XParam, XModel);
	//CompareCPUvsGPU(XParam, XForcing, XModel, XModel_g);
	//Gaussianhump(XParam, XModel, XModel_g);
	

	//===========================================
	//   End of Model
	//===========================================
	XParam.endcputime = clock();
	//============================================
	// Cleanup and free memory

	exit(0);
}
