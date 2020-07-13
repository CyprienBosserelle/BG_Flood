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


template <class T>
__host__ __device__ double calcres(T dx, int level)
{
	return level < 0 ? dx * (1 << abs(level)) : dx / (1 << level);
}

/*! \fn int main(int argc, char **argv)
* Main function 
* This function is the entry point to the software
*/
int main(int argc, char **argv)
{

	//The main function setups all the init of the model and then calls the mainloop to actually run the model

	//===========================================
	//  Define the main variables controling the model 
	Param XParam;
	


	//First part reads the inputs to the model
	//then allocate memory on GPU and CPU
	//Then prepare and initialise memory and arrays on CPU and GPU
	// Setup initial condition
	// Adapt grid if require
	// Prepare output file
	// Run main loop
	// Clean up and close


	// Start timer to keep track of time
	XParam.startcputime = clock();

	// Create/overight existing 
	create_logfile();

	//////////////////////////////////////////////////////
	/////             Read Operational file          /////
	//////////////////////////////////////////////////////

	XParam = Readparamfile(XParam);
	
	

	


	double levdx = calcres(XParam.dx ,XParam.initlevel);// true grid resolution as in dx/2^(initlevel)
	printf("levdx=%f;1 << XParam.initlevel=%f\n", levdx, calcres(1.0, XParam.initlevel));

	XParam.nx = (XParam.xmax - XParam.xo) / (levdx)+1;
	XParam.ny = (XParam.ymax - XParam.yo) / (levdx)+1; //+1?




	exit(0);
}
