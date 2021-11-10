
#ifndef FORCING_H
#define FORCING_H

#include "General.h"
#include "Input.h"

struct TexSetP
{
	float xo, yo, dx; // used to calculate coordinates insode the device function
	float nowvalue;
	bool uniform;
	cudaArray* CudArr;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	//texture<float, 2, cudaReadModeElementType> tex;
	struct cudaResourceDesc resDesc;
	struct cudaTextureDesc texDesc;
	cudaTextureObject_t tex = 0;
};

struct bndTexP
{
	TexSetP WLS;
	TexSetP Uvel;
	TexSetP Vvel;
};

class forcingmap : public inputmap {
public:
	
	int nt;
	bool uniform = false;
	double to, tmax;
	double dt;
	int instep = 0; // last step that was read
	std::string inputfile;
	std::vector<Windin> unidata; // only used if uniform forcing
	double nowvalue; // temporary storage for value at a given time
	TexSetP GPU;

};

template <class T>
class deformmap : public inputmap
{
	//Deform are maps to applie to both zs and zb; this is often co-seismic vertical deformation used to generate tsunami initial wave
	// Here you can spread the deformation across a certain amount of time and apply it at any point in the model
public:
	double startime = 0.0;
	double duration = 0.0;
	T* val;
	

	TexSetP GPU;
};


template <class T>
struct DynForcingP: public forcingmap
{
	T *now;
	T *before, *after;
	T* val; // useful for reading from file
	
	// gpu version of these array
	T* now_g;
	T* before_g, * after_g;

};

template <class T>
struct StaticForcingP : public inputmap
{
	T *val;
	

};

//bnd
class bndparam {
public:
	std::vector<SLTS> data;
	bool on = false;
	int type = 1; // 0:Wall (no slip); 1:neumann (zeros gradient) [Default]; 2:sealevel dirichlet; 3: Absorbing 1D 4: Absorbing 2D (not yet implemented)
	std::string inputfile;
	int nbnd; // number of forcing bnds along the side (>=1 is side is on)
	int nblk = 0; //number of blocks where this bnd applies
	int side = 0; // 0: top bnd, 1: rightbnd, 2: bot bnd, 3, Left bnd
	int isright = 0;
	int istop = 0;
	bndTexP GPU;
	int* blks; // array of block where bnd applies 
	int* blks_g; // Also needed for GPU (because it should be a gpu allocated pointer) This is not pretty at all! In the future maybe using pagelocked memory or other new type may be beneficial 
};






template <class T>
struct Forcing
{
	DynForcingP<T> UWind;
	DynForcingP<T> VWind;
	DynForcingP<T> Rain;
	/* Rain dynamic forcing: This allow to force a time varying, space varying rain on the model, in mm/h.
	The rain can also be forced using a time serie (rain will be considered uniform on the domain)
	Ex: For a uniform rain: "rain=rain_forcing.txt" (2 column values, time (not necessary unformly distributed) and rain intensity)
	Ex: For a non-uniform rain: "rain=rain_forcing.nc?rain" (to define the entry netcdf file and the variable associated to the rain "rain", after the "?")
	Default: None
	*/
	DynForcingP<T> Atmp;

	std::vector<StaticForcingP<T>> Bathy; //Should be a vector at some point
	/* Bathymetry/Topography input, ONLY NECESSARY INPUT
	Different format are accepted: .asc, .nc, .md. , the grid must be regular with growing coordinate.
	This grid will define the extend of the model domain and model resolution (if not inform by the user).
	The coordinate can be cartesian or spheric (To be check).
	A list of file can also be use to provide a thiner resolution localy for example.
	The first file will be use to define the domain area and base resolution but the following file
	will be used during the refinement process.
	Ex: "bathy=Westport_DEM_2020.nc?z" or "topo=Westport_DEM_2020.asc"
	Ex: "bathy=South_Island_200.nc?z, West_Coast_100.nc?z, Westport_10.nc?z"
	Default: None but input NECESSARY
	*/

	StaticForcingP<T> cf; //cfmap;
	/*Bottom friction coefficient map (associated to the chosen bottom friction model)
	NEED TO BE MODIFIED TO HAVE THE GOOD KEYS
	Ex: cf=0.001;
	Ex: cfmap=bottom_friction.nc?bfc;
	Default: 0.0001 (constant)
	*/

	std::vector<StaticForcingP<int>> targetadapt;

	std::vector<deformmap<T>> deform;
	/*Deform are maps to applie to both zs and zb; this is often co-seismic vertical deformation used to generate tsunami initial wave
	Here you can spread the deformation across a certain amount of time and apply it at any point in the model
	Ex: XXXXXXXXXXXXXXXX
	Default: None
	*/
	

	std::vector<River> rivers;
	/*The river is added as a vertical discharge on a chosen area (the user input consisting in a Time serie and a rectangular area definition: river = Fluxfile,xstart,xend,ystart,yend).
	The whole cells containing the corners of the area will be included in the area, no horizontal velocity is applied.
	To add multiple rivers, just add different lines in the input file (one by river).
	Ex: river = Votualevu_R.txt,1867430,1867455,3914065,3914090;
	Default: None
	*/

	bndparam left;
	/* 0:Wall (no slip); 1:neumann (zeros gradient) [Default]; 2:sealevel dirichlet; 3: Absorbing 1D 4: Absorbing 2D (not yet implemented)
	For type 2 and 3 boundary, a file need to be added to determine the vaules at the boundary. This file will consist in a first time
	column (with possibly variable time steps) and levels in the following columns (1 column correspond to a constant value along the boundary,
	2 column will correspond to values at boundary edges with linear evolution in between, n colums will correspond to n regularly space
	applied values along the boundary)
	Ex: left = 0;
	Ex: left = 2,leftBnd.txt;
	Default: 1 *****TO DISCUSS******
	*/

	bndparam right;
	/*Same as left boundary*/
	bndparam top;
	/*Same as left boundary*/
	bndparam bot;
	/*Same as left boundary*/
	
	
};











// End of global definition
#endif
