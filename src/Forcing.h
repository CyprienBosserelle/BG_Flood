
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
	//If changing this default value, please change documentation later on the file
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


class AOIinfo {
public:
	std::string file;
	Polygon poly;
	bool active=false;
};


template <class T>
struct Forcing
{
	DynForcingP<T> UWind;
	DynForcingP<T> VWind;
	//Forcing the Wind;
	/*Spacially varying: 2 files are given, 1st file is U wind and second is V wind ( no rotation of the data is performed)
	Spacially uniform: 1 file is given then a 3 column file is expected, showing time, windspeed and direction.
	Wind direction is rotated (later) to the grid direction (using grdalpha input parameter)
	Ex: Wind = mywind.nc?uw,mywind.nc?vw
	Ex: Wind = MyWind.txt
	Default: None
	*/
	

	DynForcingP<T> Rain;
	/* This allow to force a time varying, space varying rain intensity on the model, in mm/h.
	Spacially varrying (rain map), a netcdf file is expected (with the variable associated to the rain after "?").
	Spacially uniform: the rain is forced using a time serie using a 2 column values table containing time (not necessary unformly distributed) and rain.
	Ex: rain=rain_forcing.txt 
	Ex: rain=rain_forcing.nc?RainIntensity
	Default: None
	*/
	DynForcingP<T> Atmp;
	/* The forcing pressure is expected to be in Pa and the effect of the atmospheric pressure gradient is calculated as the difference to a reference pressure Paref, converted to a height using Pa2.
	Ex: Atmp=AtmosphericPressure.nc?p
	Default: None
	*/

	std::vector<StaticForcingP<T>> Bathy; //Should be a vector at some point
	/* Bathymetry/Topography input, ONLY NECESSARY INPUT
	Different format are accepted: .asc, .nc, .md. , the grid must be regular with growing coordinate.
	This grid will define the extend of the model domain and model resolution (if not inform by the user).
	The coordinate can be cartesian or spherical (still in development).
	A list of file can also be use to provide a thiner resolution localy by using the key word each time on a different line.
	The first file will be use to define the domain area and base resolution but the following file
	will be used during the refinement process.
	Ex: bathy=Westport_DEM_2020.nc?z
	Ex: topo=Westport_DEM_2020.asc
	Default: None but input NECESSARY
	*/

	StaticForcingP<T> cf;
	/*Bottom friction coefficient map (associated to the chosen bottom friction model)
	Ex: cf=0.001;
	Ex: cf=bottom_friction.nc?bfc;
	Default: (see constant in parameters)
	*/

	std::vector<StaticForcingP<int>> targetadapt;

	std::vector<deformmap<T>> deform;
	/*Deform are maps to apply to both zs and zb; this is often co-seismic vertical deformation used to generate tsunami initial wave
	Here you can spread the deformation across a certain amount of time and apply it at any point in the model.
	Ex: deform = myDeform.nc?z_def,3.0,10.0;
	Ex: deform = *filename*, *time of initial rupture*, *rising time*;
	Default: None
	*/
	

	std::vector<River> rivers;
	/*The river is added as a vertical discharge on a chosen area (the user input consisting in a Time serie and a rectangular area definition).
	The whole cells containing the corners of the area will be included in the area, no horizontal velocity is applied.
	To add multiple rivers, just add different lines in the input file (one by river).
	Ex: river = Votualevu_R.txt,1867430,1867455,3914065,3914090;
	Ex: river = *Fluxfile*, *xstart*, *xend*, *ystart*, *yend*;
	Default: None
	*/

	bndparam left;
	/* 0:Wall (no slip); 1:neumann (zeros gradient) [Default]; 2:sealevel dirichlet; 3: Absorbing 1D 4: Absorbing 2D (not yet implemented)
	For type 2 and 3 boundary, a file need to be added to determine the vaules at the boundary. This file will consist in a first column containing time (with possibly variable time steps) and forcing values in the following columns (1 column of values corresponding to a constant value along the boundary, 2 columns correspond to values at boundary edges with linear evolution in between, n columns correspond to n regularly spaced values applied along the boundary)
	Ex: left = 0;
	Ex: left = leftBnd.txt,2;
	Default: 1
	*/

	bndparam right;
	/*Same as left boundary
	Ex: right = 0;
	Ex: right = rightBnd.txt,2;
	Default: 1
	*/

	bndparam top;
	/*Same as left boundary
	Ex: top = 0;
	Ex: top = topBnd.txt,2;
	Default: 1
	*/

	bndparam bot;
	/*Same as left boundary
	Ex: bot = 0;
	Ex: bot = botBnd.txt,2;
	Default: 1
	*/

	AOIinfo AOI;
	/*Area of interest polygon
	Ex: AOI=myarea.gmt;
	the iinput file is a text file with 2 column containing the cordinat of a closed polygon (last line==first line)
	Default: N/A
	*/
	
};











// End of global definition
#endif
