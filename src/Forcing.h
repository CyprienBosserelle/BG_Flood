
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
	int type = 1; // 0:Wall (no slip); 1:neumann (zeros gredient) [Default]; 2:sealevel dirichlet; 3: Absorbing 1D 4: Absorbing 2D (not yet implemented)
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
	DynForcingP<T> Atmp;

	std::vector<StaticForcingP<T>> Bathy; //Should be a vector at some point
	StaticForcingP<T> cf;

	std::vector<StaticForcingP<int>> targetadapt;

	std::vector<deformmap<T>> deform;

	std::vector<River> rivers;

	bndparam right;
	bndparam left;
	bndparam top;
	bndparam bot;

	
	
};











// End of global definition
#endif
