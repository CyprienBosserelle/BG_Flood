
#ifndef ARRAYS_H
#define ARRAYS_H

#include "General.h"
#include "Input.h"


template <class T>
struct GradientsP
{
	T* dzsdx;
	T* dhdx;
	T* dudx;
	T* dvdx;

	T* dzsdy;
	T* dhdy;
	T* dudy;
	T* dvdy;

	T* dzbdx;
	T* dzbdy;
};


template <class T>
struct EvolvingP
{
	T* zs;
	T* h;
	T* u;
	T* v;
};

//subclass inheriting from EvolvingP for Mean/Max
template <class T>
struct EvolvingP_M : public EvolvingP<T>
{
	T* U;  //Norm of the velocity
	T* hU; //h*sqrt(u^2+v^2)
};

template <class T>
struct FluxP
{
	T* Su,* Sv;
	T* Fqux, * Fquy;
	T* Fqvx, * Fqvy;
	T* Fhu, * Fhv;
};

template <class T>
struct AdvanceP
{
	T* dh;
	T* dhu;
	T* dhv;
};


struct maskinfo 
{

	int nblk = 0; //number of blocks where this bnd applies

	int* blks; // array of block where bnd applies 
	// 8 digit binary where 1 is a mask and 0 is not a mask with the first digit represent the left bottom side the rest is clockwise (i.e.left-bot left-top, top-left, top-right, right-top, right-bot, bot-right, bot-left)
	int* side; // e.g. 11000000 for the entire left side being a mask
};

// outzone info used to actually write the nc files (one nc file by zone, the default zone is the full domain)
struct outzoneB 
{
	int nblk; //number of blocks concerned
	int* blk; // one zone will spread across multiple blocks (entire blocks containing a part of the area will be output)
	double xo, xmax, yo, ymax; // Real zone for output (because we output full blocks)(corner of cells, as Xparam.xo)
	std::string outname; // name for the output file (one for each zone)
	int maxlevel; // maximum level in the zone
	int minlevel; //minimum level in the zone
};


template <class T>
struct BlockP
{
	T* xo, *yo;
	int* BotLeft, *BotRight;
	int* TopLeft, *TopRight;
	int* LeftBot, *LeftTop;
	int* RightBot, *RightTop;

	int* level;
	int* active; // active blocks
	int* activeCell; //To apply forcings (rain) only on these

	maskinfo mask;
	
	std::vector<outzoneB> outZone;
};


struct AdaptP
{
	int *newlevel;
	int *availblk, * csumblk;
	int *invactive;
	bool * coarsen, *refine;

};




struct BndblockP
{
	int nblkriver, nblkTs, nbndblkleft, nbndblkright, nbndblktop, nbndblkbot;
	int* river;
	int* Tsout;
	//int * DrainSink;
	//int * DrainSource;
	//int * Bridges;

	int* left;
	int* right;
	int* top;
	int* bot;




};







template <class T>
struct TimeP
{
	T totaltime;
	T dt;
	T* dtmax;
	T* arrmax, *arrmin;
};

template <class T>
struct Model
{
	EvolvingP<T> evolv;
	EvolvingP<T> evolv_o;

	GradientsP<T> grad;
	FluxP<T> flux;
	AdvanceP<T> adv;
	
	//external forcing
	T* zb;
	T* cf;
	T* il;
	T* cl;

	//GroundWater elevation (due to the accumulation of water by infiltration during the simulation)
	T* hgw;
	
	// Used for external forcing too
	// May need a better placeholder
	T* Patm, *datmpdx, *datmpdy;

	TimeP<T> time;

	

	// 
	std::map<std::string, T *> OutputVarMap;

	//other output
	//std::vector< std::vector< Pointout > > TSallout;
	T* TSstore;//buffer for TS data so not to save to disk too often
	//T* vort;
	//T* U;
	EvolvingP_M<T> evmean;
	EvolvingP_M<T> evmax;
	T* wettime; //Inundation duration (h > 0.1)

	//Block information
	BlockP<T> blocks;

	AdaptP adapt;

	BndblockP bndblk;


	

};

// structure of useful variable for runing the main loop
template <class T>
struct Loop
{
	double nextoutputtime;
	double dt;
	double dtmax;
	double totaltime;
	// Needed to average mean varable for output
	int nstep = 0;
	//useful for calculating avg timestep
	int nstepout = 0;

	// usefull for Time series output
	int nTSsteps = 0;
	std::vector< std::vector< Pointout > > TSAllout;

	int windstep = 1;
	int atmpstep = 1;
	int rainstep = 1;

	bool winduniform;
	bool rainuniform;
	bool atmpuniform;

	T uwinduni = T(0.0);
	T vwinduni = T(0.0);
	T atmpuni;
	T rainuni = T(0.0);

	// CUDA specific stuff

	dim3 blockDim;// (16, 16, 1);
	dim3 gridDim;

	const int num_streams = 4;

	cudaStream_t streams[4];

	T epsilon;
	T hugeposval;
	T hugenegval;

};



// End of global definition
#endif
