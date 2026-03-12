
#ifndef ARRAYS_H
#define ARRAYS_H

#include "General.h"
#include "Input.h"


/**
 * @brief Structure holding gradient arrays for physical variables.
 * @tparam T Data type
 */
template <class T>
struct GradientsP
{
	/** Surface elevation gradient in x-direction */
	T* dzsdx;
	/** Water depth gradient in x-direction */
	T* dhdx;
	/** Velocity u gradient in x-direction */
	T* dudx;
	/** Velocity v gradient in x-direction */
	T* dvdx;

	/** Surface elevation gradient in y-direction */
	T* dzsdy;
	/** Water depth gradient in y-direction */
	T* dhdy;
	/** Velocity u gradient in y-direction */
	T* dudy;
	/** Velocity v gradient in y-direction */
	T* dvdy;

	/** Bed elevation gradient in x-direction */
	T* dzbdx;
	/** Bed elevation gradient in y-direction */
	T* dzbdy;
};

/**
 * @brief Structure holding gradient arrays (no z relative variables).
 * @tparam T Data type
 */
template <class T>
struct GradientsMLP
{
	/** Water depth gradient in x-direction */
	T* dhdx;
	/** Velocity u gradient in x-direction */
	T* dudx;
	/** Velocity v gradient in x-direction */
	T* dvdx;
	/** Water depth gradient in y-direction */
	T* dhdy;
	/** Velocity u gradient in y-direction */
	T* dudy;
	/** Velocity v gradient in y-direction */
	T* dvdy;
};


/**
 * @brief Structure holding evolving physical variables.
 * @tparam T Data type
 */
template <class T>
struct EvolvingP
{
	/** Surface elevation */
	T* zs;
	/** Water depth */
	T* h;
	/** X velocity component u */
	T* u;
	/** Y velocity component v */
	T* v;
};

/**
 * @brief Structure holding evolving variables (no z relative variables).
 * @tparam T Data type
 */
template <class T>
struct EvolvingMLP
{
	/** Water depth */
	T* h;
	/** Velocity u */
	T* u;
	/** Velocity v */
	T* v;
};

//subclass inheriting from EvolvingP for Mean/Max
/**
 * @brief Structure for mean/max evolving variables, inherits from EvolvingP.
 * @tparam T Data type
 */
template <class T>
struct EvolvingP_M : public EvolvingP<T>
{
	/** Norm of the velocity */
	T* U;
	/** Product of water depth and velocity norm */
	T* hU;
};

/**
 * @brief Structure holding flux variables for advection.
 * @tparam T Data type
 */
template <class T>
struct FluxP
{
	/** Source term for u */
	T* Su;
	/** Source term for v */
	T* Sv;
	/** Flux of u in x-direction */
	T* Fqux;
	/** Flux of u in y-direction */
	T* Fquy;
	/** Flux of v in x-direction */
	T* Fqvx;
	/** Flux of v in y-direction */
	T* Fqvy;
	/** Flux of h in u-direction */
	T* Fhu;
	/** Flux of h in v-direction */
	T* Fhv;
};

/**
 * @brief Structure holding flux variables (no z relative variables).
 * @tparam T Data type
 */
template <class T>
struct FluxMLP
{
	/** Water depth flux in u-direction */
	T* hu;
	/** Water depth flux in v-direction */
	T* hv;
	/** h*f flux in u-direction */
	T* hfu;
	/** h*f flux in v-direction */
	T* hfv;
	/** h*a flux in u-direction */
	T* hau;
	/** h*a flux in v-direction */
	T* hav;
	/** Flux of u in x-direction */
	T* Fux;
	/** Flux of v in y-direction */
	T* Fvy;
	/** Flux of u in y-direction */
	T* Fuy;
	/** Flux of v in x-direction */
	T* Fvx;
};

/**
 * @brief Structure holding advance variables for time stepping.
 * @tparam T Data type
 */
template <class T>
struct AdvanceP
{
	/** Change in water depth */
	T* dh;
	/** Change in velocity u */
	T* dhu;
	/** Change in velocity v */
	T* dhv;
};


struct outP
{
	float* z;
	short* z_s;
	int level;
	double xmin, xmax, ymin, ymax;
};


struct maskinfo 
{

	int nblk = 0; //number of blocks where this bnd applies

	int* blks; // array of block where bnd applies 
	// 8 digit binary where 1 is a mask and 0 is not a mask with the first digit represent the left bottom side the rest is clockwise (i.e.left-bot left-top, top-left, top-right, right-top, right-bot, bot-right, bot-left)
	int* side; // e.g. 11000000 for the entire left side being a mask

	int type = 0;
};

template <class T>
struct RiverInfo
{
	int nbir;
	int nburmax; // size of (max number of) unique block with rivers  
	int nribmax; // size of (max number of) rivers in one block
	int* Xbidir; // array of block id for each river size(nburmax,nribmax)
	int* Xridib; // array of river id in each block size(nburmax,nribmax)
	T* xstart;
	T* xend;
	T* ystart;
	T *yend;
	T* qnow; // qnow is a pin mapped and so both pointers are needed here
	T* qnow_g; // this simplify the code later

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
	std::vector<double> OutputT; //Next time for the output of this zone
	int index_next_OutputT = 0; //Index of next time output
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



template <class T>
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

	RiverInfo<T> Riverinfo;


};

struct RiverBlk
{
	std::vector<int> block;
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
	FluxMLP<T> fluxml;
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
	std::map<std::string, std::string> Outvarlongname;
	std::map<std::string, std::string> Outvarstdname;
	std::map<std::string, std::string> Outvarunits;
	std::vector<double> OutputT;

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

	BndblockP<T> bndblk;


	

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
	// Needed to identify next output time
	int indNextoutputtime = 0;

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
