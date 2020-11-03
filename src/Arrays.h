
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
};


template <class T>
struct EvolvingP
{
	T* zs;
	T* h;
	T* u;
	T* v;
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

template <class T>
struct BlockP
{
	T* xo, *yo;
	int* BotLeft, *BotRight;
	int* TopLeft, *TopRight;
	int* LeftBot, *LeftTop;
	int* RightBot, *RightTop;

	int* level;
	int* active;

	maskinfo mask;
	
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
	
	// Used for external forcing too
	// May need a better placeholder
	T* Patm, *datmpdx, *datmpdy;

	TimeP<T> time;

	

	// 
	std::map<std::string, T *> OutputVarMap;

	//other output
	//std::vector< std::vector< Pointout > > TSallout;
	T* TSstore;//buffer for TS data so not to save to disk too often
	T* vort;
	EvolvingP<T> evmean;
	EvolvingP<T> evmax;

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
