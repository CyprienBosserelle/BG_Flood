
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

	
};


struct AdaptP
{
	int *newlevel;
	int *availblk, * csumblk;
	int *invactive;
	bool * coarsen, *refine;

};




template <class T>
struct ExternalParam
{

};


struct bndTexP
{
	cudaArray* WLS;
	cudaArray* Uvel;
	cudaArray* Vvel;

	cudaChannelFormatDesc channelDescwls = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDescuvel = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDescvvel = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	texture<float, 2, cudaReadModeElementType> texWLS;
	texture<float, 2, cudaReadModeElementType> texUVEL;
	texture<float, 2, cudaReadModeElementType> texVVEL;
};

template <class T>
struct TexSetP
{
	cudaArray* CudArr;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	texture<float, 2, cudaReadModeElementType> tex;
};


template <class T>
struct TexForcingP
{
	//map forcing textures
	TexSetP<T> UWind;
	TexSetP<T> VWind;
	TexSetP<T> Patm;
	TexSetP<T> Rain;

	//bnd forcing textures
	bndTexP leftbnd;
	bndTexP rightbnd;
	bndTexP topbnd;
	bndTexP botbnd;


};




template <class T>
struct TimeP
{
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
	T* datmpdx, *datmpdy;

	//texture for boundary and map forcing
	//is this OK on CPU?
	TexForcingP<T> Tex;

	// 
	std::map<std::string, T *> OutputVar;

	//other output
	T* TSstore;//buffer for TS data so not to dave too often
	T* vort;
	EvolvingP<T> evmean;
	EvolvingP<T> evmax;

	//Block information
	BlockP<T> blocks;
};





// End of global definition
#endif
