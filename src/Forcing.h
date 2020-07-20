
#ifndef FORCING_H
#define FORCING_H

#include "General.h"
#include "Input.h"


class forcingmap : public inputmap {
public:
	
	int nt;
	int uniform = 0;
	double to, tmax;
	double dt;
	std::string inputfile;
	std::vector<Windin> unidata; // only used if uniform forcing

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

};


template <class T>
struct DynForcingP: public forcingmap
{
	T *now;
	T *before, *after;
	T* val; // useful for reading form file
	//Add map here?

};

template <class T>
struct StaticForcingP : public inputmap
{
	T *val;
	

};

template <class T>
struct Forcing
{
	DynForcingP<T> UWind;
	DynForcingP<T> VWind;
	DynForcingP<T> Rain;
	DynForcingP<T> Atmp;

	StaticForcingP<T> Bathy; //Should be a vector at some point
	StaticForcingP<T> cf;

	std::vector<deformmap<T>> deform;

	std::vector<River> rivers;

};











// End of global definition
#endif
