
#ifndef INPUT_H
#define INPUT_H

#include "General.h"

class TSnode {
public:
	int i, j, block;
	double x, y;
};

class Flowin {
public:
	double time, q;
};

class Mapparam {
public:

};

class River {
public:
	std::vector<int> i, j, block; // one river can spring across multiple cells
	double disarea; // discharge area
	double xstart,xend, ystart,yend; // location of the discharge as a rectangle
	std::string Riverflowfile; // river flow input time[s] flow in m3/s
	std::vector<Flowin> flowinput; // vector to store the data of the river flow input file
	
};

class inputmap {
public:
	int nx = 0;
	int ny= 0;
	double xo = 0.0;
	double yo = 0.0;
	double xmax = 0.0;
	double ymax = 0.0;
	double dx = 0.0;
	double grdalpha=0.0;
	std::string inputfile;
};

class SLTS {
public:
	double time;
	std::vector<double> wlevs;
	std::vector<double> uuvel;
	std::vector<double> vvvel;

};

class Windin {
public:
	double time;
	double wspeed;
	double wdirection;
	double uwind;
	double vwind;


};

class forcingmap {
public:
	int nx, ny;
	int nt;
	int uniform = 0;
	double to, tmax;
	double xo, yo;
	double xmax, ymax;
	double dx;
	double dt;
	std::string inputfile;
	std::vector<Windin> data; // only used if uniform forcing

};

class deformmap{
	//Deform are maps to applie to both zs and zb; this is often co-seismic vertical deformation used to generate tsunami initial wave
	// Here you can spread the deformation across a certain amount of time and apply it at any point in the model
public:
	inputmap grid;
	double startime = 0.0;
	double duration = 0.0;
	

};



class bndparam {
public:
	std::vector<SLTS> data;
	bool on = false;
	int type = 1; // 0:Wall (no slip); 1:neumann (zeros gredient) [Default]; 2:sealevel dirichlet; 3: Absorbing 1D 4: Absorbing 2D (not yet implemented)
	std::string inputfile;
	int nblk = 0; //number of blocks where this bnd applies
	int side = 0; // 0: top bnd, 1: rightbnd, 2: bot bnd, 3, Left bnd
};


class Pointout {
public:
	double time, zs, hh, uu,vv;
};


// End of global definition
#endif
