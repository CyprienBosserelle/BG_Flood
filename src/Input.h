
#ifndef INPUT_H
#define INPUT_H

#include "General.h"

// Timeseries output
class TSoutnode {
public:
	int i, j, block;
	double x, y;
	std::string outname;
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
	std::string extension;
	std::string varname;
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


// used as vector class to store Time series outputs
class Pointout {
public:
	double time, zs, h, u,v;
};


// End of global definition
#endif
