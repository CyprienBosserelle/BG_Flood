
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

// Flexible definition of time outputs
class T_output {
public: 
	double init = NAN;
	double tstep = NAN;
	double end = NAN;
	std::vector<std::string> inputstr;
	std::vector<double> val;
};

// Special output zones for nc files, informatin given by the user
class outzoneP {
public:
	//std::vector<int> blocks; // one zone will spread across multiple blocks (entire blocks containing a part of the area will be output)
	double xstart, xend, ystart, yend; // definition of the zone needed for special nc output (rectangular zone) by the user
	//double xo, xmax, yo, ymax; // Real zone for output (because we output full blocks)
	std::string outname; // name for the output file (one for each zone)
	T_output Toutput; // time for outputs for the zone
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
	double to, tmax;
	std::string Riverflowfile; // river flow input time[s] flow in m3/s
	std::vector<Flowin> flowinput; // vector to store the data of the river flow input file
	
};

class Culvert {
public:
	int type = 0; //Type of culvert (0: pump, 1: one-way culvert, trap-door, 2: two-way classic culvert)
	int shape = 0; //Shape of the culvert (0: rectangular, 1: circular)
	double x1, x2, y1, y2; // location of the input and outputs (or 2 points defining the culvert)
	double width = 1.0; // Diameter (for circular shape) or width (for rectangular shape) of the culvert in m.
	double fac = 1.0; // Multiplier for accounting for multiple barrels (e.g. fac = 2.0) or partially blocked (e.g. fac=0.8)
	double height = 0.0; // height for rectangular shaped culverts, ignored for circular shaped ones. 
	double length; //Length of the culvert in m.
	int ix1, iy1, block1, ix2, iy2, block2; // start and end of the culvert cells (dx_local).
	double dx1, dx2;// start and end of the culvert cells (dx_local).
	double Qmax = 1.0; //Maximum discharge for the culvert in m3/s.
	double n = 0.013; //Manning roughness coefficient inside the culvert (default for concrete)
	double k_ex = 0.7; //Exit loss coefficient (default for sudden expansion of flow, such as in a typical culvert, down to 0.3 (minimum) if transition is less abrupt)
	double k_en = 0.5; //Entrance loss coefficient (default for sharpedged culvert entrance with no rounding, 0.2 appropriated if well rounded entrance)
	double C_d = -999.0; //Discharge coefficient for the submerged culvert (default of 1.0 for circular, 0.62 for rectangular)
	double zb1 = -999.0; //Bottom elevation of the inlet / outlet given by the user or zb.
	double zb2 = -999.0; //Bottom elevation of the inlet / outlet given by the user or zb.
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
	double dy = 0.0;
	double grdalpha=0.0;
	double denanval = NAN;
	bool flipxx = false;
	bool flipyy = false;
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

class Vertex {
public:
	double x, y;
};

class Polygon {
public:
	double xmin, xmax, ymin, ymax;
	std::vector<Vertex> vertices;
};

// End of global definition
#endif
