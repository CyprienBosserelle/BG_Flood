
#ifndef READFORCING_H
#define READFORCING_H

#include "General.h"
#include "Input.h"
#include "Param.h"
#include "Write_txt.h"
#include "read_netcdf.h"
#include "Forcing.h"
#include "Util_CPU.h"

template <class T> void readforcing(Param& XParam, Forcing<T>& XForcing);

std::vector<SLTS> readbndfile(std::string filename, Param XParam, int side);
std::vector<SLTS> readWLfile(std::string WLfilename);
std::vector<SLTS> readNestfile(std::string ncfile, int hor, double eps, double bndxo, double bndxmax, double bndy);
std::vector<Flowin> readFlowfile(std::string Flowfilename);
std::vector<Windin> readINfileUNI(std::string filename);
std::vector<Windin> readWNDfileUNI(std::string filename, double grdalpha);

inputmap readcfmaphead(inputmap Roughmap);
void readmapdata(inputmap Roughmap, float * &cfmapinput);
forcingmap readforcingmaphead(forcingmap Fmap);
inputmap readBathyhead(inputmap BathyParam);
void readbathyHeadMD(std::string filename, int &nx, int &ny, double &dx, double &grdalpha);
extern "C" void readbathyMD(std::string filename, float *&zb);
extern "C" void readXBbathy(std::string filename, int nx, int ny, float *&zb);


void readbathyASCHead(std::string filename, int &nx, int &ny, double &dx, double &xo, double &yo, double &grdalpha);
void readbathyASCzb(std::string filename, int nx, int ny, float* &zb);

// End of global definition
#endif
