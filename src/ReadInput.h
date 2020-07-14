
#ifndef READINPUT_H
#define READINPUT_H

#include "General.h"
#include "Input.h"
#include "Param.h"
#include "Write_txt.h"
#include "read_netcdf.h"


Param Readparamfile(Param XParam);
std::vector<SLTS> readbndfile(std::string filename, Param XParam, int side);
std::vector<SLTS> readWLfile(std::string WLfilename);
std::vector<SLTS> readNestfile(std::string ncfile, int hor, double eps, double bndxo, double bndxmax, double bndy);
std::vector<Flowin> readFlowfile(std::string Flowfilename);
std::vector<Windin> readINfileUNI(std::string filename);
std::vector<Windin> readWNDfileUNI(std::string filename, double grdalpha);
Param readparamstr(std::string line, Param param);
Param checkparamsanity(Param XParam);
double setendtime(Param XParam);
std::string findparameter(std::string parameterstr, std::string line);
void split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);
std::string trim(const std::string& str, const std::string& whitespace);
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
