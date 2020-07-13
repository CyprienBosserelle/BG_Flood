
#ifndef READINPUT_H
#define READINPUT_H

#include "General.h"
#include "Input.h"
#include "Param.h"
#include "Write_txt.h"


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
double interptime(double next, double prev, double timenext, double time);
void readbathyASCHead(std::string filename, int &nx, int &ny, double &dx, double &xo, double &yo, double &grdalpha);
void readbathyASCzb(std::string filename, int nx, int ny, float* &zb);
double BilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y);
double BarycentricInterpolation(double q1, double x1, double y1, double q2, double x2, double y2, double q3, double x3, double y3, double x, double y);

// End of global definition
#endif
