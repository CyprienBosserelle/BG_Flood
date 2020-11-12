
#ifndef READFORCING_H
#define READFORCING_H

#include "General.h"
#include "Input.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "read_netcdf.h"
#include "Forcing.h"
#include "Util_CPU.h"
#include "Setup_GPU.h"

template<class T> void readforcing(Param& XParam, Forcing<T> & XForcing);

std::vector<SLTS> readbndfile(std::string filename, Param XParam, int side);
std::vector<SLTS> readWLfile(std::string WLfilename);

std::vector<SLTS> readNestfile(std::string ncfile, std::string varname, int hor, double eps, double bndxo, double bndxmax, double bndy);

std::vector<Flowin> readFlowfile(std::string Flowfilename);
std::vector<Windin> readINfileUNI(std::string filename);
std::vector<Windin> readWNDfileUNI(std::string filename, double grdalpha);

void readDynforcing(bool gpgpu,double totaltime, DynForcingP<float>& Dforcing);

template<class T> T readforcinghead(T Fmap);
//template<class T> T readBathyhead(T BathyParam);
template<class T> void readstaticforcing(T& Sforcing);
template <class T> void readstaticforcing(int step, T& Sforcing);

template <class T> void readforcingdata(int step, T forcing);
void readforcingdata(double totaltime, DynForcingP<float>& forcing);
void readbathyHeadMD(std::string filename, int &nx, int &ny, double &dx, double &grdalpha);
template <class T> void readbathyMD(std::string filename, T*& zb);
template <class T> void readXBbathy(std::string filename, int nx, int ny, T*& zb);


void readbathyASCHead(std::string filename, int &nx, int &ny, double &dx, double &xo, double &yo, double &grdalpha);
template <class T> void readbathyASCzb(std::string filename, int nx, int ny, T*& zb);

template <class T> void InterpstepCPU(int nx, int ny, int hdstep, float totaltime, float hddt, T*& Ux, T* Uo, T* Un);

// End of global definition
#endif
