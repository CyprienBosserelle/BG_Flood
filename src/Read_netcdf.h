
#ifndef READNETCDF_H
#define READNETCDF_H

#include "General.h"
#include "Input.h"
#include "ReadInput.h"
#include "Write_txtlog.h"
#include "Write_netcdf.h"
#include "Util_CPU.h"
#include "GridManip.h"
#include "Forcing.h"



inline int nc_get_var_T(int ncid, int varid, float * &zb);
inline int nc_get_var_T(int ncid, int varid, double * &zb);

inline int nc_get_vara_T(int ncid, int varid, const size_t* startp, const size_t* countp, float * &zb);
inline int nc_get_vara_T(int ncid, int varid, const size_t* startp, const size_t* countp, double * &zb);

inline int nc_get_var1_T(int ncid, int varid, const size_t* startp, float * zsa);
inline int nc_get_var1_T(int ncid, int varid, const size_t* startp, double * zsa);

//template <class T> int readnczb(int nx, int ny, const std::string ncfile, T * &zb);
//int readnczb(int nx, int ny, const std::string ncfile, float*& zb);
//int readnczb(int nx, int ny, const std::string ncfile, double*& zb);
std::string checkncvarname(int ncid, std::string stringA, std::string stringB, std::string stringC, std::string stringD);

void readgridncsize(const std::string ncfilestr, const std::string varstr, int &nx, int &ny, int &nt, double &dx, double &xo, double &yo, double &to, double &xmax, double &ymax, double &tmax);
int readvarinfo(std::string filename, std::string Varname, size_t *&ddimU);
int readnctime(std::string filename, double * &time);
template <class T> int readncslev1(std::string filename, std::string varstr, size_t indx, size_t indy, size_t indt, bool checkhh, double eps, T * &zsa);
template <class T> int readvardata(std::string filename, std::string Varname, int step, T*& vardata);
//template <class T> int readhotstartfile(Param XParam, int * leftblk, int *rightblk, int * topblk, int* botblk, double * blockxo, double * blockyo, T * &zs, T * &zb, T * &hh, T *&uu, T * &vv);

void readWNDstep(forcingmap WNDUmap, forcingmap WNDVmap, int steptoread, float *&Uo, float *&Vo);
void readATMstep(forcingmap ATMPmap, int steptoread, float *&Po);
// End of global definition
#endif
