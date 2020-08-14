
#ifndef READINPUT_H
#define READINPUT_H

#include "General.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "Forcing.h"
#include "Util_CPU.h"


template <class T> T readfileinfo(std::string input, T outinfo);

void Readparamfile(Param& XParam, Forcing<float>& XForcing);

template <class T> Forcing<T> readparamstr(std::string line, Forcing<T> forcing);

Param readparamstr(std::string line, Param param);

template <class T>Forcing<T> readparamstr(std::string line, Forcing<T> forcing);
void checkparamsanity(Param& XParam, Forcing<float>& XForcing);
double setendtime(Param XParam,Forcing<float> XForcing);
std::string findparameter(std::string parameterstr, std::string line);
void split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);
std::string trim(const std::string& str, const std::string& whitespace);



// End of global definition
#endif
