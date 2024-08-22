//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//Copyright (C) 2018 Bosserelle                                                 //
//                                                                              //
//This program is free software: you can redistribute it and/or modify          //
//it under the terms of the GNU General Public License as published by          //
//the Free Software Foundation.                                                 //
//                                                                              //
//This program is distributed in the hope that it will be useful,               //
//but WITHOUT ANY WARRANTY; without even the implied warranty of                //    
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                 //
//GNU General Public License for more details.                                  //
//                                                                              //
//You should have received a copy of the GNU General Public License             //
//along with this program.  If not, see <http://www.gnu.org/licenses/>.         //
//////////////////////////////////////////////////////////////////////////////////

#include "ReadInput.h"

// Collection of functions to read input to the model


/*! \fn T readfileinfo(std::string input,T outinfo)
* convert file name into name and extension
* This is used for various input classes
*
* template inputmap readfileinfo<inputmap>(std::string input, inputmap outinfo);
* template forcingmap readfileinfo<forcingmap>(std::string input, forcingmap outinfo);
* template StaticForcingP<float> readfileinfo<StaticForcingP<float>>(std::string input, StaticForcingP<float> outinfo);
* template DynForcingP<float> readfileinfo<DynForcingP<float>>(std::string input, DynForcingP<float> outinfo);
* template deformmap<float> readfileinfo<deformmap<float>>(std::string input, deformmap<float> outinfo);
*/
template <class T> T readfileinfo(std::string input, T outinfo)
{
	// Outinfo is based on an inputmap (or it's sub classes)

	//filename include the file extension

	std::vector<std::string> extvec = split(input, '.');

	//outinfo.inputfile = extvec.front();

	std::vector<std::string> nameelements, filename;
	//
	nameelements = split(extvec.back(), '?');

	filename = split(input, '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		outinfo.extension = nameelements[0];
		outinfo.varname = nameelements.back();

	}
	else
	{
		outinfo.extension = extvec.back();
		outinfo.varname = "z";
	}

	//Reconstruct filename with extension but without varname
	//outinfo.inputfile = extvec.front() + "." + outinfo.extension;
	outinfo.inputfile = filename.front();

	return outinfo;
}

template inputmap readfileinfo<inputmap>(std::string input, inputmap outinfo);
template forcingmap readfileinfo<forcingmap>(std::string input, forcingmap outinfo);
template StaticForcingP<float> readfileinfo<StaticForcingP<float>>(std::string input, StaticForcingP<float> outinfo);
template DynForcingP<float> readfileinfo<DynForcingP<float>>(std::string input, DynForcingP<float> outinfo);
template deformmap<float> readfileinfo<deformmap<float>>(std::string input, deformmap<float> outinfo);




/*! \fn void Readparamfile(Param &XParam, Forcing<float> & XForcing)
* Open the BG_param.txt file and read the parameters
* save the parameter in the Param class and or Forcing class.
*/
void Readparamfile(Param& XParam, Forcing<float>& XForcing, std::string Paramfile)
{
	//
	log("\nReading parameter file: " + Paramfile + " ...");
	//std::ifstream fs("BG_param.txt");
	std::ifstream fs(Paramfile);

	if (fs.fail()) {
		//std::cerr << "BG_param.txt file could not be opened" << std::endl;
		log("ERROR: BG_param.txt file could not be opened...use this log file to create a file named BG_param.txt");
		SaveParamtolog(XParam);

		exit(1);

	}
	else
	{
		// Read and interpret each line of the BG_param.txt
		std::string line;
		while (std::getline(fs, line))
		{

			//Get param or skip empty lines
			if (!line.empty() && line.substr(0, 1).compare("#") != 0)
			{
				XParam = readparamstr(line, XParam);
				XForcing = readparamstr(line, XForcing);

				//std::cout << line << std::endl;
			}

		}
		fs.close();


	}

}





/*! \fn Param readparamstr(std::string line, Param param)
* Read BG_param.txt line and convert parameter to the righ parameter in the class
* retrun an updated Param class
*/
Param readparamstr(std::string line, Param param)
{


	std::string parameterstr, parametervalue;
	std::vector<std::string> paramvec;
	///////////////////////////////////////////////////////
	// General parameters
	//

	parameterstr = "test";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.test = std::stoi(parametervalue);
	}

	paramvec = { "GPUDEVICE","gpu" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.GPUDEVICE = std::stoi(parametervalue);
	}

	parameterstr = "doubleprecision";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.doubleprecision = std::stoi(parametervalue);
	}

	parameterstr = "engine";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> buttingerstr = { "b","butt","buttinger","1" };
		std::size_t found;
		bool foo = false;
		for (int ii = 0; ii < buttingerstr.size(); ii++)
		{
			found = case_insensitive_compare(parametervalue, buttingerstr[ii]);// it needs to strictly compare
			if (found == 0)
			{
				param.engine = 1;
				foo = true;
			}

		}
		if (!foo)
		{
			param.engine = 2;
		}
	}
	///////////////////////////////////////////////////////
	// Adaptation
	//
	parameterstr = "maxlevel";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.maxlevel = std::stoi(parametervalue);
	}

	parameterstr = "minlevel";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.minlevel = std::stoi(parametervalue);
	}

	parameterstr = "initlevel";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.initlevel = std::stoi(parametervalue);
	}

	paramvec = { "adaptmaxiteration","maxiterationadapt" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.adaptmaxiteration = std::stoi(parametervalue);
	}

	parameterstr = "conserveElevation";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.conserveElevation = readparambool(parametervalue, param.conserveElevation);
	}

	paramvec = { "wetdryfix","reminstab","fixinstab" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{

		param.wetdryfix = readparambool(parametervalue, param.wetdryfix);

	}


	parameterstr = "membuffer";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.membuffer = std::stod(parametervalue);
	}

	///////////////////////////////////////////////////////
	// Flow parameters
	//
	parameterstr = "eps";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.eps = std::stod(parametervalue);
	}

	paramvec = { "cf","roughness","cfmap" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		if (std::any_of(parametervalue.begin(), parametervalue.end(), ::isalpha) == false) //(std::isdigit(parametervalue[0]) == true)
		{
			param.cf = std::stod(parametervalue);
		}
	}

	paramvec = { "il","Rain_il","initialloss" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		if (std::any_of(parametervalue.begin(), parametervalue.end(), ::isalpha) == false) //(std::isdigit(parametervalue[0]) == true)
		{
			param.il = std::stod(parametervalue);
			param.infiltration = true;
		}
	}

	paramvec = { "cl","Rain_cl","continuousloss" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		if (std::any_of(parametervalue.begin(), parametervalue.end(), ::isalpha) == false) //(std::isdigit(parametervalue[0]) == true)
		{
			param.cl = std::stod(parametervalue);
			param.infiltration = true;
		}
	}

	paramvec = { "VelThreshold","vthresh","vmax","velmax" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.VelThreshold = std::stod(parametervalue);
	}

	parameterstr = "Cd";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Cd = std::stod(parametervalue);
	}

	parameterstr = "Pa2m";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Pa2m = std::stod(parametervalue);
	}

	parameterstr = "Paref";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Paref = std::stod(parametervalue);
	}

	parameterstr = "mask";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.mask = std::stod(parametervalue);
	}

	///////////////////////////////////////////////////////
	// Timekeeping parameters
	//
	parameterstr = "dt";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.dt = std::stod(parametervalue);

	}

	parameterstr = "dtmin";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.dtmin = std::stod(parametervalue);

	}
	parameterstr = "bndtaper";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.bndtaper = std::stod(parametervalue);

	}

	parameterstr = "CFL";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.CFL = std::stod(parametervalue);

	}
	parameterstr = "theta";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.theta = std::stod(parametervalue);

	}

	paramvec = { "outputtimestep","outtimestep","outputstep" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.outputtimestep = std::stod(parametervalue);

	}

	paramvec = { "endtime", "stoptime", "end", "stop","end_time","stop_time" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.endtime = readinputtimetxt(parametervalue, param.reftime);

	}

	paramvec = { "totaltime","inittime","starttime", "start_time", "init_time", "start", "init" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		//param.totaltime = std::stod(parametervalue);
		param.totaltime = readinputtimetxt(parametervalue, param.reftime);

	}

	parameterstr = "dtinit";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.dtinit = std::stod(parametervalue);

	}

	paramvec = { "reftime","referencetime","timeref" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		if (param.reftime.empty())
		{
			param.reftime = parametervalue;
		}

	}

	///////////////////////////////////////////////////////
	// Input and output files
	//

	parameterstr = "outfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.outfile = parametervalue;

	}


	// Below is a bit more complex than usual because more than 1 node can be outputed as a timeseries
	paramvec = { "TSnodesout","TSOutput" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		TSoutnode node;
		std::vector<std::string> nodeitems = split(parametervalue, ',');
		if (nodeitems.size() >= 3)
		{
			node.outname = nodeitems[0];
			node.x = std::stod(nodeitems[1]);
			node.y = std::stod(nodeitems[2]);

			param.TSnodesout.push_back(node);
		}
		else
		{
			std::cerr << "Node input failed there should be 3 arguments (comma separated) when inputing a outout node: TSOutput = filename, xvalue, yvalue; see log file for details" << std::endl;

			log("Node input failed there should be 3 arguments (comma separated) when inputing a outout node: TSOutput = filename, xvalue, yvalue; see log file for details. Input was: " + parametervalue);

		}

	}



	//outvars
	parameterstr = "outvars";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> vars = split(parametervalue, ',');
		for (int nv = 0; nv < vars.size(); nv++)
		{
			//Verify that the variable name makes sense?
			//Need to add more here

			std::vector<std::string> SupportedVarNames = { "zb","zs","u","v","h","hmean","zsmean","umean","vmean","hUmean","Umean","hmax","zsmax","umax","vmax","hUmax","Umax","twet","dhdx","dhdy","dzsdx","dzsdy","dzbdx","dzbdy","dudx","dudy","dvdx","dvdy","Fhu","Fhv","Fqux","Fqvy","Fquy","Fqvx","Su","Sv","dh","dhu","dhv","cf","Patm","datmpdx","datmpdy","il","cl","hgw" };

			std::string vvar = trim(vars[nv], " ");
			for (int isup = 0; isup < SupportedVarNames.size(); isup++)
			{

				//std::cout << "..." << vvar << "..." << std::endl;
				if (vvar.compare(SupportedVarNames[isup]) == 0)
				{
					param.outvars.push_back(vvar);
					break;
				}

			}

			param.outmean = (vvar.compare("hmean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("zsmean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("umean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("vmean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("Umean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("hUmean") == 0) ? true : param.outmean;

			param.outmax = (vvar.compare("hmax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("zsmax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("umax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("vmax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("Umax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("hUmax") == 0) ? true : param.outmax;

			param.outtwet = (vvar.compare("twet") == 0) ? true : param.outtwet;

			//param.outvort = (vvar.compare("vort") == 0) ? true : param.outvort;
			//param.outU = (vvar.compare("U") == 0) ? true : param.outU;
		}



	}


	// Same as for TSnodesout, the same key word can be used for different zones Output
	parameterstr = "outzone";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		outzoneP zone;
		std::vector<std::string> zoneitems = split(parametervalue, ',');
		if (zoneitems.size() >= 5)
		{
			zone.outname = zoneitems[0];
			zone.xstart = std::stod(zoneitems[1]);
			zone.xend = std::stod(zoneitems[2]);
			zone.ystart = std::stod(zoneitems[3]);
			zone.yend = std::stod(zoneitems[4]);
		}
		if (zoneitems.size() > 5)
		{
			std::vector<std::string> Toutputpar_vect = split_full(zoneitems[5], ':');
			if (Toutputpar_vect.size() == 3)
			{
				double init, tstep, end;
				double tiny = 0.000001;
				if (!Toutputpar_vect[0].empty()) {
					init = std::stod(Toutputpar_vect[0]);
				}
				if (!Toutputpar_vect[1].empty()) {
					tstep = std::max(std::stod(Toutputpar_vect[1]),tiny);
				}
				if (!Toutputpar_vect[2].empty()) {
					end = std::stod(Toutputpar_vect[2]);
				}

				int nstep = (end - init) / tstep + 1;

				for (int k = 0; k < nstep; k++)
				{
					zone.Toutput.val.push_back(std::min(init + tstep * k,end));
				}

			}
			else if (Toutputpar_vect.size() > 1)
			{
				//Failed: Toutput must be exactly 3 values, separated by ":" for a vector shape, in virst position. "t_init:t_step:t_end" (with possible empty values as "t_init:t_setps: " to use the last time steps as t_end;
				std::cerr << "Failed: Toutput must be exactly 3 values, separated by ':' for a vector shape, in first position. 't_init : t_step : t_end' (with possible empty values as 't_init : t_setps : ' to use the last time steps as t_end; see log file for details" << std::endl;

				log("Failed: Toutput must be exactly 3 values, separated by ':' for a vector shape, in virst position. 't_init : t_step : t_end' (with possible empty values as 't_init : t_setps : ' to use the last time steps as t_end;");
				log(parametervalue);
			}
			else { //only values
				zone.Toutput.val.push_back(std::stod(Toutputpar_vect[0]));
			}
			if (zoneitems.size() > 6) //vector + values
			{
				for (int ii = 6; ii < zoneitems.size(); ii++)
				{
					zone.Toutput.val.push_back(std::stod(zoneitems[ii]));
				}
			}
		}
		else if (zoneitems.size() == 5)//No time input in the zone area
		{
			zone.Toutput = param.Toutput;
		}
		else
		{
			std::cerr << "Zone input failed there should be at least 5 arguments (comma separated) when inputing a outout zone: outzone = filename, xstart, xend, ystart, yend; see log file for details" << std::endl;
			log("Node input failed there should be at least 5 arguments (comma separated) when inputing a outout zone: outzone = filename, xstart, xend, ystart, yend; see log file for details (with possibly some time inputs after). Input was: " + parametervalue);
		}
		param.outzone.push_back(zone);
	}

	parameterstr = "resetmax";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		if (std::stoi(parametervalue) == 1)
		{
			param.resetmax = true;
		}
	}

	// WARNING FOR DEBUGGING PURPOSE ONLY
	// For debugging one can shift the output by 1 or -1 in the i and j direction.
	// this will save the value in the halo to the output file allowing debugging of values there.
	parameterstr = "outishift";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.outishift = std::stoi(parametervalue);
	}
	parameterstr = "outjshift";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.outjshift = std::stoi(parametervalue);
	}

	////////////////////////////////////////////////////////////////


	parameterstr = "nx";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.nx = std::stoi(parametervalue);
	}

	parameterstr = "ny";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.ny = std::stoi(parametervalue);
	}

	parameterstr = "dx";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.dx = std::stod(parametervalue);
	}

	parameterstr = "grdalpha";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.grdalpha = std::stod(parametervalue);
	}

	paramvec = { "xo","xmin" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.xo = std::stod(parametervalue);
	}

	paramvec = { "yo","ymin" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.yo = std::stod(parametervalue);
	}

	parameterstr = "xmax";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.xmax = std::stod(parametervalue);
	}

	parameterstr = "ymax";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.ymax = std::stod(parametervalue);
	}

	parameterstr = "g";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.g = std::stod(parametervalue);

	}

	parameterstr = "rho";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.rho = std::stod(parametervalue);
	}

	parameterstr = "smallnc";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.smallnc = std::stoi(parametervalue);
	}
	parameterstr = "scalefactor";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.scalefactor = std::stof(parametervalue);
	}
	parameterstr = "addoffset";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.addoffset = std::stof(parametervalue);
	}
	parameterstr = "posdown";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.posdown = std::stoi(parametervalue);
	}

#ifdef USE_CATALYST
	parameterstr = "use_catalyst";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.use_catalyst = std::stoi(parametervalue);
	}
	parameterstr = "catalyst_python_pipeline";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.catalyst_python_pipeline = std::stoi(parametervalue);
	}
	parameterstr = "vtk_output_frequency";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.vtk_output_frequency = std::stoi(parametervalue);
	}
	parameterstr = "vtk_output_time_interval";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.vtk_output_time_interval = std::stod(parametervalue);
	}
	parameterstr = "vtk_outputfile_root";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.vtk_outputfile_root = parametervalue;
	}
	parameterstr = "python_pipeline";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.python_pipeline = parametervalue;
	}
#endif

	paramvec = { "zsinit", "initzs" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.zsinit = std::stod(parametervalue);
	}

	parameterstr = "zsoffset";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.zsoffset = std::stod(parametervalue);
	}
	paramvec = { "rainbnd", "rainonbnd" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.rainbnd = readparambool(parametervalue, param.rainbnd);

	}


	parameterstr = "hotstartfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.hotstartfile = parametervalue;

	}

	parameterstr = "hotstep";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.hotstep = std::stoi(parametervalue);
	}


	paramvec = { "spherical", "geo" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.spherical = readparambool(parametervalue, param.spherical);
	}

	parameterstr = "Radius";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Radius = std::stod(parametervalue);
	}

	parameterstr = "frictionmodel";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.frictionmodel = std::stoi(parametervalue);
	}

	parameterstr = "Adaptation";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> adaptpar = split(parametervalue, ',');

		if (!adaptpar.empty())
		{
			param.AdaptCrit = adaptpar[0];
			if (adaptpar.size() > 1)
				param.Adapt_arg1 = adaptpar[1];
			if (adaptpar.size() > 2)
				param.Adapt_arg2 = adaptpar[2];
			if (adaptpar.size() > 3)
				param.Adapt_arg3 = adaptpar[3];
			if (adaptpar.size() > 4)
				param.Adapt_arg4 = adaptpar[4];
			if (adaptpar.size() > 5)
				param.Adapt_arg5 = adaptpar[5];
		}
	}

	paramvec = { "crs", "spatialref", "spatial_ref", "wtk", "crsinfo","crs_info" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.crs_ref = parametervalue;
	}

	//Read Flexible Toutput variable
	parameterstr = "Toutput";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> Toutputpar = split(parametervalue, ',');

		if (!Toutputpar.empty())
		{
			std::vector<std::string> Toutputpar_vect = split_full(Toutputpar[0], ':');
			if (Toutputpar_vect.size() == 3)
			{
				/*
				if (!Toutputpar_vect[0].empty()) {
					param.Toutput.init = std::stod(Toutputpar_vect[0]);
				}
				if (!Toutputpar_vect[1].empty()) {
					param.Toutput.tstep = std::stod(Toutputpar_vect[1]);
				}
				if (!Toutputpar_vect[2].empty()) {
					param.Toutput.end = std::stod(Toutputpar_vect[2]);
				}
				*/
				double init, tstep, end;
				double tiny = 0.000001;
				if (!Toutputpar_vect[0].empty()) {
					init = std::stod(Toutputpar_vect[0]);
				}
				if (!Toutputpar_vect[1].empty()) {
					tstep = std::max(std::stod(Toutputpar_vect[1]), tiny);
				}
				if (!Toutputpar_vect[2].empty()) {
					end = std::stod(Toutputpar_vect[2]);
				}

				int nstep = (end - init) / tstep + 1;

				for (int k = 0; k < nstep; k++)
				{
					param.Toutput.val.push_back(std::min(init + tstep * k, end));
				}

			}
			else if (Toutputpar_vect.size() > 1)
			{
				//Failed: Toutput must be exactly 3 values, separated by ":" for a vector shape, in virst position. "t_init:t_step:t_end" (with possible empty values as "t_init:t_setps: " to use the last time steps as t_end;
				std::cerr << "Failed: Toutput must be exactly 3 values, separated by ':' for a vector shape, in virst position. 't_init : t_step : t_end' (with possible empty values as 't_init : t_setps : ' to use the last time steps as t_end; see log file for details" << std::endl;

				log("Failed: Toutput must be exactly 3 values, separated by ':' for a vector shape, in virst position. 't_init : t_step : t_end' (with possible empty values as 't_init : t_setps : ' to use the last time steps as t_end;");
				log(parametervalue);
			}
			else {
				param.Toutput.val.push_back(std::stod(Toutputpar_vect[0]));
			}
			if (Toutputpar.size() > 1)
			{
				for (int ii = 1; ii < Toutputpar.size(); ii++)
				{
					param.Toutput.val.push_back(std::stod(Toutputpar[ii]));
				}
			}
		}
	}

	paramvec = { "savebyblk", "writebyblk","saveperblk", "writeperblk","savebyblock", "writebyblock","saveperblock", "writeperblock" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.savebyblk = readparambool(parametervalue, param.savebyblk);
	}

	return param;
}



/*! \fn Forcing<T> readparamstr(std::string line, Forcing<T> forcing)
* Read BG_param.txt line and convert parameter to the righ parameter in the class
* return an updated Param class
*/
template <class T>
Forcing<T> readparamstr(std::string line, Forcing<T> forcing)
{
	std::string parameterstr, parametervalue;
	std::vector<std::string> paramvec;

	paramvec = { "Bathy","bathyfile","bathymetry","depfile","depthfile","topofile","topo","DEM" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		StaticForcingP<float> infobathy;
		forcing.Bathy.push_back(readfileinfo(parametervalue, infobathy));
		//std::cerr << "Bathymetry file found!" << std::endl;
	}



	paramvec = { "AOI","aoipoly" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		forcing.AOI.file = parametervalue;
		forcing.AOI.active = true;
	}

	/*parameterstr = "bathyfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		StaticForcingP<float> infobathy;
		forcing.Bathy.push_back(readfileinfo(parametervalue, infobathy));
		//forcing.Bathy = readfileinfo(parametervalue, forcing.Bathy);
		//std::cerr << "Bathymetry file found!" << std::endl;
	}

	parameterstr = "bathymetry";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		StaticForcingP<float> infobathy;
		forcing.Bathy.push_back(readfileinfo(parametervalue, infobathy));
		//forcing.Bathy = readfileinfo(parametervalue, forcing.Bathy);
		//std::cerr << "Bathymetry file found!" << std::endl;
	}

	//
	parameterstr = "depfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		StaticForcingP<float> infobathy;
		forcing.Bathy.push_back(readfileinfo(parametervalue, infobathy));
		//forcing.Bathy = readfileinfo(parametervalue, forcing.Bathy);
	}*/


	// Boundaries

	paramvec = { "left","leftbndfile","leftbnd" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		//forcing.left = readbndline(parametervalue);
		forcing.bndseg.push_back(readbndlineside(parametervalue, "left"));



	}

	paramvec = { "right","rightbndfile","rightbnd" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		//forcing.right = readbndline(parametervalue);
		forcing.bndseg.push_back(readbndlineside(parametervalue, "right"));

	}

	paramvec = { "top","topbndfile","topbnd" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		//forcing.top = readbndline(parametervalue);
		forcing.bndseg.push_back(readbndlineside(parametervalue, "top"));
	}

	paramvec = { "bot","botbndfile","botbnd","bottom" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		//forcing.bot = readbndline(parametervalue);
		forcing.bndseg.push_back(readbndlineside(parametervalue, "bot"));
	}

	paramvec = { "bnd","bndseg" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		//forcing.bot = readbndline(parametervalue);
		forcing.bndseg.push_back(readbndline(parametervalue));
	}


	//Tsunami deformation input files
	parameterstr = "deform";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		deformmap<float> thisdeform;
		std::vector<std::string> items = split(parametervalue, ',');
		//Need sanity check here
		thisdeform = readfileinfo(items[0], thisdeform);
		//thisdeform.inputfile = items[0];
		if (items.size() > 1)
		{
			thisdeform.startime = std::stod(items[1]);

		}
		if (items.size() > 2)
		{
			thisdeform.duration = std::stod(items[2]);

		}

		forcing.deform.push_back(thisdeform);

	}

	//Tsunami deformation input files
	parameterstr = "cavity";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		deformmap<float> thisdeform;

		thisdeform.iscavity = true;
		std::vector<std::string> items = split(parametervalue, ',');
		//Need sanity check here
		thisdeform = readfileinfo(items[0], thisdeform);
		//thisdeform.inputfile = items[0];
		if (items.size() > 1)
		{
			thisdeform.startime = std::stod(items[1]);

		}
		if (items.size() > 2)
		{
			thisdeform.duration = std::stod(items[2]);

		}

		forcing.deform.push_back(thisdeform);

	}

	//River
	paramvec = { "rivers","river" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> vars = split(parametervalue, ',');
		if (vars.size() == 5)
		{
			River thisriver;
			thisriver.Riverflowfile = trim(vars[0], " ");
			thisriver.xstart = std::stod(vars[1]);
			thisriver.xend = std::stod(vars[2]);
			thisriver.ystart = std::stod(vars[3]);
			thisriver.yend = std::stod(vars[4]);

			forcing.rivers.push_back(thisriver);
		}
		else
		{
			//Failed there should be 5 arguments (comma separated) when inputing a river: filename, xstart,xend,ystart,yend;
			std::cerr << "River input failed there should be 5 arguments (comma separated) when inputing a river: river = filename, xstart,xend,ystart,yend; see log file for details" << std::endl;

			log("River input below failed there should be 5 arguments (comma separated) when inputing a river: river = filename, xstart,xend,ystart,yend;");
			log(parametervalue);
		}
	}

	// friction coefficient (mapped or constant)
	// if it is a constant no-need to do anything below but if it is a file it overwrites any other value
	paramvec = { "cf","roughness","cfmap" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		if (std::any_of(parametervalue.begin(), parametervalue.end(), ::isalpha)) //(std::isdigit(parametervalue[0]) == false)
		{
			//forcing.cf = readfileinfo(parametervalue, forcing.cf);
			StaticForcingP<float> infoRoughness;
			forcing.cf.push_back(readfileinfo(parametervalue, infoRoughness));
		}
	}


	//if (!parametervalue.empty())
	//{
	//
		//std::cerr << "Bathymetry file found!" << std::endl;
	//}

	// Rain losses, initial and continuous loss
	paramvec = { "il","Rain_il","initialloss" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		if (std::any_of(parametervalue.begin(), parametervalue.end(), ::isalpha)) //(std::isdigit(parametervalue[0]) == false)
		{
			forcing.il = readfileinfo(parametervalue, forcing.il);
		}
	}
	paramvec = { "cl","Rain_cl","continuousloss" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		if (std::any_of(parametervalue.begin(), parametervalue.end(), ::isalpha)) //(std::isdigit(parametervalue[0]) == false)
		{
			forcing.cl = readfileinfo(parametervalue, forcing.cl);
		}
	}

	// wind forcing
	paramvec = { "Wind","windfiles" }; //## forcing.Wind
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{

		std::vector<std::string> vars = split(parametervalue, ',');
		if (vars.size() == 2)
		{
			// If 2 parameters (files) are given then 1st file is U wind and second is V wind.
			// This is for variable winds no rotation of the data is performed

			forcing.UWind = readfileinfo(trim(vars[0], " "), forcing.UWind);
			forcing.VWind = readfileinfo(trim(vars[1], " "), forcing.VWind);
		}
		else if (vars.size() == 1)
		{
			// if 1 parameter(file) is given then a 3 column file is expected showing time windspeed and direction
			// wind direction is rotated (later) to the grid direction (via grdalpha)
			forcing.UWind = readfileinfo(parametervalue, forcing.UWind);
			forcing.UWind.uniform = 1;

			//apply the same for Vwind? seem unecessary but need to be careful later in the code
		}
		else
		{
			//Failed there should be 5 arguments (comma separated) when inputing a river: filename, xstart,xend,ystart,yend;
			//std::cerr << "Wind input failed there should be 2 arguments (comma separated) when inputing a wind: windfiles = windfile.nc?uwind, windfile.nc?vwind; see log file for details" << std::endl;

			log("Wind input failed there should be 2 arguments(comma separated) when inputing a wind : windfiles = windfile.nc ? uwind, windfile.nc ? vwind; see log file for details");
			log(parametervalue);
		}

	}

	// atmospheric pressure forcing
	paramvec = { "Atmp","atmpfile" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		// needs to be a netcdf file 
		forcing.Atmp = readfileinfo(parametervalue, forcing.Atmp);

	}

	// rain forcing
	paramvec = { "Rain","rainfile" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		// netcdf file == Variable spatially
		// txt file (other than .nc) == spatially cst (txt file with 2 col time and mmm/h )
		forcing.Rain = readfileinfo(parametervalue, forcing.Rain);

		//set the expected type of input

		if (forcing.Rain.extension.compare("nc") == 0)
		{
			forcing.Rain.uniform = 0;
		}
		else
		{
			forcing.Rain.uniform = 1;
		}

	}

	parameterstr = "Adaptation";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> adaptpar = split(parametervalue, ',');
		// special case for 'Targetlevel' adaptation
		if (!adaptpar.empty())
		{
			//if (adaptpar[0].compare("Targetlevel") == 0)
			if (case_insensitive_compare(adaptpar[0], std::string("Targetlevel")) == 0)
			{
				for (int ng = 1; ng < adaptpar.size(); ng++)
				{
					StaticForcingP<int> infogrid;
					forcing.targetadapt.push_back(readfileinfo(adaptpar[ng], infogrid));
				}
			}
		}

	}

	return forcing;
}


/*! \fn void checkparamsanity(Param & XParam, Forcing<float> & XForcing)
* Check the Sanity of both Param and Forcing class
* If required some parameter are infered
*/
void checkparamsanity(Param& XParam, Forcing<float>& XForcing)
{
	Param DefaultParams;

	double tiny = 0.0000001;

	// Sanity check for model levels
	int minlev = XParam.minlevel;
	int maxlev = XParam.maxlevel;

	if (minlev == -99999)
	{
		minlev = XParam.initlevel;
	}
	if (maxlev == -99999)
	{
		maxlev = XParam.initlevel;
	}

	XParam.maxlevel = utils::max(maxlev, minlev);
	XParam.minlevel = utils::min(maxlev, minlev);

	XParam.initlevel = utils::min(utils::max(XParam.minlevel, XParam.initlevel), XParam.maxlevel);

	//force double for Rain on grid cases
	if (!XForcing.Rain.inputfile.empty())
	{
		XParam.doubleprecision = 1;
	}

	XParam.blkmemwidth = XParam.blkwidth + 2 * XParam.halowidth;
	XParam.blksize = utils::sq(XParam.blkmemwidth);

	///////////////////////////////////////////
	//  Read Bathy Information
	///////////////////////////////////////////

	//this sets xo yo  etc...

	// Any of xo,yo,xmax,ymax or dx not defined is assigned the value from bathy file
	//default value is nan in default param file

	//inputmap Bathymetry;
	//Bathymetry.inputfile = XForcing.Bathy.inputfile;
	//XForcing.Bathy = readforcinghead(XForcing.Bathy);



	if (std::isnan(XParam.xo))
		XParam.xo = XForcing.Bathy[0].xo - (0.5 * XForcing.Bathy[0].dx);
	if (std::isnan(XParam.xmax))
		XParam.xmax = XForcing.Bathy[0].xmax + (0.5 * XForcing.Bathy[0].dx);
	if (std::isnan(XParam.yo))
		XParam.yo = XForcing.Bathy[0].yo - (0.5 * XForcing.Bathy[0].dx);
	if (std::isnan(XParam.ymax))
		XParam.ymax = XForcing.Bathy[0].ymax + (0.5 * XForcing.Bathy[0].dx);

	if (std::isnan(XParam.dx))
		XParam.dx = XForcing.Bathy[0].dx;

	if (std::isnan(XParam.grdalpha))
		XParam.grdalpha = XForcing.Bathy[0].grdalpha; // here the default bathy grdalpha is 0.0 as defined by inputmap/Bathymetry class


	//Check Bathy input type
	if (XForcing.Bathy[0].extension.compare("dep") == 0 || XForcing.Bathy[0].extension.compare("bot") == 0)
	{
		if (std::isnan(XParam.dx))
		{
			//std::cerr << "FATAL ERROR: nx or ny or dx were not specified. These parameters are required when using ." << bathyext << " file" << std::endl;
			log("FATAL ERROR: nx or ny or dx were not specified. These parameters are required when using ." + XForcing.Bathy[0].extension + " file");
			exit(1);
		}
	}

	double levdx = calcres(XParam.dx, XParam.initlevel);// true grid resolution as in dx/2^(initlevel)
	//printf("levdx=%f;1 << XParam.initlevel=%f\n", levdx, calcres(1.0, XParam.initlevel));

	// First estimate nx and ny
	XParam.nx = ftoi((XParam.xmax - XParam.xo) / (levdx));
	XParam.ny = ftoi((XParam.ymax - XParam.yo) / (levdx)); //+1?
	//if desire size in one direction is under the bathy resolution or dx requested
	if (XParam.nx == 0) { XParam.nx = 1; }
	if (XParam.ny == 0) { XParam.ny = 1; }


	// Adjust xmax and ymax so that nx and ny are a factor of XParam.blkwidth [16]
	XParam.xmax = XParam.xo + (ceil(XParam.nx / ((double)XParam.blkwidth)) * ((double)XParam.blkwidth)) * levdx;
	XParam.ymax = XParam.yo + (ceil(XParam.ny / ((double)XParam.blkwidth)) * ((double)XParam.blkwidth)) * levdx;

	// Update nx and ny 
	XParam.nx = ftoi((XParam.xmax - XParam.xo) / (levdx));
	XParam.ny = ftoi((XParam.ymax - XParam.yo) / (levdx)); //+1?

	log("\nAdjusted model domain (xo/xmax/yo/ymax): ");
	log("\t" + std::to_string(XParam.xo) + "/" + std::to_string(XParam.xmax) + "/" + std::to_string(XParam.yo) + "/" + std::to_string(XParam.ymax));
	log("\t Initial resolution (level " + std::to_string(XParam.initlevel) + ") = " + std::to_string(levdx));

	if (XParam.spherical == false)
	{
		XParam.delta = XParam.dx;
		XParam.grdalpha = XParam.grdalpha * pi / 180.0; // grid rotation

	}
	else
	{
		//Geo grid

		XParam.delta = XParam.dx * XParam.Radius * pi / 180.0;
		//XParam.engine = 2;

		//printf("Using spherical coordinate; delta=%f rad\n", XParam.delta);
		log("Using spherical coordinate; delta=" + std::to_string(XParam.delta));
		if (XParam.grdalpha != 0.0)
		{
			//printf("grid rotation in spherical coordinate is not supported yet. grdalpha=%f rad\n", XParam.grdalpha);
			log("grid rotation in spherical coordinate is not supported yet. grdalpha=" + std::to_string(XParam.grdalpha * 180.0 / pi));
		}
	}

	// Read/setup bdn segment polygon. Note this can't be part of the "readforcing" step because xmin, xmax ymin ymax are not known then
	for (int iseg = 0; iseg < XForcing.bndseg.size(); iseg++)
	{
		XForcing.bndseg[iseg].poly = readbndpolysegment(XForcing.bndseg[iseg], XParam);
		if (XForcing.bndseg[iseg].type == 2)
		{
			XForcing.bndseg[iseg].type = 3;
		}


		XForcing.bndseg[iseg].left.isright = -1;
		XForcing.bndseg[iseg].left.istop = 0;

		XForcing.bndseg[iseg].right.isright = 1;
		XForcing.bndseg[iseg].right.istop = 0;

		XForcing.bndseg[iseg].top.isright = 0;
		XForcing.bndseg[iseg].top.istop = 1;

		XForcing.bndseg[iseg].bot.isright = 0;
		XForcing.bndseg[iseg].bot.istop = -1;
	}

	bndsegment remainderblk;

	remainderblk.left.isright = -1;
	remainderblk.left.istop = 0;

	remainderblk.right.isright = 1;
	remainderblk.right.istop = 0;

	remainderblk.top.isright = 0;
	remainderblk.top.istop = 1;

	remainderblk.bot.isright = 0;
	remainderblk.bot.istop = -1;
	remainderblk.type = XParam.aoibnd;

	XForcing.bndseg.push_back(remainderblk);
	for (int iseg = 0; iseg < XForcing.bndseg.size(); iseg++)
	{

		AllocateCPU(1, 1, XForcing.bndseg[iseg].left.blk);
		AllocateCPU(1, 1, XForcing.bndseg[iseg].right.blk);
		AllocateCPU(1, 1, XForcing.bndseg[iseg].top.blk);
		AllocateCPU(1, 1, XForcing.bndseg[iseg].bot.blk);

		AllocateCPU(1, 1, XForcing.bndseg[iseg].left.qmean);
		AllocateCPU(1, 1, XForcing.bndseg[iseg].right.qmean);
		AllocateCPU(1, 1, XForcing.bndseg[iseg].top.qmean);
		AllocateCPU(1, 1, XForcing.bndseg[iseg].bot.qmean);
	}





	//setup extra infor about boundaries
	// This is not needed anymore
	XForcing.left.side = 3;
	XForcing.left.isright = -1;
	XForcing.left.istop = 0;

	XForcing.right.side = 1;
	XForcing.right.isright = 1;
	XForcing.right.istop = 0;

	XForcing.top.side = 0;
	XForcing.top.isright = 0;
	XForcing.top.istop = 1;

	XForcing.bot.side = 2;
	XForcing.bot.isright = 0;
	XForcing.bot.istop = -1;


	//

	XForcing.Atmp.clampedge = float(XParam.Paref);

	if (!XForcing.Atmp.inputfile.empty())
	{
		XParam.atmpforcing = true;
		XParam.engine = 3;
	}


	// Make sure the nriver in param (used for preallocation of memory) and number of rivers in XForcing are consistent
	XParam.nrivers = int(XForcing.rivers.size());



	// Check whether endtime was specified by the user
	//No; i.e. endtimne =0.0
	//so the following conditions are useless



	if (abs(XParam.endtime - DefaultParams.endtime) <= tiny)
	{
		//No; i.e. endtimne =0.0
		XParam.endtime = 1.0 / tiny; //==huge
	}

	XParam.endtime = setendtime(XParam, XForcing);


	// Assign a value for reftime if not yet set. 
	//It is needed in the Netcdf file generation
	if (XParam.reftime.empty())
	{
		XParam.reftime = "2000-01-01T00:00:00";
	}

	log("Reference time: " + XParam.reftime);
	log("Model Initial time: " + std::to_string(XParam.totaltime));

	log("Model end time: " + std::to_string(XParam.endtime));

	// Check that outputtimestep is not zero, so at least the first and final time step are saved
	// If only the model stepup is needed than just run with endtime=0.0
	if (abs(XParam.outputtimestep - DefaultParams.outputtimestep) <= tiny)
	{
		XParam.outputtimestep = XParam.endtime;
		//otherwise there is really no point running the model
	}
	if (XParam.outputtimestep > XParam.endtime)
	{
		XParam.outputtimestep = XParam.endtime;
		//otherwise, no final output
	}

	//Initialisation of the main time output vector
	//Initialise default values for Toutput (output times for map outputs)
	InitialiseToutput(XParam.Toutput, XParam);


	// Initialisation of the time output vector for the zones outputs
	if (XParam.outzone.size() > 0)
	{
		for (int ii = 0; ii < XParam.outzone.size(); ii++)
		{
			{
				InitialiseToutput(XParam.outzone[ii].Toutput, XParam);
			}
		}
	}



	if (XParam.outvars.empty() && XParam.outputtimestep > 0.0)
	{
		//a nc file was specified but no output variable were specified
		std::vector<std::string> SupportedVarNames = { "zb", "zs", "u", "v", "h" };
		for (int isup = 0; isup < SupportedVarNames.size(); isup++)
		{
			XParam.outvars.push_back(SupportedVarNames[isup]);

		}

	}


	// Check whether a cuda compatible GPU is present
	if (XParam.GPUDEVICE >= 0)
	{
		// Init GPU
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		cudaDeviceProp prop;

		if (XParam.GPUDEVICE > (nDevices - 1))
		{
			//  if no GPU device are present then use the CPU (GPUDEVICE = -1)
			XParam.GPUDEVICE = (nDevices - 1);
		}
		cudaGetDeviceProperties(&prop, XParam.GPUDEVICE);
		//printf("There are %d GPU devices on this machine\n", nDevices);
		log("There are " + std::to_string(nDevices) + " GPU devices on this machine");

		if (XParam.GPUDEVICE >= 0)
		{

			log("Using Device: " + std::string(prop.name));
		}
		else
		{
			log("No GPU device were detected on this machine... Using CPU instead");
		}

	}


	if (XParam.minlevel != XParam.maxlevel)
	{
		if (XParam.AdaptCrit.empty())
		{
			XParam.AdaptCrit = "Threshold";
			XParam.Adapt_arg1 = "0.0";
			XParam.Adapt_arg2 = "h";
		}
	}

	//Check that we have both initial loss and continuous loss if one is given
	if (!XForcing.il.inputfile.empty())
	{
		if (XForcing.cl.inputfile.empty())
		{
			log("Error: File identified for initial loss but no data entered for continuous loss.\n Please, enter a ");
		}
	}
	if (!XForcing.cl.inputfile.empty())
	{
		if (XForcing.il.inputfile.empty())
		{
			log("Error: File identified for continuous loss but no data entered for initial loss");
		}
	}

	//Check that the Initial Loss/ Continuing Loss model is used if il, cl or hgw output are asked by user.
	if (!XParam.infiltration) // (XForcing.il.inputfile.empty() && XForcing.cl.inputfile.empty() && (XParam.il == 0.0) && (XParam.cl == 0.0))
	{
		std::vector<std::string> namestr = { "il","cl","hgw" };
		for (int ii = 0; ii < namestr.size(); ii++)
		{
			std::vector<std::string>::iterator itr = std::find(XParam.outvars.begin(), XParam.outvars.end(), namestr[ii]);
			if (itr != XParam.outvars.end())
			{
				log("The output variable associated to the ILCL model \"" + namestr[ii] + "\" is requested but the model is not used. The variable is removed from the outputs.");
				XParam.outvars.erase(itr);
			}
		}
	}

	//Check that the atmospheric forcing is used if datmpdx, datmpdy output are asked by user.
	if (XForcing.Atmp.inputfile.empty())
	{
		std::vector<std::string> namestr = { "datmpdx", "datmpdy" };
		for (int ii = 0; ii < namestr.size(); ii++)
		{
			std::vector<std::string>::iterator itr = std::find(XParam.outvars.begin(), XParam.outvars.end(), namestr[ii]);
			if (itr != XParam.outvars.end())
			{
				log("The output variable associated to the atmosheric forcing \"" + namestr[ii] + "\" is requested but the model is not used. The variable is removed from the outputs.");
				XParam.outvars.erase(itr);
			}
		}

	}

}

//Initialise default values for Toutput (output times for map outputs)
void InitialiseToutput(T_output& Toutput_loc, Param XParam)
{
	if (std::isnan(Toutput_loc.init))
	{
		Toutput_loc.init = XParam.totaltime;
	}
	if (std::isnan(Toutput_loc.end))
	{
		Toutput_loc.end = XParam.endtime;
	}
	if (std::isnan(Toutput_loc.tstep))
	{
		Toutput_loc.tstep = XParam.outputtimestep;
	}
}

/*! \fn double setendtime(Param XParam,Forcing<float> XForcing)
* Calculate/modify endtime based on maximum time in forcing
*
*/
double setendtime(Param XParam, Forcing<float> XForcing)
{
	//endtime cannot be bigger than the smallest time set in a boundary
	SLTS tempSLTS;
	double endtime = XParam.endtime;
	if (XForcing.left.on)
	{
		tempSLTS = XForcing.left.data.back();
		endtime = utils::min(endtime, tempSLTS.time);

	}
	if (XForcing.right.on)
	{
		tempSLTS = XForcing.right.data.back();
		endtime = utils::min(endtime, tempSLTS.time);
	}
	if (XForcing.top.on)
	{
		tempSLTS = XForcing.top.data.back();
		endtime = utils::min(endtime, tempSLTS.time);
	}
	if (XForcing.bot.on)
	{
		tempSLTS = XForcing.bot.data.back();
		endtime = utils::min(endtime, tempSLTS.time);
	}

	if (endtime < XParam.endtime)
	{
		log("\nWARNING: Boundary definition too short, endtime of the simulation reduced to : " + std::to_string(endtime));
	}

	return endtime;
}

/*! \fn std::string findparameter(std::string parameterstr, std::string line)
* separate parameter from value
*
*/
std::string findparameter(std::vector<std::string> parameterstr, std::string line)
{
	std::size_t found;
	std::string parameternumber, left, right;
	std::vector<std::string> splittedstr;

	// first look for an equal sign
	// No equal sign mean not a valid line so skip
	splittedstr = split(line, '=');
	if (splittedstr.size() > 1)
	{
		left = trim(splittedstr[0], " ");
		right = splittedstr[1]; // if there are more than one equal sign in the line the second one is ignored
		for (int ieq = 2; ieq < splittedstr.size(); ieq++)
		{
			right = right + "=" + splittedstr[ieq];
		}
		for (int ii = 0; ii < parameterstr.size(); ii++)
		{
			found = case_insensitive_compare(left, parameterstr[ii]);// it needs to strictly compare
			if (found == 0)
				break;
		}
		if (found == 0) // found the parameter
		{
			//std::cout <<"found LonMin at : "<< found << std::endl;
			//Numberstart = found + parameterstr.length();
			splittedstr = split(right, ';');
			if (splittedstr.size() >= 1)
			{
				parameternumber = splittedstr[0];
			}
			//std::cout << parameternumber << std::endl;

		}
	}
	return trim(parameternumber, " ");
	//return parameternumber;
}


std::string findparameter(std::string parameterstr, std::string line)
{
	std::vector<std::string> parametervec;

	parametervec.push_back(parameterstr);
	return findparameter(parametervec, line);
}


/*! \fn void split(const std::string &s, char delim, std::vector<std::string> &elems)
* split string based in character
*
*/
void split(const std::string& s, char delim, std::vector<std::string>& elems) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		if (!item.empty())//skip empty tokens
		{
			elems.push_back(item);
		}

	}
}

/*! \fn std::vector<std::string> split(const std::string &s, char delim)
* split string based in character
*
*/
std::vector<std::string> split(const std::string& s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}



/*! \fn void split_full(const std::string &s, char delim, std::vector<std::string> &elems)
* split string based in character, conserving empty item
*
*/
void split_full(const std::string& s, char delim, std::vector<std::string>& elems) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		std::string::iterator end_pos = std::remove(item.begin(), item.end(), ' ');
		item.erase(end_pos, item.end());
		elems.push_back(item);
	}
	if (s[s.length() - 1] == delim)
	{
		std::string item;
		elems.push_back(item);
	}
}

/*! \fn std::vector<std::string> split_full(const std::string &s, char delim)
* split string based in character, conserving empty items
*
*/
std::vector<std::string> split_full(const std::string& s, char delim) {
	std::vector<std::string> elems;
	split_full(s, delim, elems);
	return elems;
}


std::vector<std::string> split(const std::string s, const std::string delim)
{
	size_t ide = 0;
	int loc = 0;
	std::vector<std::string> output;
	std::string rem = s;


	while (ide < std::string::npos || output.size() == 0)
	{

		ide = rem.find(delim);
		if (ide == 0 || ide == std::string::npos)
		{
			output.push_back(rem);
			ide = std::string::npos;
		}
		else
		{
			output.push_back(rem.substr(loc, ide));
		}

		if (ide < (rem.length() - delim.length()))
		{
			loc = int(ide + delim.length());
			rem = rem.substr(loc);
		}
	}

	return output;



}


/*! \fn std::string trim(const std::string& str, const std::string& whitespace)
* remove leading and trailing space in a string
*
*/
std::string trim(const std::string& str, const std::string& whitespace)
{
	const auto strBegin = str.find_first_not_of(whitespace);
	if (strBegin == std::string::npos)
		return ""; // no content

	const auto strEnd = str.find_last_not_of(whitespace);
	const auto strRange = strEnd - strBegin + 1;

	return str.substr(strBegin, strRange);
}

/*! \fn std::size_t case_insensitive_compare(const std::string& str, const std::string& str)
* case non-sensitive string comparison (return 0 if the same, as for the "compare" function)
*
*/
std::size_t case_insensitive_compare(std::string s1, std::string s2)
{
	//Convert s1 and s2 to lower case strings
	std::transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
	std::transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
	//if (s1.compare(s2) == 0)
	return s1.compare(s2);
}

std::size_t case_insensitive_compare(std::string s1, std::vector<std::string> vecstr)
{
	std::size_t found;
	//Convert s1 and s2 to lower case strings
	for (int ii = 0; ii < vecstr.size(); ii++)
	{
		found = case_insensitive_compare(s1, vecstr[ii]);// it needs to strictly compare
		if (found == 0)
		{
			break;
		}
	}
	return found;
}


bndsegment readbndlineside(std::string parametervalue, std::string side)
{
	bndsegment bnd;


	std::vector<std::string> items = split(parametervalue, ',');

	if (items.size() == 1)
	{
		bnd.type = std::stoi(items[0]);

	}
	else if (items.size() >= 2)
	{
		const char* cstr = items[1].c_str();

		if (isdigit(cstr[0]))
		{
			//?
			bnd.type = std::stoi(items[1]);
			bnd.inputfile = items[0];
			bnd.on = true;



		}
		else
		{
			bnd.type = std::stoi(items[0]);
			bnd.inputfile = items[1];
			bnd.on = true;
		}

	}
	bnd.polyfile = side;
	if (bnd.on)
	{
		bnd.WLmap = readfileinfo(bnd.inputfile, bnd.WLmap);

		//set the expected type of input

		if (bnd.WLmap.extension.compare("nc") == 0)
		{
			bnd.WLmap.uniform = 0;
			bnd.uniform = 0;
		}
		else
		{
			bnd.WLmap.uniform = 1;
			bnd.uniform = 1;
		}
	}
	return bnd;
}


bndsegment readbndline(std::string parametervalue)
{
	//bndseg = area.txt, waterlevelforcing, 1;
	bndsegment bnd;
	std::vector<std::string> items = split(parametervalue, ',');
	if (items.size() == 1)
	{
		bnd.type = std::stoi(items[0]);

	}
	else if (items.size() >= 2)
	{
		const char* cstr = items[1].c_str();
		if (items[1].length() > 2)
		{
			bnd.polyfile = items[0];
			bnd.type = std::stoi(items[2]);
			bnd.inputfile = items[1];
			bnd.on = true;

		}
		else
		{
			bnd.polyfile = items[0];
			bnd.type = std::max(std::stoi(items[1]), 1); // only 2 param implies that it is either a wall or Neumann bnd

		}
	}


	//set the expected type of input

	if (bnd.on)
	{
		bnd.WLmap = readfileinfo(bnd.inputfile, bnd.WLmap);

		//set the expected type of input

		if (bnd.WLmap.extension.compare("nc") == 0)
		{
			bnd.WLmap.uniform = 0;
			bnd.uniform = 0;
		}
		else
		{
			bnd.WLmap.uniform = 1;
			bnd.uniform = 1;
		}
	}
	return bnd;
}



bool readparambool(std::string paramstr, bool defaultval)
{
	bool out = defaultval;
	std::vector<std::string> truestr = { "1","true","yes", "on" };
	std::vector<std::string> falsestr = { "-1","false","no","off" };

	if (case_insensitive_compare(paramstr, truestr) == 0)
	{
		out = true;
	}
	if (case_insensitive_compare(paramstr, falsestr) == 0)
	{
		out = false;
	}

	return out;
}




//inline bool fileexists(const std::string& name) {
//	struct stat buffer;
//	return (stat(name.c_str(), &buffer) == 0);
//}


