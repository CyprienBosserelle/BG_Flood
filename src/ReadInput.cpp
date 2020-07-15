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


/*! \fn Param Readparamfile(Param XParam)
* Open the BG_param.txt file and read the parameters
* save the parameter in the Param structure and return an XParam.
*/
void Readparamfile(Param &XParam, Forcing<float> & XForcing)
{
	//
	std::ifstream fs("BG_param.txt");

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

		//////////////////////////////////////////////////////
		/////             Sanity check                   /////
		//////////////////////////////////////////////////////

		

		checkparamsanity(XParam,XForcing);

	}
	
}






Param readparamstr(std::string line, Param param)
{


	std::string parameterstr, parametervalue;

	///////////////////////////////////////////////////////
	// General parameters
	//


	parameterstr = "gpudevice";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.GPUDEVICE = std::stoi(parametervalue);
	}

	parameterstr = "GPUDEVICE";
	parametervalue = findparameter(parameterstr, line);
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
	
	parameterstr = "cf";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.cf = std::stod(parametervalue);
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

	
	parameterstr = "outputtimestep";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.outputtimestep = std::stod(parametervalue);

	}
	parameterstr = "outtimestep";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.outputtimestep = std::stod(parametervalue);

	}

	parameterstr = "endtime";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.endtime = std::stod(parametervalue);

	}
	parameterstr = "totaltime";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.totaltime = std::stod(parametervalue);

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
	parameterstr = "TSOfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.TSoutfile.push_back(parametervalue);
		
	}

	parameterstr = "TSnode";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> nodes = split(parametervalue, ',');
		//Need sanity check here
		TSnode node;
		node.x = std::stod(nodes[0]);
		node.y = std::stod(nodes[1]);

		//i and j are calculated in the Sanity check

		param.TSnodesout.push_back(node);

		
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
			std::vector<std::string> SupportedVarNames = { "zb", "zs", "uu", "vv", "hh", "hhmean", "zsmean", "uumean", "vvmean", "hhmax", "zsmax", "uumax", "vvmax" ,"vort"};
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

			param.outmean = (vvar.compare("hhmean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("zsmean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("uumean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("vvmean") == 0) ? true : param.outmean;

			param.outmax = (vvar.compare("hhmax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("zsmax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("uumax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("vvmax") == 0) ? true : param.outmax;

			param.outvort = (vvar.compare("vort") == 0) ? true : param.outvort;
		}
		

		
	}

	


	parameterstr = "resetmax";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.resetmax = std::stoi(parametervalue);
	}


	parameterstr = "leftbndfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.leftbnd.inputfile = parametervalue;
		param.leftbnd.on = 1;
		//std::cerr << "Bathymetry file found!" << std::endl;
	}

	parameterstr = "rightbndfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.rightbnd.inputfile = parametervalue;
		param.rightbnd.on = 1;
		//std::cerr << "Bathymetry file found!" << std::endl;
	}
	parameterstr = "topbndfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.topbnd.inputfile = parametervalue;
		param.topbnd.on = 1;
		//std::cerr << "Bathymetry file found!" << std::endl;
	}
	parameterstr = "botbndfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.botbnd.inputfile = parametervalue;
		param.botbnd.on = 1;
		//std::cerr << "Bathymetry file found!" << std::endl;
	}

	parameterstr = "left";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.leftbnd.type = std::stoi(parametervalue);
	}
	parameterstr = "right";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.rightbnd.type = std::stoi(parametervalue);
	}
	parameterstr = "top";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.topbnd.type = std::stoi(parametervalue);
	}
	parameterstr = "bot";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.botbnd.type = std::stoi(parametervalue);
	}

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

	parameterstr = "xo";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.xo = std::stod(parametervalue);
	}
	parameterstr = "xmin";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.xo = std::stod(parametervalue);
	}

	parameterstr = "yo";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.yo = std::stod(parametervalue);
	}
	parameterstr = "ymin";
	parametervalue = findparameter(parameterstr, line);
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

	parameterstr = "initzs";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.zsinit = std::stod(parametervalue);
	}

	parameterstr = "zsinit";
	parametervalue = findparameter(parameterstr, line);
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
	

	parameterstr = "spherical";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.spherical = std::stoi(parametervalue);
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
	

	return param;
}

template <class T>
Forcing<T> readparamstr(std::string line, Forcing<T> forcing)
{
	std::string parameterstr, parametervalue;

	parameterstr = "bathy";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		forcing.Bathy.inputfile = parametervalue;
		//std::cerr << "Bathymetry file found!" << std::endl;
	}

	parameterstr = "bathyfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		forcing.Bathy.inputfile = parametervalue;
		//std::cerr << "Bathymetry file found!" << std::endl;
	}

	parameterstr = "bathymetry";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		forcing.Bathy.inputfile = parametervalue;
		//std::cerr << "Bathymetry file found!" << std::endl;
	}

	//
	parameterstr = "depfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		forcing.Bathy.inputfile = parametervalue;
	}

	//Tsunami deformation input files
	parameterstr = "deform";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		deformmap<float> thisdeform;
		std::vector<std::string> items = split(parametervalue, ',');
		//Need sanity check here
		thisdeform.inputfile = items[0];
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
	parameterstr = "river";
	parametervalue = findparameter(parameterstr, line);
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

	// Mapped friction
	parameterstr = "cfmap";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		forcing.cf.inputfile = parametervalue;

	}
	parameterstr = "roughnessmap";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		forcing.cf.inputfile = parametervalue;

	}

	// wind forcing
	parameterstr = "windfiles";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		std::vector<std::string> vars = split(parametervalue, ',');
		if (vars.size() == 2)
		{
			// If 2 parameters (files) are given then 1st file is U wind and second is V wind.
			// This is for variable winds no rotation of the data is performed
			
			forcing.UWind.inputfile = trim(vars[0], " ");
			forcing.UWind.inputfile = trim(vars[1], " ");
		}
		else if (vars.size() == 1)
		{
			// if 1 parameter(file) is given then a 3 column file is expected showing time windspeed and direction
			// wind direction is rotated (later) to the grid direction (via grdalfa)
			forcing.UWind.inputfile = parametervalue;
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

	// atmpress forcing
	parameterstr = "atmpfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		// needs to be a netcdf file 
		forcing.Atmp.inputfile = parametervalue;
	}

	// atmpress forcing
	parameterstr = "rainfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		// netcdf file == Variable spatially
		// txt file (other than .nc) == spatially cst (txt file with 2 col time and mmm/h )
		forcing.Rain.inputfile = parametervalue;
		
		std::string fileext;

		std::vector<std::string> extvec = split(parametervalue, '.');

		std::vector<std::string> nameelements;
		//by default we expect tab delimitation
		nameelements = split(extvec.back(), '?');
		if (nameelements.size() > 1)
		{
			//variable name is not given so it is assumed to be z
			fileext = nameelements[0];
		}
		else
		{
			fileext = extvec.back();
		}

		//set the expected type of input

		if (fileext.compare("nc") == 0)
		{
			forcing.Rain.uniform = 0;
		}
		else
		{
			forcing.Rain.uniform = 1;
		}



	}

	return forcing;
}

void checkparamsanity(Param & XParam, Forcing<float> & XForcing)
{
	Param DefaultParams;

	double tiny = 0.0000001;

	//force double for Rain on grid cases
	if (!XForcing.Rain.inputfile.empty())
	{
		XParam.doubleprecision = 1;
	}

	///////////////////////////////////////////
	//  Read Bathy Information
	///////////////////////////////////////////

	//this sets xo yo  etc...

	// Any of xo,yo,xmax,ymax or dx not defined is assigned the value from bathy file
	//default value is nan in default param file

	inputmap Bathymetry;
	Bathymetry.inputfile = XForcing.Bathy.inputfile;
	Bathymetry = readBathyhead(Bathymetry);
	XForcing.Bathy.xo = Bathymetry.xo;
	XForcing.Bathy.yo = Bathymetry.yo;
	XForcing.Bathy.xmax = Bathymetry.xmax;
	XForcing.Bathy.ymax = Bathymetry.ymax;
	XForcing.Bathy.nx = Bathymetry.nx;
	XForcing.Bathy.ny = Bathymetry.ny;



	if (std::isnan(XParam.xo))
		XParam.xo = XForcing.Bathy.xo;
	if (std::isnan(XParam.xmax))
		XParam.xmax = XForcing.Bathy.xmax;
	if(std::isnan(XParam.yo))
		XParam.yo = XForcing.Bathy.yo;
	if (std::isnan(XParam.ymax))
		XParam.ymax = XForcing.Bathy.ymax;

	if (std::isnan(XParam.dx))
		XParam.dx = XForcing.Bathy.dx;
	
	if (std::isnan(XParam.grdalpha))
		XParam.grdalpha = Bathymetry.grdalpha; // here the default bathy grdalpha is 0.0 as defined by inputmap/Bathymetry class


	//Check Bathy input type
	std::string bathyext;
	std::vector<std::string> extvec = split(XForcing.Bathy.inputfile, '.');
	bathyext = extvec.back();
	
	if (bathyext.compare("dep") == 0 || bathyext.compare("bot") == 0)
	{
		if (std::isnan(XParam.dx))
		{
			//std::cerr << "FATAL ERROR: nx or ny or dx were not specified. These parameters are required when using ." << bathyext << " file" << std::endl;
			log("FATAL ERROR: nx or ny or dx were not specified. These parameters are required when using ." + bathyext + " file");
			exit(1);
		}
	}

	double levdx = calcres(XParam.dx, XParam.initlevel);// true grid resolution as in dx/2^(initlevel)
	//printf("levdx=%f;1 << XParam.initlevel=%f\n", levdx, calcres(1.0, XParam.initlevel));

	XParam.nx = (XParam.xmax - XParam.xo) / (levdx)+1;
	XParam.ny = (XParam.ymax - XParam.yo) / (levdx)+1; //+1?


	if (XParam.spherical < 1)
	{
		XParam.delta = XParam.dx;
		XParam.grdalpha = XParam.grdalpha*pi / 180.0; // grid rotation

	}
	else
	{
		//Geo grid
		XParam.delta = XParam.dx * XParam.Radius*pi / 180.0;
		//printf("Using spherical coordinate; delta=%f rad\n", XParam.delta);
		log("Using spherical coordinate; delta=" + std::to_string(XParam.delta));
		if (XParam.grdalpha != 0.0)
		{
			//printf("grid rotation in spherical coordinate is not supported yet. grdalpha=%f rad\n", XParam.grdalpha);
			log("grid rotation in spherical coordinate is not supported yet. grdalpha=" + std::to_string(XParam.grdalpha*180.0 / pi));
		}
	}

	




	// Check whether endtime was specified by the user
	//No; i.e. endtimne =0.0
	//so the following conditions are useless
	
	
	
	if (abs(XParam.endtime - DefaultParams.endtime) <= tiny)
	{
		//No; i.e. endtimne =0.0
		XParam.endtime = 1.0 / tiny; //==huge
		
	//	if (slbnd.back().time>0.0 && wndbnd.back().time > 0.0)
	//	{
	//		XParam.endtime = min(slbnd.back().time, wndbnd.back().time);
	//	}
	//
	//	
	}
	
	// Endtime is checked versus the bnd input i.e. model cannot go further than secified bnd (actually in GPU case it can but it probably shoudn't)
	// This is done in a separate function.




	// Check that outputtimestep is not zero, so at least the first and final time step are saved
	// If only the model stepup is needed than just run with endtime=0.0
	if (abs(XParam.outputtimestep - DefaultParams.outputtimestep) <= tiny)
	{
		XParam.outputtimestep = XParam.endtime;
		//otherwise there is really no point running the model
	}







	//Check that there are as many file specified for Time series output as there are vectors of nodes
	if (XParam.TSoutfile.size() != XParam.TSnodesout.size())
	{
		// Issue a Warning
		//std::cout << "WARNING: the number of timeseries output files is not equal to the number of nodes specified" << std::endl;
		//std::cout << "for each location where timeseries output file is required, the XBG_param.txt file shoud contain 2 lines see example felow to extract in 2 locations:" << std::endl;
		//std::cout << "TSOfile = Reef_Crest.txt" << std::endl;
		//std::cout << "TSnode = 124,239;" << std::endl;
		//std::cout << "TSOfile = shore.txt" << std::endl;
		//std::cout << "TSnode = 233,256;" << std::endl;

		log("WARNING: the number of timeseries output files is not equal to the number of nodes specified");
		log("for each location where timeseries output file is required, the XBG_param.txt file shoud contain 2 lines see example felow to extract in 2 locations:");
		log("TSOfile = Reef_Crest.txt");
		log("TSnode = 124,239;");
		log("TSOfile = Shore.txt");
		log("TSnode = 233,256;");
		//min not defined for const so use this convoluted statement below
		size_t minsize;
		if (XParam.TSoutfile.size() > XParam.TSnodesout.size())
		{
			minsize = XParam.TSnodesout.size();
		}

		if (XParam.TSoutfile.size() < XParam.TSnodesout.size())
		{
			minsize = XParam.TSoutfile.size();
		}

				
		XParam.TSoutfile.resize(minsize);
		XParam.TSnodesout.resize(minsize);
	}

	

	if (XParam.outvars.empty() && XParam.outputtimestep > 0.0)
	{
		//a nc file was specified but no output variable were specified
		std::vector<std::string> SupportedVarNames = { "zb", "zs", "uu", "vv", "hh" }; 
		for (int isup = 0; isup < SupportedVarNames.size(); isup++)
		{
			XParam.outvars.push_back(SupportedVarNames[isup]);
				
		}

	}


	

	
}

double setendtime(Param XParam)
{
	//endtime cannot be bigger thn the smallest time set in a boundary
	SLTS tempSLTS;
	double endtime = XParam.endtime;
	if (XParam.leftbnd.on)
	{
		tempSLTS = XParam.leftbnd.data.back();
		endtime = utils::min( endtime, tempSLTS.time);
		
	}
	if (XParam.rightbnd.on)
	{
		tempSLTS = XParam.rightbnd.data.back();
		endtime = utils::min(endtime, tempSLTS.time);
	}
	if (XParam.topbnd.on)
	{
		tempSLTS = XParam.topbnd.data.back();
		endtime = utils::min(endtime, tempSLTS.time);
	}
	if (XParam.botbnd.on)
	{
		tempSLTS = XParam.botbnd.data.back();
		endtime = utils::min(endtime, tempSLTS.time);
	}

	return endtime;
}

std::string findparameter(std::string parameterstr, std::string line)
{
	std::size_t found;
	std::string parameternumber,left,right;
	std::vector<std::string> splittedstr;
	
	// first look fo an equal sign
	// No equal sign mean not a valid line so skip
	splittedstr=split(line, '=' );
	if (splittedstr.size()>1)
	{
		left = trim(splittedstr[0]," ");
		right = splittedstr[1]; // if there are more than one equal sign in the line the second one is ignored
		found = left.compare(parameterstr);// it needs to strictly compare
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
}

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
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


std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

std::string trim(const std::string& str, const std::string& whitespace)
{
	const auto strBegin = str.find_first_not_of(whitespace);
	if (strBegin == std::string::npos)
		return ""; // no content

	const auto strEnd = str.find_last_not_of(whitespace);
	const auto strRange = strEnd - strBegin + 1;

	return str.substr(strBegin, strRange);
}



