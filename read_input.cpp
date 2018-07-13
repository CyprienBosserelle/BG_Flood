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




#include "Header.cuh"

std::vector<SLTS> readWLfile(std::string WLfilename)
{
	std::vector<SLTS> slbnd;

	std::ifstream fs(WLfilename);

	if (fs.fail()) {
		std::cerr << WLfilename << " Water level bnd file could not be opened" << std::endl;
		write_text_to_log_file("ERROR: Water level bnd file could not be opened ");
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;
	SLTS slbndline;
	while (std::getline(fs, line))
	{
		//std::cout << line << std::endl;

		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			//Data should be in teh format :

			//by default we expect tab delimitation
			lineelements = split(line, '\t');
			if (lineelements.size() < 2)
			{
				// Is it space delimited?
				lineelements.clear();
				lineelements = split(line, ' ');
			}

			if (lineelements.size() < 2)
			{
				//Well it has to be comma delimited then
				lineelements.clear();
				lineelements = split(line, ',');
			}
			if (lineelements.size() < 2)
			{
				// Giving up now! Could not read the files
				//issue a warning and exit
				std::cerr << WLfilename << "ERROR Water level bnd file format error. only " << lineelements.size() << " where 2 were expected. Exiting." << std::endl;
				write_text_to_log_file("ERROR:  Water level bnd file (" + WLfilename + ") format error. only " + std::to_string(lineelements.size()) + " where 2 were expected. Exiting.");
				write_text_to_log_file(line);
				exit(1);
			}


			slbndline.time = std::stod(lineelements[0]);
			slbndline.wlev = std::stod(lineelements[1]);
			
			

			//slbndline = readBSHline(line);
			slbnd.push_back(slbndline);
			//std::cout << line << std::endl;
		}

	}
	fs.close();

	//std::cout << slbnd[0].wlev << std::endl;


	return slbnd;
}

Param readparamstr(std::string line, Param param)
{


	std::string parameterstr, parametervalue;

	///////////////////////////////////////////////////////
	// General parameters
	parameterstr = "bathy";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Bathymetryfile = parametervalue;
		//std::cerr << "Bathymetry file found!" << std::endl;
	}
	
	//
	parameterstr = "depfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Bathymetryfile = parametervalue;
	}
		
	
	

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

	
	parameterstr = "outputtimestep";
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

			param.outhhmean = (vvar.compare("hhmean") == 0) ? 1 : param.outhhmean;
			param.outzsmean = (vvar.compare("zsmean") == 0) ? 1 : param.outzsmean;
			param.outuumean = (vvar.compare("uumean") == 0) ? 1 : param.outuumean;
			param.outvvmean = (vvar.compare("vvmean") == 0) ? 1 : param.outvvmean;

			param.outhhmax = (vvar.compare("hhmax") == 0) ? 1 : param.outhhmax;
			param.outzsmax = (vvar.compare("zsmax") == 0) ? 1 : param.outzsmax;
			param.outuumax = (vvar.compare("uumax") == 0) ? 1 : param.outuumax;
			param.outvvmax = (vvar.compare("vvmax") == 0) ? 1 : param.outvvmax;

			param.outvort = (vvar.compare("vort") == 0) ? 1 : param.outvort;
		}
		

		
	}

	parameterstr = "leftbndfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.leftbndfile = parametervalue;
		//std::cerr << "Bathymetry file found!" << std::endl;
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
		param.grdalpha = std::stod(parametervalue);
	}

	parameterstr = "yo";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.grdalpha = std::stod(parametervalue);
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
		param.scalefactor = std::stod(parametervalue);
	}
	parameterstr = "addoffset";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.addoffset = std::stod(parametervalue);
	}
	parameterstr = "posdown";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.posdown = std::stoi(parametervalue);
	}
	return param;
}

Param checkparamsanity(Param XParam)
{
	Param DefaultParams;

	double tiny = 0.0000001;

	//Check Bathy input type
	std::string bathyext;
	std::vector<std::string> extvec = split(XParam.Bathymetryfile, '.');
	bathyext = extvec.back();
	if (bathyext.compare("nc") == 0)
	{
		if (abs(XParam.grdalpha - DefaultParams.grdalpha) < tiny)
		{
			
			write_text_to_log_file("For nc of bathy file please specify grdalpha in the XBG_param.txt (if different then default [0])");
		}
	}
	if (bathyext.compare("dep") == 0 || bathyext.compare("bot") == 0)
	{
		if (XParam.nx <= 0 || XParam.ny <= 0 || XParam.dx < tiny)
		{
			std::cerr << "FATAL ERROR: nx or ny or dx were not specified. These parameters are required when using ." << bathyext << " file" << std::endl;
			write_text_to_log_file("FATAL ERROR: nx or ny or dx were not specified. These parameters are required when using ." + bathyext + " file");
			exit(1);
		}
	}

	if (XParam.nx <= 0 || XParam.ny <= 0 || XParam.dx < tiny)
	{
		std::cerr << "FATAL ERROR: nx or ny or dx could not specified." << std::endl;
		write_text_to_log_file("FATAL ERROR: nx or ny or dx could not specified.");
		exit(1);
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
		std::cout << "WARNING: the number of timeseries output files is not equal to the number of nodes specified" << std::endl;
		std::cout << "for each location where timeseries output file is required, the XBG_param.txt file shoud contain 2 lines see example felow to extract in 2 locations:" << std::endl;
		std::cout << "TSOfile = Reef_Crest.txt" << std::endl;
		std::cout << "TSnode = 124,239;" << std::endl;
		std::cout << "TSOfile = shore.txt" << std::endl;
		std::cout << "TSnode = 233,256;" << std::endl;

		write_text_to_log_file("WARNING: the number of timeseries output files is not equal to the number of nodes specified");
		write_text_to_log_file("for each location where timeseries output file is required, the XBG_param.txt file shoud contain 2 lines see example felow to extract in 2 locations:");
		write_text_to_log_file("TSOfile = Reef_Crest.txt");
		write_text_to_log_file("TSnode = 124,239;");
		write_text_to_log_file("TSOfile = Shore.txt");
		write_text_to_log_file("TSnode = 233,256;");
		//min not defined for const so use this convoluted statement below
		int minsize;
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

	//Chaeck that if timeseries output nodes are specified that they are within nx and ny
	if (XParam.TSnodesout.size() > 0)
	{
		for (int o = 0; o < XParam.TSnodesout.size(); o++)
		{
			XParam.TSnodesout[o].i = min(max((int)round((XParam.TSnodesout[o].x - XParam.xo) / XParam.dx), 0), XParam.nx - 1);
			XParam.TSnodesout[o].j = min(max((int)round((XParam.TSnodesout[o].y - XParam.yo) / XParam.dx), 0), XParam.ny - 1);
			
		}

	}

	if (XParam.outvars.empty() && XParam.outputtimestep > 0)
	{
		//a nc file was specified but no output variable were specified
		std::vector<std::string> SupportedVarNames = { "zb", "zs", "uu", "vv", "hh" }; 
		for (int isup = 0; isup < SupportedVarNames.size(); isup++)
		{
			XParam.outvars.push_back(SupportedVarNames[isup]);
				
		}

	}

	return XParam;
}

std::string findparameter(std::string parameterstr, std::string line)
{
	std::size_t found, Numberstart, Numberend;
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

void readbathyHead(std::string filename, int &nx, int &ny, double &dx, double &grdalpha)
{
	

	std::ifstream fs(filename);

	if (fs.fail()) {
		std::cerr << filename << " bathy file (md file) could not be opened" << std::endl;
		write_text_to_log_file("ERROR: bathy file could not be opened ");
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;

	std::getline(fs, line);
	// skip empty lines
	if (!line.empty())
	{

		//by default we expect tab delimitation
		lineelements = split(line, '\t');
		if (lineelements.size() < 5)
		{
			// Is it space delimited?
			lineelements.clear();
			lineelements = split(line, ' ');
		}

		if (lineelements.size() < 5)
		{
			//Well it has to be comma delimited then
			lineelements.clear();
			lineelements = split(line, ',');
		}
		if (lineelements.size() < 5)
		{
			// Giving up now! Could not read the files
			//issue a warning and exit
			std::cerr << filename << "ERROR Wind bnd file format error. only " << lineelements.size() << " where 5 were expected. Exiting." << std::endl;
			write_text_to_log_file("ERROR:  Wind bnd file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where 3 were expected. Exiting.");
			write_text_to_log_file(line);
			exit(1);
		}

		nx = std::stoi(lineelements[0]);
		ny = std::stoi(lineelements[1]);
		dx = std::stod(lineelements[2]);
		grdalpha = std::stod(lineelements[4]);
	}

	fs.close();
}




extern "C" void readbathy(std::string filename, float *&zb)
{
	//read input data:
	//printf("bathy: %s\n", filename);
	FILE *fid;
	int nx, ny;
	double dx, grdalpha;
	//read md file
	fid = fopen(filename.c_str(), "r");
	fscanf(fid, "%u\t%u\t%lf\t%*f\t%lf", &nx, &ny, &dx, &grdalpha);
	grdalpha = grdalpha*pi / 180; // grid rotation

	int jread;
	//int jreadzs;
	for (int fnod = ny; fnod >= 1; fnod--)
	{

		fscanf(fid, "%u", &jread);
		
		for (int inod = 0; inod < nx; inod++)
		{
			fscanf(fid, "%f", &zb[inod + (jread - 1)*nx]);

		}
	}

	fclose(fid);
}

extern "C" void readXBbathy(std::string filename, int nx,int ny, float *&zb)
{
	//read input data:
	//printf("bathy: %s\n", filename);
	FILE *fid;
	
	//read md file
	fid = fopen(filename.c_str(), "r");
	
	

	
	//int jreadzs;
	for (int jnod = 0; jnod < ny; jnod++)
	{

		

		for (int inod = 0; inod < nx; inod++)
		{
			fscanf(fid, "%f", &zb[inod + (jnod)*nx]);

		}
	}

	fclose(fid);
}



void write_text_to_log_file(std::string text)
{
	std::ofstream log_file(
		"BG_log.txt", std::ios_base::out | std::ios_base::app);
	log_file << text << std::endl;
	log_file.close(); //destructor implicitly does it
}

void SaveParamtolog(Param XParam)
{
	write_text_to_log_file("#################################");
	write_text_to_log_file("# Bathymetry file");
	write_text_to_log_file("bathy = " + XParam.Bathymetryfile + ";");
	write_text_to_log_file("nx = " + std::to_string(XParam.nx) + ";");
	write_text_to_log_file("ny = " + std::to_string(XParam.ny) + ";");
	write_text_to_log_file("dx = " + std::to_string(XParam.dx) + ";");
	write_text_to_log_file("grdalpha = " + std::to_string(XParam.grdalpha*180.0/pi) + ";");
	write_text_to_log_file("\n");


	write_text_to_log_file("gpudevice = " + std::to_string(XParam.GPUDEVICE) + ";");
	write_text_to_log_file("\n");
	write_text_to_log_file("# Flow parameters");
	write_text_to_log_file("eps = " + std::to_string(XParam.eps) + ";");
	write_text_to_log_file("cf = " + std::to_string(XParam.cf) + ";");
	
	write_text_to_log_file("theta = " + std::to_string(XParam.theta) + ";");
	
	
	write_text_to_log_file("Cd = " + std::to_string(XParam.Cd) + ";");
	
	write_text_to_log_file("\n");
	write_text_to_log_file("# Timekeeping parameters");
	write_text_to_log_file("CFL = " + std::to_string(XParam.CFL) + ";");
	write_text_to_log_file("outputtimestep = " + std::to_string(XParam.outputtimestep) + ";");
	std::string alloutvars= "";
	for (int nvar = 0; nvar < XParam.outvars.size(); nvar++)
	{
		if (nvar > 0)
		{
			alloutvars = alloutvars + ", ";
		}
		alloutvars = alloutvars + XParam.outvars[nvar];
	}
	write_text_to_log_file("outvars = " + alloutvars + ";");


	write_text_to_log_file("endtime = " + std::to_string(XParam.endtime) + ";");
	write_text_to_log_file("\n");
	write_text_to_log_file("# Files");
	write_text_to_log_file("outfile = " + XParam.outfile + ";");
	
	if (!XParam.TSoutfile.empty())
	{
		for (int o = 0; o < XParam.TSoutfile.size(); o++)
		{
			write_text_to_log_file("TSOfile = " + XParam.TSoutfile[o] + ";");
			write_text_to_log_file("TSnode = " + std::to_string(XParam.TSnodesout[o].i) + "," + std::to_string(XParam.TSnodesout[o].j) + ";");
		}
	}
	write_text_to_log_file("\n");
	write_text_to_log_file("# Others");
	write_text_to_log_file("g = " + std::to_string(XParam.g) + ";");
	write_text_to_log_file("rho = " + std::to_string(XParam.rho) + ";");
}


double interptime(double next, double prev, double timenext, double time)
{
	return prev + (time) / (timenext)*(next - prev);
}

