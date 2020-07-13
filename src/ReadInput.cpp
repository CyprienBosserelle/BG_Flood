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



Param Readparamfile(Param XParam)
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
				//std::cout << line << std::endl;
			}

		}
		fs.close();

		//////////////////////////////////////////////////////
		/////             Sanity check                   /////
		//////////////////////////////////////////////////////

		

		

	}
	return XParam
}


std::vector<SLTS> readbndfile(std::string filename,Param XParam, int side)
{
	// read bnd or nest file
	// side is for deciding whether we are talking about a left(side=0) bot (side =1) right (side=2) or top (side=3)
	// Warning just made this up and need to have some sort of convention in the model
	std::string fileext;
	std::vector<std::string> extvec = split(filename, '.');

	std::vector<std::string> nameelements;

	std::vector<SLTS> Bndinfo;

	//
	//printf("%d\n", side);

	double xxo, xxmax, yy;
	int hor;
	switch (side)
	{
		case 0://Left bnd
		{
			xxo = XParam.yo;
			xxmax = XParam.ymax;
			yy = XParam.xo;
			hor = 0;
			break;
		}
		case 1://Bot bnd
		{
			xxo = XParam.xo;
			xxmax = XParam.xmax;
			yy = XParam.yo;
			hor = 1;
			break;
		}
		case 2://Right bnd
		{
			xxo = XParam.yo;
			xxmax = XParam.ymax;
			yy = XParam.xmax;
			hor = 0;
			break;
		}
		case 3://Top bnd
		{
			xxo = XParam.xo;
			xxmax = XParam.xmax;
			yy = XParam.ymax;
			hor = 1;
			break;
		}
	}


	//printf("%f\t%f\t%f\n", xxo, xxmax, yy);

	nameelements = split(extvec.back(), '?');
	if (nameelements.size() > 1)
	{
		
		fileext = nameelements[0];
	}
	else
	{
		fileext = extvec.back();
	}

	if (fileext.compare("nc") == 0)
	{
		//Bndinfo = readNestfile(filename);
		Bndinfo = readNestfile(filename, hor, XParam.eps, xxo, xxmax, yy);
	}
	else
	{
		Bndinfo = readWLfile(filename);
	}

	// Add zsoffset
	for (int i = 0; i < Bndinfo.size(); i++)
	{
		for (int n = 0; n < Bndinfo[i].wlevs.size(); n++)
		{
			Bndinfo[i].wlevs[n] = Bndinfo[i].wlevs[n] + XParam.zsoffset;
		}
	}


	return Bndinfo;
}

std::vector<SLTS> readWLfile(std::string WLfilename)
{
	std::vector<SLTS> slbnd;

	std::ifstream fs(WLfilename);

	if (fs.fail()) {
		//std::cerr << WLfilename << " Water level bnd file could not be opened" << std::endl;
		log("ERROR: Water level bnd file could not be opened : " + WLfilename);
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;
	std::vector<double> WLS;
	SLTS slbndline;
	while (std::getline(fs, line))
	{
		//std::cout << line << std::endl;

		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			//Data should be in the format : time,Water level 1,Water level 2,...Water level n
			//Location where the water level is 0:ny/(nwl-1):ny where nwl i the number of wlevnodes

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
				//std::cerr << WLfilename << "ERROR Water level bnd file format error. only " << lineelements.size() << " where at least 2 were expected. Exiting." << std::endl;
				log("ERROR:  Water level bnd file (" + WLfilename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 2 were expected. Exiting.");
				log(line);
				exit(1);
			}


			slbndline.time = std::stod(lineelements[0]);

			for (int n = 1; n < lineelements.size(); n++)
			{
				WLS.push_back(std::stod(lineelements[n]));
			}



			slbndline.wlevs = WLS;
			
			

			//slbndline = readBSHline(line);
			slbnd.push_back(slbndline);
			//std::cout << line << std::endl;
			WLS.clear();
		}

	}
	fs.close();

	//std::cout << slbnd[0].wlev << std::endl;


	return slbnd;
}

std::vector<SLTS> readNestfile(std::string ncfile, int hor ,double eps, double bndxo, double bndxmax, double bndy)
{
	// Prep boundary input vector from anorthe model output file
	//this function works for botom top bnd as written but flips x and y for left and right bnds
	// hor controls wheter the boundary is a top/botom bnd hor=1 or left/right hor=0 
	std::vector<SLTS> slbnd;
	SLTS slbndline;
	
	std::vector<double> WLS,Unest,Vnest;
	//Define NC file variables
	int nnx, nny, nt, nbndpts, indxx, indyy, indx, indy,nx, ny;
	double dx, xxo, yyo, to, xmax, ymax, tmax,xo,yo;
	double * ttt, *zsa;
	bool checkhh = false;
	int iswet;

	// Read NC info
	readgridncsize(ncfile, nnx, nny, nt, dx, xxo, yyo, to, xmax, ymax, tmax);
	
	if (hor == 0)
	{
		nx = nny;
		ny = nnx;
		xo = yyo;
		yo = xxo;

	}
	else
	{
		nx = nnx;
		ny = nny;
		xo = xxo;
		yo = yyo;
	}

	// Read time vector
	ttt=(double *)malloc(nt*sizeof(double));
	zsa = (double *)malloc(1*sizeof(double));
	readnctime(ncfile, ttt);


	

	nbndpts = (int)((bndxmax - bndxo) / dx)+1;

	//printf("%f\t%f\t%f\t%f\n", bndxmax, bndxo, xo, yo);
	//printf("%f\t%f\t%f\t%f\n", bndxmax, bndxo, xxo, yyo);
	//printf("%f\t%d\t%d\t%f\n", bndy, nx, ny, dx);

	//printf("%d\n", nbndpts);
	std::string ncfilestr;
	std::string varstr,varstruu,varstrvv;


	//char ncfile[]="ocean_ausnwsrstwq2.nc";
	std::vector<std::string> nameelements;
	nameelements = split(ncfile, '?');
	if (nameelements.size() > 1)
	{

		ncfilestr = nameelements[0];
		varstr = nameelements[1];
	}
	else
	{

		ncfilestr = ncfile;
		varstr = "zs";
		checkhh = true;
	}


	for (int it = 0; it < nt; it++)
	{
		slbndline.time = ttt[it];
		for (int ibnd = 0; ibnd < nbndpts; ibnd++)
		{
			//
			// Read// interpolate data for each bnds
			indxx = max(min((int)((bndxo+(dx*ibnd) - xo) / dx), nx - 1), 0);
			indyy = max(min((int)((bndy - yo) / dx), ny - 1), 0);

			if (hor == 0)
			{
				indy = indxx;
				indx = indyy;
			}
			else
			{
				indx = indxx;
				indy = indyy;
			}

			iswet=readncslev1(ncfile, varstr, indx, indy, it, checkhh,eps, zsa);
			//varstr
			//printf("%d\t%d\t%d\tzs=%f\t%d\n", it,indx, indy, zsa[0],iswet);

			if (iswet == 0)
			{
				if (WLS.size() >= 1)
				{
					zsa[0] = WLS.back();
				}
				else
				{
					zsa[0] = 0.0;
				}
			}

			WLS.push_back(zsa[0]);

			//printf("zs=%f\\n", zsa[0]);

			// If true nesting then uu and vv are expected to be present in the netcdf file 

			if (checkhh)
			{
				varstruu = "uu";
				iswet = readncslev1(ncfilestr, varstruu, indx, indy, it, checkhh, eps, zsa);
				//printf("%d\t%d\t%d\tuu=%f\t%d\n", it, indx, indy, zsa[0], iswet);
				//printf("%d\t%d\t%f\n", indx, indy, zsa[0]);

				if (iswet == 0)
				{

					if (Unest.size() >= 1)
					{
						zsa[0] = Unest.back();
					}
					else
					{
						zsa[0] = 0.0;
					}
				}

				Unest.push_back(zsa[0]);

				varstrvv = "vv";
				iswet = readncslev1(ncfile, varstrvv, indx, indy, it, checkhh, eps, zsa);
				//printf("%d\t%d\t%d\tvv=%f\t%d\n", it, indx, indy, zsa[0], iswet);
				//printf("%d\t%d\t%f\n", indx, indy, zsa[0]);

				if (iswet == 0)
				{
					if (Vnest.size() >= 1)
					{
						zsa[0] = Vnest.back();
					}
					else
					{
						zsa[0] = 0.0;
					}
				}

				Vnest.push_back(zsa[0]);
			}




		}
		slbndline.wlevs = WLS;
		WLS.clear();
		if (checkhh)
		{
			slbndline.uuvel = Unest;
			slbndline.vvvel = Vnest;
			Unest.clear();
			Vnest.clear();
		}

		slbnd.push_back(slbndline);
		//std::cout << line << std::endl;
		
	}
	///To Be continued
	
	free(ttt);
	free(zsa);
	return slbnd;
}

std::vector<Flowin> readFlowfile(std::string Flowfilename)
{
	std::vector<Flowin> slbnd;

	std::ifstream fs(Flowfilename);

	if (fs.fail()) {
		std::cerr << Flowfilename << " Flow file could not be opened" << std::endl;
		write_text_to_log_file("ERROR: Flow/River file could not be opened ");
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;
	//std::vector<double> WLS;
	Flowin slbndline;
	while (std::getline(fs, line))
	{
		//std::cout << line << std::endl;

		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			//Data should be in the format : time,Water level 1,Water level 2,...Water level n
			//Location where the water level is 0:ny/(nwl-1):ny where nwl i the number of wlevnodes

			//by default we expect tab delimitation
			lineelements = split(line, '\t');
			if (lineelements.size() != 2)
			{
				// Is it space delimited?
				lineelements.clear();
				lineelements = split(line, ' ');
			}

			if (lineelements.size() != 2)
			{
				//Well it has to be comma delimited then
				lineelements.clear();
				lineelements = split(line, ',');
			}
			if (lineelements.size() != 2)
			{
				// Giving up now! Could not read the files
				//issue a warning and exit
				std::cerr << Flowfilename << "ERROR flow file format error. only " << lineelements.size() << " where at least 2 were expected. Exiting." << std::endl;
				write_text_to_log_file("ERROR:  flow file (" + Flowfilename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 2 were expected. Exiting.");
				write_text_to_log_file(line);
				exit(1);
			}


			slbndline.time = std::stod(lineelements[0]);

			



			slbndline.q = std::stod(lineelements[1]);;



			//slbndline = readBSHline(line);
			slbnd.push_back(slbndline);
			//std::cout << line << std::endl;
			//WLS.clear();
		}

	}
	fs.close();

	//std::cout << slbnd[0].wlev << std::endl;


	return slbnd;
}
std::vector<Windin> readINfileUNI(std::string filename)
{
	std::vector<Windin> wndinput;

	std::ifstream fs(filename);

	if (fs.fail()) {
		std::cerr << filename << "ERROR: Atm presssure / Rainfall file could not be opened" << std::endl;
		write_text_to_log_file("ERROR: Atm presssure / Rainfall file could not be opened ");
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;
	std::vector<double> WLS;
	Windin wndline;
	while (std::getline(fs, line))
	{
		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			//Data should be in the format : time,wind speed, wind dir, uwind vwind
			//Location where the water level is 0:ny/(nwl-1):ny where nwl i the number of wlevnodes

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
				std::cerr << filename << "ERROR Atm presssure / Rainfall  file format error. only " << lineelements.size() << " where at least 2 were expected. Exiting." << std::endl;
				write_text_to_log_file("ERROR:  Atm presssure / Rainfall file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 2 were expected. Exiting.");
				write_text_to_log_file(line);
				exit(1);
			}


			wndline.time = std::stod(lineelements[0]);
			wndline.wspeed = std::stod(lineelements[1]);
			
			wndinput.push_back(wndline);
			

		}

	}
	fs.close();

	return wndinput;
}
std::vector<Windin> readWNDfileUNI(std::string filename, double grdalpha)
{
	// Warning grdapha is expected in radian here
	std::vector<Windin> wndinput;

	std::ifstream fs(filename);

	if (fs.fail()) {
		std::cerr << filename << "ERROR: Wind file could not be opened" << std::endl;
		write_text_to_log_file("ERROR: Wind file could not be opened ");
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;
	std::vector<double> WLS;
	Windin wndline;
	while (std::getline(fs, line))
	{
		//std::cout << line << std::endl;

		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			//Data should be in the format : time,wind speed, wind dir, uwind vwind
			//Location where the water level is 0:ny/(nwl-1):ny where nwl i the number of wlevnodes

			//by default we expect tab delimitation
			lineelements = split(line, '\t');
			if (lineelements.size() < 3)
			{
				// Is it space delimited?
				lineelements.clear();
				lineelements = split(line, ' ');
			}

			if (lineelements.size() < 3)
			{
				//Well it has to be comma delimited then
				lineelements.clear();
				lineelements = split(line, ',');
			}
			if (lineelements.size() < 3)
			{
				// Giving up now! Could not read the files
				//issue a warning and exit
				std::cerr << filename << "ERROR Wind  file format error. only " << lineelements.size() << " where at least 3 were expected. Exiting." << std::endl;
				write_text_to_log_file("ERROR:  Wind file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 3 were expected. Exiting.");
				write_text_to_log_file(line);
				exit(1);
			}


			wndline.time = std::stod(lineelements[0]);
			if (lineelements.size() == 5)
			{
				// U and v are explicitelly stated
				wndline.wspeed = std::stod(lineelements[1]); // Actually his is a dummy 
				wndline.wdirection= std::stod(lineelements[2]); // Actually his is a dummy
				wndline.uwind = std::stod(lineelements[3]);
				wndline.vwind = std::stod(lineelements[4]);
			}
			else
			{
				// read speed and direction and directly convert to u and v
				wndline.wspeed = std::stod(lineelements[1]); // Actually his is a dummy 
				wndline.wdirection = std::stod(lineelements[2]);
				double theta = (1.5*pi - grdalpha) - wndline.wdirection*pi / 180;

				wndline.uwind = wndline.wspeed*cos(theta);
				wndline.vwind = wndline.wspeed*sin(theta);
			}
			//slbndline.wlevs = WLS;



			//slbndline = readBSHline(line);
			wndinput.push_back(wndline);
			//std::cout << line << std::endl;
			
		}

	}
	fs.close();

	//std::cout << slbnd[0].wlev << std::endl;


	return wndinput;
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
		param.Bathymetry.inputfile = parametervalue;
		//std::cerr << "Bathymetry file found!" << std::endl;
	}
	
	//
	parameterstr = "depfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Bathymetry.inputfile = parametervalue;
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

	//Tsunami deformation input files
	parameterstr = "deform";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		
		deformmap thisdeform;
		std::vector<std::string> items = split(parametervalue, ',');
		//Need sanity check here
		thisdeform.grid.inputfile = items[0];
		if (items.size() > 1)
		{
			thisdeform.startime = std::stod(items[1]);

		}
		if (items.size() > 2)
		{
			thisdeform.duration = std::stod(items[2]);

		}

		param.deform.push_back(thisdeform);

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

	parameterstr = "river";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> vars = split(parametervalue, ',');
		if (vars.size() == 5)
		{
			River thisriver;
			thisriver.Riverflowfile = trim(vars[0], " ");
			thisriver.xstart= std::stod(vars[1]);
			thisriver.xend= std::stod(vars[2]);
			thisriver.ystart = std::stod(vars[3]);
			thisriver.yend = std::stod(vars[4]);

			param.Rivers.push_back(thisriver);
		}
		else
		{
			//Failed there should be 5 arguments (comma separated) when inputing a river: filename, xstart,xend,ystart,yend;
			std::cerr << "River input failed there should be 5 arguments (comma separated) when inputing a river: river = filename, xstart,xend,ystart,yend; see log file for details" << std::endl;
			
			write_text_to_log_file("River input below failed there should be 5 arguments (comma separated) when inputing a river: river = filename, xstart,xend,ystart,yend;");
			write_text_to_log_file(parametervalue);
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
	// Mapped friction
	parameterstr = "cfmap";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		
		param.roughnessmap.inputfile = parametervalue;
		
	}
	parameterstr = "roughnessmap";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		param.roughnessmap.inputfile = parametervalue;

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
			param.windU.inputfile = trim(vars[0], " ");
			param.windV.inputfile = trim(vars[1], " ");
		}
		else if (vars.size() == 1)
		{
			// if 1 parameter(file) is given then a 3 column file is expected showing time windspeed and direction
			// wind direction is rotated (later) to the grid direction (via grdalfa)
			param.windU.inputfile = parametervalue;
			param.windU.uniform = 1;
			//apply the same for Vwind? seem unecessary but need to be careful later in the code
		}
		else
		{
			//Failed there should be 5 arguments (comma separated) when inputing a river: filename, xstart,xend,ystart,yend;
			std::cerr << "Wind input failed there should be 2 arguments (comma separated) when inputing a wind: windfiles = windfile.nc?uwind, windfile.nc?vwind; see log file for details" << std::endl;

			write_text_to_log_file("Wind input failed there should be 2 arguments(comma separated) when inputing a wind : windfiles = windfile.nc ? uwind, windfile.nc ? vwind; see log file for details");
			write_text_to_log_file(parametervalue);
		}

	}

	// atmpress forcing
	parameterstr = "atmpfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		// needs to be a netcdf file 

		param.atmP.inputfile = parametervalue;

	}

	// atmpress forcing
	parameterstr = "rainfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		// netcdf file == Variable spatially
		// txt file (other than .nc) == spatially cst (txt file with 2 col time and mmm/h )
		param.Rainongrid.inputfile = parametervalue;

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
			param.Rainongrid.uniform = 0;
		}
		else
		{
			param.Rainongrid.uniform = 1;
		}



	}

	return param;
}



Param checkparamsanity(Param XParam)
{
	Param DefaultParams;

	double tiny = 0.0000001;

	//force double for Rain on grid cases
	if (!XParam.Rainongrid.inputfile.empty())
	{
		XParam.doubleprecision = 1;
	}

	///////////////////////////////////////////
	//  Read Bathy Information
	///////////////////////////////////////////

	//this sets xo yo  etc...

	// Any of xo,yo,xmax,ymax or dx not defined is assigned the value from bathy file
	//default value is nan in default param file
	XParam.Bathymetry = readBathyhead(XParam.Bathymetry);
	if (std::isnan(XParam.xo))
		XParam.xo = XParam.Bathymetry.xo;
	if (std::isnan(XParam.xmax))
		XParam.xmax = XParam.Bathymetry.xmax;
	if(std::isnan(XParam.yo))
		XParam.yo = XParam.Bathymetry.yo;
	if (std::isnan(XParam.ymax))
		XParam.ymax = XParam.Bathymetry.ymax;

	if (std::isnan(XParam.dx))
		XParam.dx = XParam.Bathymetry.dx;

	if (std::isnan(XParam.grdalpha))
		XParam.grdalpha = XParam.Bathymetry.grdalpha; // here the default bathy grdalpha is 0.0 as defined by inputmap/Bathymetry class


	//Check Bathy input type
	std::string bathyext;
	std::vector<std::string> extvec = split(XParam.Bathymetry.inputfile, '.');
	bathyext = extvec.back();
	if (bathyext.compare("nc") == 0)
	{
		if (abs(XParam.grdalpha - DefaultParams.grdalpha) < tiny)
		{
			
			log("For nc of bathy file please specify grdalpha in the XBG_param.txt (if different then default [0])");
		}
	}
	if (bathyext.compare("dep") == 0 || bathyext.compare("bot") == 0)
	{
		if (XParam.nx <= 0 || XParam.ny <= 0 || XParam.dx < tiny)
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

	return XParam;
}

double setendtime(Param XParam)
{
	//endtime cannot be bigger thn the smallest time set in a boundary
	SLTS tempSLTS;
	double endtime = XParam.endtime;
	if (XParam.leftbnd.on)
	{
		tempSLTS = XParam.leftbnd.data.back();
		endtime = min( endtime, tempSLTS.time);
		
	}
	if (XParam.rightbnd.on)
	{
		tempSLTS = XParam.rightbnd.data.back();
		endtime = min(endtime, tempSLTS.time);
	}
	if (XParam.topbnd.on)
	{
		tempSLTS = XParam.topbnd.data.back();
		endtime = min(endtime, tempSLTS.time);
	}
	if (XParam.botbnd.on)
	{
		tempSLTS = XParam.botbnd.data.back();
		endtime = min(endtime, tempSLTS.time);
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

inputmap readcfmaphead(inputmap Roughmap)
{
	// Read critical parameter for the roughness map or deformation file grid input
	write_text_to_log_file("Rougness map was specified. Checking file... " );
	std::string fileext;
	double dummy;
	std::vector<std::string> extvec = split(Roughmap.inputfile, '.');

	std::vector<std::string> nameelements;
	//by default we expect tab delimitation
	nameelements = split(extvec.back(), '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		fileext = nameelements[0];
	}
	else
	{
		fileext = extvec.back();
	}
	write_text_to_log_file("cfmap file extension : " + fileext);

	if (fileext.compare("md") == 0)
	{
		write_text_to_log_file("Reading 'md' file");
		readbathyHeadMD(Roughmap.inputfile, Roughmap.nx, Roughmap.ny, Roughmap.dx, dummy);
		Roughmap.xo = 0.0;
		Roughmap.yo = 0.0;
		Roughmap.xmax = (Roughmap.nx - 1)*Roughmap.dx;
		Roughmap.ymax = (Roughmap.ny - 1)*Roughmap.dx;
	}
	if (fileext.compare("nc") == 0)
	{
		int dummy;
		double dummya, dummyb;
		write_text_to_log_file("Reading cfmap as netcdf file");
		readgridncsize(Roughmap.inputfile, Roughmap.nx, Roughmap.ny,dummy, Roughmap.dx, Roughmap.xo, Roughmap.yo, dummya, Roughmap.xmax, Roughmap.ymax, dummyb);
		//write_text_to_log_file("For nc of bathy file please specify grdalpha in the BG_param.txt (default 0)");
		//Roughmap.xo = 0.0;
		//Roughmap.yo = 0.0;
		//Roughmap.xmax = (Roughmap.nx - 1)*Roughmap.dx;
		//Roughmap.ymax = (Roughmap.ny - 1)*Roughmap.dx;

	}
	if (fileext.compare("dep") == 0 || fileext.compare("bot") == 0)
	{
		//XBeach style file
		//write_text_to_log_file("Reading " + bathyext + " file");
		//write_text_to_log_file("For this type of bathy file please specify nx, ny, dx, xo, yo and grdalpha in the XBG_param.txt");
	}
	if (fileext.compare("asc") == 0)
	{
		//
		write_text_to_log_file("Reading cfmap as asc file");
		readbathyASCHead(Roughmap.inputfile, Roughmap.nx, Roughmap.ny, Roughmap.dx, Roughmap.xo, Roughmap.yo, dummy);
		
		Roughmap.xmax = Roughmap.xo + (Roughmap.nx - 1)*Roughmap.dx;
		Roughmap.ymax = Roughmap.yo + (Roughmap.ny - 1)*Roughmap.dx;
	}






	return Roughmap;
}

void readmapdata(inputmap Roughmap, float * &cfmapinput)
{
	// Check extension 
	std::string fileext;

	std::vector<std::string> extvec = split(Roughmap.inputfile, '.');

	std::vector<std::string> nameelements;
	//by default we expect tab delimitation
	nameelements = split(extvec.back(), '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		fileext = nameelements[0];
	}
	else
	{
		fileext = extvec.back();
	}

	//Now choose the right function to read the data

	if (fileext.compare("md") == 0)
	{
		readbathyMD(Roughmap.inputfile, cfmapinput);
	}
	if (fileext.compare("nc") == 0)
	{
		readnczb(Roughmap.nx, Roughmap.ny, Roughmap.inputfile, cfmapinput);
	}
	if (fileext.compare("bot") == 0 || fileext.compare("dep") == 0)
	{
		readXBbathy(Roughmap.inputfile, Roughmap.nx, Roughmap.ny, cfmapinput);
	}
	if (fileext.compare("asc") == 0)
	{
		//
		readbathyASCzb(Roughmap.inputfile, Roughmap.nx, Roughmap.ny, cfmapinput);
	}

	//return 1;
}

forcingmap readforcingmaphead(forcingmap Fmap)
{
	// Read critical parameter for the forcing map
	write_text_to_log_file("Forcing map was specified. Checking file... ");
	std::string fileext;
	double dummy;
	std::vector<std::string> extvec = split(Fmap.inputfile, '.');

	std::vector<std::string> nameelements;
	//by default we expect tab delimitation
	nameelements = split(extvec.back(), '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		fileext = nameelements[0];
	}
	else
	{
		fileext = extvec.back();
	}
	

	if (fileext.compare("nc") == 0)
	{
		write_text_to_log_file("Reading Forcing file as netcdf file");
		readgridncsize(Fmap.inputfile, Fmap.nx, Fmap.ny, Fmap.nt, Fmap.dx, Fmap.xo, Fmap.yo, Fmap.to, Fmap.xmax, Fmap.ymax, Fmap.tmax);
		

	}
	else
	{
		write_text_to_log_file("Forcing file needs to be a .nc file you also need to specify the netcdf variable name like this ncfile.nc?myvar");
	}
	


	return Fmap;
}

inputmap readBathyhead(inputmap BathyParam)
{
	std::string bathyext;

	//read bathy and perform sanity check

	if (!BathyParam.inputfile.empty())
	{
		printf("bathy: %s\n", BathyParam.inputfile.c_str());

		write_text_to_log_file("bathy: " + BathyParam.inputfile);

		std::vector<std::string> extvec = split(BathyParam.inputfile, '.');

		std::vector<std::string> nameelements;
		//by default we expect tab delimitation
		nameelements = split(extvec.back(), '?');
		if (nameelements.size() > 1)
		{
			//variable name for bathy is not given so it is assumed to be zb
			bathyext = nameelements[0];
		}
		else
		{
			bathyext = extvec.back();
		}


		write_text_to_log_file("bathy extension: " + bathyext);
		if (bathyext.compare("md") == 0)
		{
			write_text_to_log_file("Reading 'md' file");
			readbathyHeadMD(BathyParam.inputfile, BathyParam.nx, BathyParam.ny, BathyParam.dx, BathyParam.grdalpha);
			BathyParam.xo = 0.0;
			BathyParam.yo = 0.0;
			BathyParam.xmax = BathyParam.xo + (BathyParam.nx - 1)*BathyParam.dx;
			BathyParam.ymax = BathyParam.yo + (BathyParam.ny - 1)*BathyParam.dx;

		}
		if (bathyext.compare("nc") == 0)
		{
			int dummy;
			double dummya, dummyb, dummyc;
			write_text_to_log_file("Reading bathy netcdf file");
			readgridncsize(BathyParam.inputfile, BathyParam.nx, BathyParam.ny, dummy, BathyParam.dx, BathyParam.xo, BathyParam.yo, dummyb, BathyParam.xmax, BathyParam.ymax, dummyc);
			write_text_to_log_file("For nc of bathy file please specify grdalpha in the BG_param.txt (default 0)");


		}
		if (bathyext.compare("dep") == 0 || bathyext.compare("bot") == 0)
		{
			//XBeach style file
			write_text_to_log_file("Reading " + bathyext + " file");
			write_text_to_log_file("For this type of bathy file please specify nx, ny, dx, xo, yo and grdalpha in the XBG_param.txt");
		}
		if (bathyext.compare("asc") == 0)
		{
			//
			write_text_to_log_file("Reading bathy asc file");
			readbathyASCHead(BathyParam.inputfile, BathyParam.nx, BathyParam.ny, BathyParam.dx, BathyParam.xo, BathyParam.yo, BathyParam.grdalpha);
			BathyParam.xmax = BathyParam.xo + (BathyParam.nx-1)*BathyParam.dx;
			BathyParam.ymax = BathyParam.yo + (BathyParam.ny-1)*BathyParam.dx;
			write_text_to_log_file("For asc of bathy file please specify grdalpha in the BG_param.txt (default 0)");
		}

		

		//XParam.nx = ceil(XParam.nx / 16) * 16;
		//XParam.ny = ceil(XParam.ny / 16) * 16;



		printf("Bathymetry grid info: nx=%d\tny=%d\tdx=%lf\talpha=%f\txo=%lf\tyo=%lf\txmax=%lf\tymax=%lf\n", BathyParam.nx, BathyParam.ny, BathyParam.dx, BathyParam.grdalpha * 180.0 / pi, BathyParam.xo, BathyParam.yo, BathyParam.xmax, BathyParam.ymax);
		write_text_to_log_file("Bathymetry grid info: nx=" + std::to_string(BathyParam.nx) + " ny=" + std::to_string(BathyParam.ny) + " dx=" + std::to_string(BathyParam.dx) + " grdalpha=" + std::to_string(BathyParam.grdalpha*180.0 / pi) + " xo=" + std::to_string(BathyParam.xo) + " yo=" + std::to_string(BathyParam.yo));






	}
	else
	{
		std::cerr << "Fatal error: No bathymetry file specified. Please specify using 'bathy = Filename.bot'" << std::endl;
		write_text_to_log_file("Fatal error : No bathymetry file specified. Please specify using 'bathy = Filename.md'");
		exit(1);
	}
	return BathyParam;
}

void readbathyHeadMD(std::string filename, int &nx, int &ny, double &dx, double &grdalpha)
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





extern "C" void readbathyMD(std::string filename, float *&zb)
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




double interptime(double next, double prev, double timenext, double time)
{
	return prev + (time) / (timenext)*(next - prev);
}

void readbathyASCHead(std::string filename, int &nx, int &ny, double &dx, double &xo, double &yo, double &grdalpha)
{
	std::ifstream fs(filename);

	if (fs.fail()) {
		std::cerr << filename << " bathy file (md file) could not be opened" << std::endl;
		write_text_to_log_file("ERROR: bathy file could not be opened ");
		exit(1);
	}

	std::string line,left,right;
	std::vector<std::string> lineelements;
	//std::size_t found;
	//std::getline(fs, line);
	int linehead = 0;

	while (linehead < 6)
	{
		std::getline(fs, line);
		// skip empty lines
		if (!line.empty())
		{

			//by default we expect tab delimitation
			lineelements = split(line, ' ');
			if (lineelements.size() < 2)
			{
				lineelements = split(line, '\t');

			}




			left = trim(lineelements[0], " ");
			right = lineelements[1]; 
			//printf("left: %s ;right: %s\n", left.c_str(), right.c_str());
			//found = left.compare("ncols");// it needs to strictly compare
			if (left.compare("ncols") == 0) // found the parameter
			{

				//
				nx = std::stoi(right);

			}

			if (left.compare("nrows") == 0) // found the parameter
			{

				//
				ny = std::stoi(right);

			}
			if (left.compare("cellsize") == 0) // found the parameter
			{

				//
				dx = std::stod(right);

			}
			if (left.compare("xllcenter") == 0) // found the parameter
			{

				//
				xo = std::stod(right);

			}
			if (left.compare("yllcenter") == 0) // found the parameter
			{

				//
				yo = std::stod(right);

			}
			//if gridnode registration this should happen
			if (left.compare("xllcorner") == 0) // found the parameter
			{

				//
				xo = std::stod(right);

			}
			if (left.compare("yllcorner") == 0) // found the parameter
			{

				//
				yo = std::stod(right);
				//This should be:
				//yo = std::stod(right) + dx / 2.0;
				//but by the time xo and yo are found dx has not been setup... awkward...

			}
			linehead++;
		}
	}
	grdalpha = 0.0;
	fs.close();

}

void readbathyASCzb(std::string filename,int nx, int ny, float* &zb)
{
	//
	std::ifstream fs(filename);
	int linehead = 0;
	std::string line;
	if (fs.fail()) {
		std::cerr << filename << " bathy file (md file) could not be opened" << std::endl;
		write_text_to_log_file("ERROR: bathy file could not be opened ");
		exit(1);
	}
	while (linehead < 6)
	{
		//Skip header
		std::getline(fs, line);
		linehead++;
	}
	//int jreadzs;
	for (int jnod = ny-1; jnod >= 0; jnod--)
	{



		for (int inod = 0; inod < nx; inod++)
		{
			//fscanf(fid, "%f", &zb[inod + (jnod)*nx]);

			fs >> zb[inod + (jnod)*nx];
			//printf("%f\n", zb[inod + (jnod)*nx]);

		}
	}

	fs.close();
}

double BilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y)
{
	double x2x1, y2y1, x2x, y2y, yy1, xx1;
	x2x1 = x2 - x1;
	y2y1 = y2 - y1;
	x2x = x2 - x;
	y2y = y2 - y;
	yy1 = y - y1;
	xx1 = x - x1;
	return 1.0 / (x2x1 * y2y1) * (
		q11 * x2x * y2y +
		q21 * xx1 * y2y +
		q12 * x2x * yy1 +
		q22 * xx1 * yy1
		);
}
double BarycentricInterpolation(double q1, double x1,double y1,double q2, double x2, double y2, double q3, double x3, double y3,double x, double y)
{
	double w1, w2, w3,D;

	D = (y2 - y3) * (x1 + x3) + (x3-x2) * (y1-y3);

	w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / D;
	w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / D;
	w3 = 1 - w1 - w2;

	return q1 * w1 + q2 * w2 + q3 * w3;
}

