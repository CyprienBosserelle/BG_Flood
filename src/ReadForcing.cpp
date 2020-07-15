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




#include "ReadForcing.h"

template <class T>
void readforcing(Param & XParam, Forcing<T> & XForcing)
{
	//=================
	// Read Bathymetry
	
	//First allocate the array for storing the data
	AllocateCPU(XForcing.Bathy.nx, XForcing.Bathy.ny, XForcing.Bathy.val);

	log("Read Bathy data...");
	
	readbathydata(XParam.posdown,XForcing.Bathy, XForcing.Bathy.val);

	log("...done");


	//=================
	// Read bnd files
	log("Read Bnd data...");

	if (!XParam.leftbnd.inputfile.empty())
	{
		//XParam.leftbnd.data = readWLfile(XParam.leftbnd.inputfile);
		XParam.leftbnd.data = readbndfile(XParam.leftbnd.inputfile, XParam, 0);

		XParam.leftbnd.on = 1; // redundant?
	}
	if (!XParam.rightbnd.inputfile.empty())
	{
		XParam.rightbnd.data = readbndfile(XParam.rightbnd.inputfile, XParam, 2);
		XParam.rightbnd.on = 1;
	}
	if (!XParam.topbnd.inputfile.empty())
	{
		XParam.topbnd.data = readbndfile(XParam.topbnd.inputfile, XParam, 3);
		XParam.topbnd.on = 1;
	}
	if (!XParam.botbnd.inputfile.empty())
	{
		XParam.botbnd.data = readbndfile(XParam.botbnd.inputfile, XParam, 1);
		XParam.botbnd.on = 1;
	}

	//Check that endtime is no longer than boundaries (if specified to other than wall or neumann)
	XParam.endtime = setendtime(XParam);

	log("...done");

	//==================
	// Friction maps 
		
	if (!XForcing.cf.inputfile.empty())
	{
		log("Read Roughness map (cf) data...");
		// roughness map was specified!
		readstaticforcing(XForcing.cf);

		log("...done");
	}
	
	//=====================
	// Deformation files
	if (XForcing.deform.size() > 0)
	{
		log("Read deform data...");
		// Deformation files was specified!

		for (int nd = 0; nd < XForcing.deform.size(); nd++)
		{
			// read the roughness map header
			readstaticforcing(XForcing.deform[nd]);
			//XForcing.deform[nd].grid = readcfmaphead(XForcing.deform[nd].grid);
			
			XParam.deformmaxtime = utils::max(XParam.deformmaxtime, XForcing.deform[nd].startime + XForcing.deform[nd].duration);
		}
		log("...done");

	}

	//======================
	// Wind file


}

template void readforcing<float>(Param& XParam, Forcing<float>& XForcing);
//template void readforcing<double>(Param& XParam, Forcing<double>& XForcing);

template <class T>
void readstaticforcing(T& Sforcing)
{
	inputmap cfmap;
	cfmap.inputfile = Sforcing.inputfile;
	cfmap = readBathyhead(cfmap);
	Sforcing.xo = cfmap.xo;
	Sforcing.yo = cfmap.yo;
	Sforcing.xmax = cfmap.xmax;
	Sforcing.ymax = cfmap.ymax;
	Sforcing.nx = cfmap.nx;
	Sforcing.ny = cfmap.ny;

	if (Sforcing.nx > 0 && Sforcing.ny > 0)
	{
		AllocateCPU(Sforcing.nx, Sforcing.ny, Sforcing.val);

		// read the roughness map header
		readbathydata(0, Sforcing, Sforcing.val);

	}
	else
	{
		//Error message
		log("Error while reading forcing map file: "+ Sforcing.inputfile);
	}
}

template void readstaticforcing<deformmap<float>>(deformmap<float>& Sforcing);
template void readstaticforcing<StaticForcingP<float>>(StaticForcingP<float>& Sforcing);


void readbathydata(int posdown, inputmap bathymeta,float * &dummy)
{
	// Check bathy extension
	std::string bathyext;

	std::vector<std::string> extvec = split(bathymeta.inputfile, '.');

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

	//Now choose the right function to read the data

	if (bathyext.compare("md") == 0)
	{
		readbathyMD(bathymeta.inputfile, dummy);
	}
	if (bathyext.compare("nc") == 0)
	{
		readnczb(bathymeta.nx, bathymeta.ny, bathymeta.inputfile, dummy);
	}
	if (bathyext.compare("bot") == 0 || bathyext.compare("dep") == 0)
	{
		readXBbathy(bathymeta.inputfile, bathymeta.nx, bathymeta.ny, dummy);
	}
	if (bathyext.compare("asc") == 0)
	{
		//
		readbathyASCzb(bathymeta.inputfile, bathymeta.nx, bathymeta.ny, dummy);
	}

	if (posdown == 1)
	{
		
		log("Bathy data is positive down...correcting");
		for (int j = 0; j < bathymeta.ny; j++)
		{
			for (int i = 0; i < bathymeta.nx; i++)
			{
				dummy[i + j * bathymeta.nx] = dummy[i + j * bathymeta.nx] * -1.0f;
				//printf("%f\n", zb[i + (j)*nx]);

			}
		}
	}
}



/*! \fn std::vector<SLTS> readbndfile(std::string filename,Param XParam, int side)
* Read boundary files
* 
*/
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
			indxx = utils::max(utils::min((int)((bndxo+(dx*ibnd) - xo) / dx), nx - 1), 0);
			indyy = utils::max(utils::min((int)((bndy - yo) / dx), ny - 1), 0);

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
				//std::cerr << Flowfilename << "ERROR flow file format error. only " << lineelements.size() << " where at least 2 were expected. Exiting." << std::endl;
				log("ERROR:  flow file (" + Flowfilename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 2 were expected. Exiting.");
				log(line);
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
		//std::cerr << filename << "ERROR: Atm presssure / Rainfall file could not be opened" << std::endl;
		log("ERROR: Atm presssure / Rainfall file could not be opened : " + filename);
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
				//std::cerr << filename << "ERROR Atm presssure / Rainfall  file format error. only " << lineelements.size() << " where at least 2 were expected. Exiting." << std::endl;
				log("ERROR:  Atm presssure / Rainfall file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 2 were expected. Exiting.");
				log(line);
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
		//std::cerr << filename << "ERROR: Wind file could not be opened" << std::endl;
		log("ERROR: Wind file could not be opened : "+ filename);
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
				//std::cerr << filename << "ERROR Wind  file format error. only " << lineelements.size() << " where at least 3 were expected. Exiting." << std::endl;
				log("ERROR:  Wind file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 3 were expected. Exiting.");
				log(line);
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



inputmap readcfmaphead(inputmap Roughmap)
{
	// Read critical parameter for the roughness map or deformation file grid input
	log("Rougness map was specified. Checking file... "+ Roughmap.inputfile);
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
	log("cfmap file extension : " + fileext);

	if (fileext.compare("md") == 0)
	{
		log("Reading 'md' file");
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
		log("Reading cfmap as netcdf file");
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
		log("Reading cfmap as asc file");
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
	log("Forcing map was specified. Checking file... ");
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
		log("Reading Forcing file as netcdf file");
		readgridncsize(Fmap.inputfile, Fmap.nx, Fmap.ny, Fmap.nt, Fmap.dx, Fmap.xo, Fmap.yo, Fmap.to, Fmap.xmax, Fmap.ymax, Fmap.tmax);
		

	}
	else
	{
		log("Forcing file needs to be a .nc file you also need to specify the netcdf variable name like this ncfile.nc?myvar");
	}
	


	return Fmap;
}

inputmap readBathyhead(inputmap BathyParam)
{
	std::string bathyext;

	//read bathy and perform sanity check

	if (!BathyParam.inputfile.empty())
	{
		//printf("bathy: %s\n", BathyParam.inputfile.c_str());

		log("Reading bathy metadata. file: " + BathyParam.inputfile);

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
		BathyParam.extension = bathyext;


		log("Bathy file extension: " + bathyext);
		if (bathyext.compare("md") == 0)
		{
			log("'md' file");
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
			log("netcdf file");
			readgridncsize(BathyParam.inputfile, BathyParam.nx, BathyParam.ny, dummy, BathyParam.dx, BathyParam.xo, BathyParam.yo, dummyb, BathyParam.xmax, BathyParam.ymax, dummyc);
			//log("For nc of bathy file please specify grdalpha in the BG_param.txt (default 0)");


		}
		if (bathyext.compare("dep") == 0 || bathyext.compare("bot") == 0)
		{
			//XBeach style file
			log( bathyext + " file");
			log("For this type of bathy file please specify nx, ny, dx, xo, yo and grdalpha in the XBG_param.txt");
		}
		if (bathyext.compare("asc") == 0)
		{
			//
			log("asc file");
			readbathyASCHead(BathyParam.inputfile, BathyParam.nx, BathyParam.ny, BathyParam.dx, BathyParam.xo, BathyParam.yo, BathyParam.grdalpha);
			BathyParam.xmax = BathyParam.xo + (BathyParam.nx-1)*BathyParam.dx;
			BathyParam.ymax = BathyParam.yo + (BathyParam.ny-1)*BathyParam.dx;
			log("For asc of bathy file please specify grdalpha in the BG_param.txt (default 0)");
		}

		

		//XParam.nx = ceil(XParam.nx / 16) * 16;
		//XParam.ny = ceil(XParam.ny / 16) * 16;



		//printf("Bathymetry grid info: nx=%d\tny=%d\tdx=%lf\talpha=%f\txo=%lf\tyo=%lf\txmax=%lf\tymax=%lf\n", BathyParam.nx, BathyParam.ny, BathyParam.dx, BathyParam.grdalpha * 180.0 / pi, BathyParam.xo, BathyParam.yo, BathyParam.xmax, BathyParam.ymax);
		log("Bathymetry grid info: nx=" + std::to_string(BathyParam.nx) + " ny=" + std::to_string(BathyParam.ny) + " dx=" + std::to_string(BathyParam.dx) + " grdalpha=" + std::to_string(BathyParam.grdalpha*180.0 / pi) + " xo=" + std::to_string(BathyParam.xo) + " xmax=" + std::to_string(BathyParam.xmax) + " yo=" + std::to_string(BathyParam.yo) + " ymax=" + std::to_string(BathyParam.ymax));






	}
	else
	{
		std::cerr << "Fatal error: No bathymetry file specified. Please specify using 'bathy = Filename.bot'" << std::endl;
		log("Fatal error : No bathymetry file specified. Please specify using 'bathy = Filename.md'");
		exit(1);
	}
	return BathyParam;
}

void readbathyHeadMD(std::string filename, int &nx, int &ny, double &dx, double &grdalpha)
{
	

	std::ifstream fs(filename);

	if (fs.fail()) {
		std::cerr << filename << " bathy file (md file) could not be opened" << std::endl;
		log("ERROR: bathy file could not be opened "+ filename);
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
			log("ERROR:  Wind bnd file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where 3 were expected. Exiting.");
			log(line);
			exit(1);
		}

		nx = std::stoi(lineelements[0]);
		ny = std::stoi(lineelements[1]);
		dx = std::stod(lineelements[2]);
		grdalpha = std::stod(lineelements[4]);
	}

	fs.close();
}


void readbathyMD(std::string filename, float*& zb)
{
	// Shit that doesn'y wor... Needs fixing 
	int nx, ny;
	float dx, grdalpha;
	std::ifstream fs(filename);

	if (fs.fail()) {
		std::cerr << filename << " bathy file (md file) could not be opened" << std::endl;
		log("ERROR: bathy file could not be opened " + filename);
		exit(1);
	}

	std::string line;

	std::vector<std::string> lineelements;

	std::getline(fs, line);
	if (!line.empty() && line.substr(0, 1).compare("#") != 0)
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
			log("ERROR:  Wind bnd file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where 3 were expected. Exiting.");
			log(line);
			exit(1);
		}

		nx = std::stoi(lineelements[0]);
		ny = std::stoi(lineelements[1]);
		dx = std::stod(lineelements[2]);
		grdalpha = std::stod(lineelements[4]);
	}

	int j = 0;
	while (std::getline(fs, line))
	{
		//std::cout << line << std::endl;

		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			lineelements = split(line, '\t');
			for (int i = 0; i < nx; i++)
			{
				zb[i + j * nx] = std::stof(lineelements[0]);
			}
			j++;
		}
	}

	fs.close();

}



 void readXBbathy(std::string filename, int nx,int ny, float *&zb)
{
	//read input data:
	//printf("bathy: %s\n", filename);
	
	
	//read md file
	 std::ifstream fs(filename);
	 std::string line;
	 std::vector<std::string> lineelements;

	 
	

	
	//int jreadzs;
	for (int jnod = 0; jnod < ny; jnod++)
	{

		std::getline(fs, line);

		for (int inod = 0; inod < nx; inod++)
		{
			//fscanf(fid, "%f", &zb[inod + (jnod)*nx]);
			zb[inod + jnod * nx] = std::stof(lineelements[0]);

		}
	}
	fs.close();
	//fclose(fid);
}






void readbathyASCHead(std::string filename, int &nx, int &ny, double &dx, double &xo, double &yo, double &grdalpha)
{
	std::ifstream fs(filename);

	if (fs.fail()) {
		std::cerr << filename << " bathy file (md file) could not be opened" << std::endl;
		log("ERROR: bathy file could not be opened " + filename);
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
		log("ERROR: bathy file could not be opened " + filename);
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



