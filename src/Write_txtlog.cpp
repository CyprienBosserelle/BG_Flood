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

#include "Write_txtlog.h"



void log(std::string text)
{

	std::cout << text << std::endl;
	write_text_to_log_file(text);

}

void create_logfile()
{
	// Reset the log file
	std::ofstream log_file(
		"BG_log.txt", std::ios_base::out | std::ios_base::trunc);

	log_file.close();

	//Logfile header
	//auto n = std::chrono::system_clock::now();
	//auto in_time_t = std::chrono::system_clock::to_time_t(n);
	//std::tm buf;
	//localtime_s(&buf, &in_time_t);
	//std::cout << std::put_time(&buf, "%Y-%m-%d %X") << std::endl;
/*
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	 std::stringstream ss;
	 ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
	 */
/*
	std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

	std::string s(30, '\0');
	std::tm buf;
	struct tm * timeinfo;
	std::localtime_s(&buf, &now);
	//std::time_t rawtime;
	//timeinfo = localtime(&rawtime);
	std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", &timeinfo);
*/
	//strftime(buffer, 80, "%d-%m-%Y %H:%M:%S", timeinfo);
	//std::string strtimenow(buffer);

	time_t rawtime;
	struct tm* timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, 80, "%d-%m-%Y %H:%M:%S", timeinfo);
	std::string strtimenow(buffer);


	log("#################################");
	log("BG_Flood v0.8");
	log("#################################");
	//log("model started at " + ss.str());
	log("#################################");
	log("#");

	write_text_to_log_file("model started at " + strtimenow);
}

void write_text_to_log_file(std::string text)
{
	std::ofstream log_file(
		"BG_log.txt", std::ios_base::out | std::ios_base::app);
	log_file << text << std::endl;
	log_file.close(); //destructor implicitly does it
}

void SaveParamtolog(Param XParam)// need to bring in Xforcing info too!
{
	write_text_to_log_file("\n");
	write_text_to_log_file("###################################");
	write_text_to_log_file("### Summary of model parameters ###");
	write_text_to_log_file("###################################");
	write_text_to_log_file("# Bathymetry file");
	//write_text_to_log_file("bathy = " + XParam.Bathymetry.inputfile + ";");
	write_text_to_log_file("posdown = " + std::to_string(XParam.posdown) + ";");
	//write_text_to_log_file("nx = " + std::to_string(XParam.nx) + ";");
	//write_text_to_log_file("ny = " + std::to_string(XParam.ny) + ";");
	write_text_to_log_file("dx = " + std::to_string(XParam.dx) + ";");
	write_text_to_log_file("delta = " + std::to_string(XParam.delta) + ";");
	write_text_to_log_file("grdalpha = " + std::to_string(XParam.grdalpha*180.0 / pi) + ";");
	write_text_to_log_file("xo = " + std::to_string(XParam.xo) + ";");
	write_text_to_log_file("yo = " + std::to_string(XParam.yo) + ";");
	write_text_to_log_file("xmax = " + std::to_string(XParam.xo) + ";");
	write_text_to_log_file("ymax = " + std::to_string(XParam.yo) + ";");
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
	write_text_to_log_file("totaltime = " + std::to_string(XParam.totaltime) + "; # Start time");
	write_text_to_log_file("endtime = " + std::to_string(XParam.endtime) + ";");
	write_text_to_log_file("outputtimestep = " + std::to_string(XParam.outputtimestep) + ";");


	std::string alloutvars = "";
	for (int nvar = 0; nvar < XParam.outvars.size(); nvar++)
	{
		if (nvar > 0)
		{
			alloutvars = alloutvars + ", ";
		}
		alloutvars = alloutvars + XParam.outvars[nvar];
	}
	write_text_to_log_file("outvars = " + alloutvars + ";");



	write_text_to_log_file("\n");
	write_text_to_log_file("# Files");
	if (!XParam.outzone.empty())
	{
		for (int o = 0; o < XParam.outzone.size(); o++)
		{
			write_text_to_log_file("outzonefile = " + XParam.outzone[o].outname + "," + std::to_string(XParam.outzone[o].xstart) + "," + std::to_string(XParam.outzone[o].xend) + "," + std::to_string(XParam.outzone[o].ystart) + "," + std::to_string(XParam.outzone[o].yend) + ";");
		}
	}
	else
	{
		write_text_to_log_file("outfile = " + XParam.outfile + ";");
	}
	
	write_text_to_log_file("smallnc = " + std::to_string(XParam.smallnc) + "; #if smallnc==1 all Output are scaled and saved as a short int");
	if (XParam.smallnc == 1)
	{
		write_text_to_log_file("scalefactor = " + std::to_string(XParam.scalefactor) + ";");
		write_text_to_log_file("addoffset = " + std::to_string(XParam.addoffset) + ";");
	}

	if (!XParam.TSnodesout.empty())
	{
		for (int o = 0; o < XParam.TSnodesout.size(); o++)
		{
			write_text_to_log_file("TSOfile = " + XParam.TSnodesout[o].outname + "," + std::to_string(XParam.TSnodesout[o].x) + "," + std::to_string(XParam.TSnodesout[o].y)+";");
			//write_text_to_log_file("TSnode = " + std::to_string(XParam.TSnodesout[o].i) + "," + std::to_string(XParam.TSnodesout[o].j) + ";");
		}
	}
	write_text_to_log_file("\n");
	write_text_to_log_file("# Boundaries");
	write_text_to_log_file("# 0:wall; 1: Neumann (Default); 2:Dirichlet (zs); 3: abs1d ");
	/*
	//write_text_to_log_file("right = " + std::to_string(XParam.rightbnd.type) + ";");
	//write_text_to_log_file("left = " + std::to_string(XParam.leftbnd.type) + ";");
	//write_text_to_log_file("top = " + std::to_string(XParam.topbnd.type) + ";");
	//write_text_to_log_file("bot = " + std::to_string(XParam.botbnd.type) + ";");

	if (!XParam.rightbnd.inputfile.empty())
		write_text_to_log_file("rightbndfile = " + XParam.rightbnd.inputfile + ";");
	if (!XParam.leftbnd.inputfile.empty())
		write_text_to_log_file("leftbndfile = " + XParam.leftbnd.inputfile + ";");
	if (!XParam.topbnd.inputfile.empty())
		write_text_to_log_file("topbndfile = " + XParam.topbnd.inputfile + ";");
	if (!XParam.botbnd.inputfile.empty())
		write_text_to_log_file("botbndfile = " + XParam.botbnd.inputfile + ";");
*/
	/*
	std::string rightbndfile;
	std::string leftbndfile;
	std::string topbndfile;
	std::string botbndfile;
	*/
	//hot start
	if (!XParam.hotstartfile.empty())
	{
		write_text_to_log_file("hotstartfile = " + XParam.hotstartfile + ";");
		write_text_to_log_file("hotstep = " + std::to_string(XParam.hotstep) + ";");
	}


	write_text_to_log_file("\n");
	write_text_to_log_file("# Others");
	write_text_to_log_file("g = " + std::to_string(XParam.g) + ";");
	write_text_to_log_file("rho = " + std::to_string(XParam.rho) + ";");
	write_text_to_log_file("\n");
}



void saveparam2netCDF(int ncid, int bgfid, Param XParam)
{
	int status, boolanswer;
	//status = nc_put_att_text(ncid, bgfid, "grid_mapping_name", crsname.size(), crsname.c_str());
	//status = nc_put_att_float(ncid, bgfid, "longitude_of_prime_meridian", NC_FLOAT, 1, &primemeridian);

	status = nc_put_att_int(ncid, bgfid, "test", NC_INT, 1, &XParam.test);
	status = nc_put_att_double(ncid, bgfid, "g", NC_DOUBLE, 1, &XParam.g);
	status = nc_put_att_double(ncid, bgfid, "rho", NC_DOUBLE, 1, &XParam.rho);
	status = nc_put_att_double(ncid, bgfid, "eps", NC_DOUBLE, 1, &XParam.eps);
	status = nc_put_att_double(ncid, bgfid, "CFL", NC_DOUBLE, 1, &XParam.CFL);
	status = nc_put_att_double(ncid, bgfid, "theta", NC_DOUBLE, 1, &XParam.theta);
	status = nc_put_att_double(ncid, bgfid, "VelThreshold", NC_DOUBLE, 1, &XParam.VelThreshold);

	status = nc_put_att_int(ncid, bgfid, "frictionmodel", NC_INT, 1, &XParam.frictionmodel);
	
	status = nc_put_att_double(ncid, bgfid, "Cd", NC_DOUBLE, 1, &XParam.Cd);
	status = nc_put_att_double(ncid, bgfid, "Pa2m", NC_DOUBLE, 1, &XParam.Pa2m);
	status = nc_put_att_double(ncid, bgfid, "Paref", NC_DOUBLE, 1, &XParam.Paref);
	status = nc_put_att_double(ncid, bgfid, "lat", NC_DOUBLE, 1, &XParam.lat);

	boolanswer = XParam.windforcing;
	status = nc_put_att_int(ncid, bgfid, "windforcing", NC_INT, 1, &boolanswer);

	boolanswer = XParam.atmpforcing;
	status = nc_put_att_int(ncid, bgfid, "atmpforcing", NC_INT, 1, &boolanswer);

	boolanswer = XParam.rainforcing;
	status = nc_put_att_int(ncid, bgfid, "rainforcing", NC_INT, 1, &boolanswer);

	boolanswer = XParam.infiltration;
	status = nc_put_att_int(ncid, bgfid, "infiltration", NC_INT, 1, &boolanswer);

	boolanswer = XParam.conserveElevation;
	status = nc_put_att_int(ncid, bgfid, "conserveElevation", NC_INT, 1, &boolanswer);

	boolanswer = XParam.wetdryfix;
	status = nc_put_att_int(ncid, bgfid, "wetdryfix", NC_INT, 1, &boolanswer);

	boolanswer = XParam.leftbnd;
	status = nc_put_att_int(ncid, bgfid, "leftbnd", NC_INT, 1, &boolanswer);

	boolanswer = XParam.rightbnd;
	status = nc_put_att_int(ncid, bgfid, "rightbnd", NC_INT, 1, &boolanswer);

	boolanswer = XParam.topbnd;
	status = nc_put_att_int(ncid, bgfid, "topbnd", NC_INT, 1, &boolanswer);

	boolanswer = XParam.botbnd;
	status = nc_put_att_int(ncid, bgfid, "botbnd", NC_INT, 1, &boolanswer);




	
	double cf = 0.0001; // Bottom friction coefficient for flow model (if constant)
	
	
	status = nc_put_att_int(ncid, bgfid, "GPUDEVICE", NC_INT, 1, &XParam.GPUDEVICE);

	status = nc_put_att_int(ncid, bgfid, "doubleprecision", NC_INT, 1, &XParam.doubleprecision);

	status = nc_put_att_int(ncid, bgfid, "engine", NC_INT, 1, &XParam.engine);
	
	status = nc_put_att_double(ncid, bgfid, "dx", NC_DOUBLE, 1, &XParam.dx);
	status = nc_put_att_double(ncid, bgfid, "delta", NC_DOUBLE, 1, &XParam.delta);

	status = nc_put_att_double(ncid, bgfid, "xo", NC_DOUBLE, 1, &XParam.xo);
	status = nc_put_att_double(ncid, bgfid, "yo", NC_DOUBLE, 1, &XParam.yo);
	status = nc_put_att_double(ncid, bgfid, "xmax", NC_DOUBLE, 1, &XParam.xmax);
	status = nc_put_att_double(ncid, bgfid, "ymax", NC_DOUBLE, 1, &XParam.ymax);

	status = nc_put_att_double(ncid, bgfid, "grdalpha", NC_DOUBLE, 1, &XParam.grdalpha);

	status = nc_put_att_int(ncid, bgfid, "nx", NC_INT, 1, &XParam.nx);
	status = nc_put_att_int(ncid, bgfid, "ny", NC_INT, 1, &XParam.ny);
	status = nc_put_att_int(ncid, bgfid, "nblk", NC_INT, 1, &XParam.nblk);

	status = nc_put_att_int(ncid, bgfid, "blkwidth", NC_INT, 1, &XParam.blkwidth);
	status = nc_put_att_int(ncid, bgfid, "blkmemwidth", NC_INT, 1, &XParam.blkmemwidth);
	status = nc_put_att_int(ncid, bgfid, "blksize", NC_INT, 1, &XParam.blksize);
	status = nc_put_att_int(ncid, bgfid, "halowidth", NC_INT, 1, &XParam.halowidth);
	status = nc_put_att_int(ncid, bgfid, "posdown", NC_INT, 1, &XParam.posdown);
	
	boolanswer = XParam.spherical;
	status = nc_put_att_int(ncid, bgfid, "spherical", NC_INT, 1, &boolanswer);
	status = nc_put_att_double(ncid, bgfid, "Radius", NC_DOUBLE, 1, &XParam.Radius);
	status = nc_put_att_double(ncid, bgfid, "mask", NC_DOUBLE, 1, &XParam.mask);

	status = nc_put_att_int(ncid, bgfid, "initlevel", NC_INT, 1, &XParam.initlevel);
	status = nc_put_att_int(ncid, bgfid, "maxlevel", NC_INT, 1, &XParam.maxlevel);
	status = nc_put_att_int(ncid, bgfid, "minlevel", NC_INT, 1, &XParam.minlevel);
	status = nc_put_att_int(ncid, bgfid, "nblkmem", NC_INT, 1, &XParam.nblkmem);
	status = nc_put_att_int(ncid, bgfid, "navailblk", NC_INT, 1, &XParam.navailblk);
	status = nc_put_att_double(ncid, bgfid, "membuffer", NC_DOUBLE, 1, &XParam.membuffer);
	
	status = nc_put_att_double(ncid, bgfid, "outputtimestep", NC_DOUBLE, 1, &XParam.outputtimestep);
	status = nc_put_att_double(ncid, bgfid, "endtime", NC_DOUBLE, 1, &XParam.endtime);
	status = nc_put_att_double(ncid, bgfid, "totaltime", NC_DOUBLE, 1, &XParam.totaltime);
	status = nc_put_att_double(ncid, bgfid, "dtinit", NC_DOUBLE, 1, &XParam.dtinit);
	status = nc_put_att_double(ncid, bgfid, "dtmin", NC_DOUBLE, 1, &XParam.dtmin);

	status = nc_put_att_double(ncid, bgfid, "zsinit", NC_DOUBLE, 1, &XParam.zsinit);
	status = nc_put_att_double(ncid, bgfid, "zsoffset", NC_DOUBLE, 1, &XParam.zsoffset);

	status = nc_put_att_text(ncid, bgfid, "hotstartfile", XParam.hotstartfile.size(), XParam.hotstartfile.c_str());
	status = nc_put_att_int(ncid, bgfid, "hotstep", NC_INT, 1, &XParam.hotstep);

	status = nc_put_att_double(ncid, bgfid, "wet_threshold ", NC_DOUBLE, 1, &XParam.wet_threshold);

	status = nc_put_att_int(ncid, bgfid, "maxTSstorage", NC_INT, 1, &XParam.maxTSstorage);

	boolanswer = XParam.resetmax;
	status = nc_put_att_int(ncid, bgfid, "resetmax", NC_INT, 1, &boolanswer);

	boolanswer = XParam.outmax;
	status = nc_put_att_int(ncid, bgfid, "outmax", NC_INT, 1, &boolanswer);

	boolanswer = XParam.outmean;
	status = nc_put_att_int(ncid, bgfid, "outmean", NC_INT, 1, &boolanswer);

	boolanswer = XParam.outtwet;
	status = nc_put_att_int(ncid, bgfid, "outtwet", NC_INT, 1, &boolanswer);

	status = nc_put_att_int(ncid, bgfid, "outishift", NC_INT, 1, &XParam.outishift);

	status = nc_put_att_int(ncid, bgfid, "outjshift", NC_INT, 1, &XParam.outjshift);

	status = nc_put_att_int(ncid, bgfid, "nrivers", NC_INT, 1, &XParam.nrivers);

	status = nc_put_att_int(ncid, bgfid, "nblkriver", NC_INT, 1, &XParam.nblkriver);

	status = nc_put_att_int(ncid, bgfid, "nbndblkleft", NC_INT, 1, &XParam.nbndblkleft);
	status = nc_put_att_int(ncid, bgfid, "nbndblkright", NC_INT, 1, &XParam.nbndblkright);
	status = nc_put_att_int(ncid, bgfid, "nbndblktop", NC_INT, 1, &XParam.nbndblktop);
	status = nc_put_att_int(ncid, bgfid, "nbndblkbot", NC_INT, 1, &XParam.nbndblkbot);

	status = nc_put_att_int(ncid, bgfid, "nmaskblk", NC_INT, 1, &XParam.nmaskblk);

	status = nc_put_att_int(ncid, bgfid, "smallnc", NC_INT, 1, &XParam.smallnc);

	status = nc_put_att_float(ncid, bgfid, "scalefactor", NC_FLOAT, 1, &XParam.scalefactor);

	status = nc_put_att_float(ncid, bgfid, "addoffset", NC_FLOAT, 1, &XParam.addoffset);

	status = nc_put_att_double(ncid, bgfid, "deformmaxtime", NC_DOUBLE, 1, &XParam.deformmaxtime);

	boolanswer = XParam.rainbnd;
	status = nc_put_att_int(ncid, bgfid, "rainbnd", NC_INT, 1, &boolanswer);

	status = nc_put_att_text(ncid, bgfid, "AdaptCrit", XParam.AdaptCrit.size(), XParam.AdaptCrit.c_str());

	status = nc_put_att_text(ncid, bgfid, "Adapt_arg1", XParam.Adapt_arg1.size(), XParam.Adapt_arg1.c_str());
	status = nc_put_att_text(ncid, bgfid, "Adapt_arg2", XParam.Adapt_arg2.size(), XParam.Adapt_arg2.c_str());
	status = nc_put_att_text(ncid, bgfid, "Adapt_arg3", XParam.Adapt_arg3.size(), XParam.Adapt_arg3.c_str());
	status = nc_put_att_text(ncid, bgfid, "Adapt_arg4", XParam.Adapt_arg4.size(), XParam.Adapt_arg4.c_str());
	status = nc_put_att_text(ncid, bgfid, "Adapt_arg5", XParam.Adapt_arg5.size(), XParam.Adapt_arg5.c_str());

	status = nc_put_att_int(ncid, bgfid, "adaptmaxiteration", NC_INT, 1, &XParam.adaptmaxiteration);

	status = nc_put_att_text(ncid, bgfid, "reftime", XParam.reftime.size(), XParam.reftime.c_str());


	std::string allouvars;
	for (int i = 0; i < XParam.outvars.size(); i++)
	{
		allouvars = allouvars + XParam.outvars[i];
		if (i < (XParam.outvars.size() - 1))
		{
			allouvars = allouvars + ", ";
		}
	}

	status = nc_put_att_text(ncid, bgfid, "outvars", allouvars.size(), allouvars.c_str());

	status = nc_put_att_text(ncid, bgfid, "outfile", XParam.outfile.size(), XParam.outfile.c_str());

	



	/*
	
	std::vector<TSoutnode> TSnodesout;
	

	
	

	std::vector<outzoneP> outzone;
	
	// deformation forcing for tsunami generation
	//std::vector<deformmap> deform;
	
	
	std::string AdaptCrit;
	int* AdaptCrit_funct_pointer;

	
	*/
}
