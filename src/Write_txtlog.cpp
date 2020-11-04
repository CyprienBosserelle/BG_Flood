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

	std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

	std::string s(30, '\0');
	std::tm buf;
	localtime_s(&buf, &now);
	std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", &buf);

	//strftime(buffer, 80, "%d-%m-%Y %H:%M:%S", timeinfo);
	//std::string strtimenow(buffer);
	log("#################################");
	log("BG_Flood v0.5");
	log("#################################");
	log("model started at " + s);
	log("#################################");
	log("#");
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
	write_text_to_log_file("outfile = " + XParam.outfile + ";");
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
