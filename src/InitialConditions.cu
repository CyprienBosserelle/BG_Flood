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


#include "InitialConditions.h"

template <class T> void InitialConditions(Param XParam, Forcing<float> XForcing, Model<T> XModel)
{
	//========================
	// Initialise Bathy data

	interp2BUQ(XParam, XModel.blocks, XForcing.Bathy, XModel.zb);

	// Set edges
	setedges(XParam, XModel.blocks, XModel.zb);

	//=========================
	// Initialise Friction map

	if (!XForcing.cf.inputfile.empty())
	{
		interp2BUQ(XParam, XModel.blocks, XForcing.cf, XModel.cf);
	}
	else
	{
		InitArrayBUQ(XParam.nblk, XParam.blkwidth, XParam.halowidth, (T)XParam.cf, XModel.cf);
	}
	// Set edges of friction map
	setedges(XParam, XModel.blocks, XModel.cf);

	//=====================================
	// Initial Condition
	
	log("Initial condition:");

	

}

template void InitialConditions<float>(Param XParam, Forcing<float> XForcing, Model<float> XModel);
template void InitialConditions<double>(Param XParam, Forcing<float> XForcing, Model<double> XModel);

template <class T> void initevolv(Param XParam, Forcing<float> XForcing, EvolvingP<T> XEv)
{
	//move this to a subroutine
	int hotstartsucess = 0;
	if (!XParam.hotstartfile.empty())
	{
		// hotstart
		log("Hotstart file used : " + XParam.hotstartfile);
		/*
		hotstartsucess = readhotstartfile(XParam, leftblk, rightblk, topblk, botblk, blockxo_d, blockyo_d, zs, zb, hh, uu, vv);

		//add offset if present
		if (abs(XParam.zsoffset - defaultParam.zsoffset) > epsilon) // apply specified zsoffset
		{
			printf("add offset to zs and hh... ");
			//
			AddZSoffset(XParam, zb, zs, hh);
		}
		*/
	}
	if (hotstartsucess == 0)
	{
		printf("Failed...  ");
		write_text_to_log_file("Hotstart failed switching to cold start");
	}


	
	if (XParam.hotstartfile.empty() || hotstartsucess == 0)
	{
		printf("Cold start  ");
		write_text_to_log_file("Cold start");
		//Cold start
		// 2 options:
		//		(1) if zsinit is set, then apply zsinit everywhere
		//		(2) zsinit is not set so interpolate from boundaries. (if no boundaries were specified set zsinit to zeros and apply case (1))

		//Param defaultParam;
		//!leftWLbnd.empty()

		//case 2b (i.e. zsinint and no boundaries were specified)
		/*
		if ((abs(XParam.zsinit - defaultParam.zsinit) <= epsilon) && (!XParam.leftbnd.on && !XParam.rightbnd.on && !XParam.topbnd.on && !XParam.botbnd.on)) //zsinit is default
		{
			XParam.zsinit = 0.0; // better default value
		}
		*/
		//case(1)
		/*
		if (abs(XParam.zsinit - defaultParam.zsinit) > epsilon) // apply specified zsinit
		{
			int coldstartsucess = 0;
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				coldstartsucess = coldstart(XParam, zb_d, uu_d, vv_d, zs_d, hh_d);
				printf("Cold start  ");
				write_text_to_log_file("Cold start");
			}
			else
			{
				coldstartsucess = coldstart(XParam, zb, uu, vv, zs, hh);
				printf("Cold start  ");
				write_text_to_log_file("Cold start");
			}

		}

		else // lukewarm start i.e. bilinear interpolation of zs at bnds // Argggh!
		{
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				warmstart(XParam, zb_d, uu_d, vv_d, zs_d, hh_d);
				printf("Warm start  ");
				write_text_to_log_file("Warm start");
			}
			else
			{
				warmstart(XParam, zb, uu, vv, zs, hh);
				printf("Warm start  ");
				write_text_to_log_file("Warm start");

			}
		}// end else
		*/
	}
}

