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

	// Initialise Bathy data
	interp2BUQ(XParam, XModel.blocks, XForcing.Bathy, XModel.zb);

	// Set edges
	setedges(XParam, XModel.blocks, XModel.zb);

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


}

template void InitialConditions<float>(Param XParam, Forcing<float> XForcing, Model<float> XModel);
template void InitialConditions<double>(Param XParam, Forcing<float> XForcing, Model<double> XModel);
