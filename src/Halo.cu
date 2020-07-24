#include "Halo.h"

template <class T> void fillHalo()
{
	
}

template <class T> void fillLeft(Param XParam,int ib,int leftib,int levelib,int levelleftib,T*z)
{
	if (ib == leftib)//
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//
			int read, write;
			read = 1 + (j + XParam.halowidth) * XParam.halowidth + ib * XParam.blksize;
			write = 0 + (j + halowidth) * blkmemwidth + ib * blksize;
			z[write] = z[read];
		}
	}

}