


step=2;
Cypdata = ncread('C:\Users\bosserellec\Documents\Visual Studio 2015\Projects\Bas_Cart_CPU\Bas_Cart_CPU\2Dvar.nc','2Dvar',[1 1 step], [Inf Inf 1], [1 1 1]);

Basdata = ncread('D:\Models\Basilisk\TestBed\Tutorial\output.nc','h',[1 1 step], [Inf Inf 1], [1 1 1]);


diff=Cypdata-Basdata;


pcolor(diff); shading flat;


