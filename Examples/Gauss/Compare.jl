using junkinthetrunk
using NetCDF
using Printf
using Plots

#cd("D:\\Projects\\BasCartGPU\\testbed\\Gaussian")

zsBSLini=dropdims(ncread("./Basilisk/output.nc","eta",start=[1,1,1],count=[-1,-1,1]),dims=3);

x=ncread("./Basilisk/output.nc","xx");
y=ncread("./Basilisk/output.nc","yy");

zsBGFini=dropdims(ncread("Testbed-Gaussian.nc","zs",start=[1,1,1],count=[-1,-1,1]),dims=3);

initdiff=maximum(zsBGFini.-zsBSLini)
heatmap(zsBGFini.-zsBSLini)

zsBSL=dropdims(ncread("./Basilisk/output.nc","eta",start=[1,1,5],count=[-1,-1,1]),dims=3);
zsBGF=dropdims(ncread("Testbed-Gaussian.nc","zs",start=[1,1,4],count=[-1,-1,1]),dims=3);
enddiff=maximum(zsBGF.-zsBSL);

heatmap(zsBGF.-zsBSL)


write2nc(x, y, zsBGF.-zsBSL, "error.nc");





end
