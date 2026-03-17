using junkinthetrunk
using NetCDF
using Printf


#cd("D:\\Projects\\BasCartGPU\\testbed\\Gaussian")

zs=ncread("./Basilisk/output.nc","eta",start=[1,1,1],count=[-1,-1,1]);

x=ncread("./Basilisk/output.nc","xx");
y=ncread("./Basilisk/output.nc","yy");

write2nc(x, y, dropdims(zs,dims=3), "zsinit.nc", ["xx","yy","zs"]);

write2nc(x, y, dropdims(zeros(size(zs)),dims=3), "bathy.nc");
