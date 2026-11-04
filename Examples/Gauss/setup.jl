
using NetCDF
using Printf

#write2nc writes a matrix to netcdf file
function write2nc(x,y,z,ncfile,varnames)

    xatts = Dict("longname" => "eastings",
        "units"    => "m")
    yatts = Dict("longname" => "northings",
                "units"    => "m")
    varatts = Dict("longname" => "Topo / Bathy above MVD53",
                "units"    => "m")
    NetCDF.nccreate(ncfile,varnames[3],varnames[1],x,xatts,varnames[2],y,yatts,atts=varatts)
    NetCDF.ncwrite(x,ncfile,varnames[1]);
    NetCDF.ncwrite(y,ncfile,varnames[2]);
    NetCDF.ncwrite(z,ncfile,varnames[3]);
    NetCDF.ncclose(ncfile);
end
function write2nc(x,y,z,ncfile)
    varnames=["x","y","z"];
    write2nc(x,y,z,ncfile,varnames)
end


#cd("D:\\Projects\\BasCartGPU\\testbed\\Gaussian")

zs=ncread("./Basilisk/output.nc","eta",start=[1,1,1],count=[-1,-1,1]);

x=ncread("./Basilisk/output.nc","xx");
y=ncread("./Basilisk/output.nc","yy");

write2nc(x, y, dropdims(zs,dims=3), "zsinit.nc", ["xx","yy","zs"]);

write2nc(x, y, dropdims(zeros(size(zs)),dims=3), "bathy.nc");
