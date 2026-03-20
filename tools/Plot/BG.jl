"""
    BG module
    Collection of function to create input and process output of the BG model

    Available functions:
    plotvar

    #Example:
    using .BG

"""
module BG

    import Printf, NetCDF, GMT
    export plotvar


    function scanminmax(file,var,stp)


        zmin=1/eps(eps(1.0))
        zmax=-1.0.*zmin
        levmax=NetCDF.ncgetatt(file, "Global", "maxlevel");
        levmin=NetCDF.ncgetatt(file, "Global", "minlevel");
        for level=levmin:levmax

            if level<0
                stlv="N";
            else
                stlv="P";
            end
            fvar=var*"_"*stlv*string(abs(level))

            scalefac=NetCDF.ncgetatt(file, fvar, "scale_factor");
            addoffset=NetCDF.ncgetatt(file, fvar, "add_offset");
            missval=NetCDF.ncgetatt(file, fvar, "_FillValue");



            zz=NetCDF.ncread(file,fvar,[1,1,max(stp,1)], [-1,-1,1]);
            if any([!isnothing(scalefac) !isnothing(addoffset)])
                zz=zz.*scalefac.+addoffset;
            end
            zz[zz.==missval].=NaN;

            zz=zz[.!isnan.(zz)];


            if !isempty(zz)
                zmin=min(zmin,minimum(zz[.!isnan.(zz)]));
                zmax=max(zmax,maximum(zz[.!isnan.(zz)]));
            end



        end
        if zmin==zmax
            zz=zmin
            zmin=zz-1.0;
            zmax=zz+1.0;
        end


        return zmin,zmax
    end

    function readBGvarlevel(file,fvar,stp)


        scalefac=NetCDF.ncgetatt(file, fvar, "scale_factor");
        addoffset=NetCDF.ncgetatt(file, fvar, "add_offset");
        missval=NetCDF.ncgetatt(file, fvar, "_FillValue");



        zz=dropdims(NetCDF.ncread(file,fvar,[1,1,max(stp,1)], [-1,-1,1]),dims=3);

        if any([!isnothing(scalefac) !isnothing(addoffset)])
            zz=zz.*scalefac.+addoffset;
        end
        zz[zz.==missval].=NaN;


        return zz
    end

    function readBGxy(file::String,level::Integer)

        if level<0
            stlv="N";
        else
            stlv="P";
        end

        xvar="xx_"*stlv*string(abs(level))
        yvar="yy_"*stlv*string(abs(level))
        xx=NetCDF.ncread(file,xvar,[1], [-1]);
        yy=NetCDF.ncread(file,yvar,[1], [-1]);
        return xx,yy
    end


    function readBGvar(file::String,var::String,step::Int)
        levmax=NetCDF.ncgetatt(file, "Global", "maxlevel");
        levmin=NetCDF.ncgetatt(file, "Global", "minlevel");

        zBG=Vector{Array{AbstractFloat,2}}(undef,(levmax-levmin+1))
        xBG=Vector{Array{AbstractFloat,1}}(undef,(levmax-levmin+1))
        yBG=Vector{Array{AbstractFloat,1}}(undef,(levmax-levmin+1))

        levels=levmin:levmax

        for level=levmin:levmax

            if level<0
                stlv="N";
            else
                stlv="P";
            end

            varstr=var*"_"*stlv*string(abs(level))

            indxlev=level-levmin+1

            zBG[indxlev]=readBGvarlevel(file,varstr,step);
            xBG[indxlev],yBG[indxlev]=readBGxy(file,level)
        end
        return levels,xBG,yBG,zBG
    end




    function plotblocks(file,var,step; showid=false)

        # Draw the blocks
        blkxo = NetCDF.ncread(file, "blockxo")
        blkyo = NetCDF.ncread(file, "blockyo")
        blkwidth= NetCDF.ncread(file, "blockwidth") .*16.0;
        dx=NetCDF.ncread(file, "blockwidth");


        nblk=length(blkxo);

        for ib=1:nblk
            rect = [blkxo[ib].-dx[ib]*0.5 blkyo[ib].-dx[ib]*0.5; blkxo[ib]+blkwidth[ib].-dx[ib]*0.5 blkyo[ib].-dx[ib]*0.5; blkxo[ib]+blkwidth[ib].-dx[ib]*0.5 blkyo[ib]+blkwidth[ib].-dx[ib]*0.5; blkxo[ib].-dx[ib]*0.5 blkyo[ib]+blkwidth[ib].-dx[ib]*0.5; blkxo[ib].-dx[ib]*0.5 blkyo[ib].-dx[ib]*0.5];
            GMT.plot!(rect, lw=1,t=80);
            if showid
                GMT.text!(GMT.text_record([blkxo[ib] blkyo[ib]], [string(ib)]), attrib=(font=(8,"Helvetica",:black),angle=0,justify=:LM));
            end
        end
    end

    function makeblocksGMT(file,var,step,outfile)
        #
        blkxo = NetCDF.ncread(file, "blockxo")
        blkyo = NetCDF.ncread(file, "blockyo")
        blkwidth= NetCDF.ncread(file, "blockwidth") .*16.0;
        dx=NetCDF.ncread(file, "blockwidth");


        nblk=length(blkxo);


        open(outfile,"w") do io

            Printf.@printf(io,">>\n");


            for ib=1:nblk
                rect = [blkxo[ib].-dx[ib]*0.5 blkyo[ib].-dx[ib]*0.5; blkxo[ib]+blkwidth[ib].-dx[ib]*0.5 blkyo[ib].-dx[ib]*0.5; blkxo[ib]+blkwidth[ib].-dx[ib]*0.5 blkyo[ib]+blkwidth[ib].-dx[ib]*0.5; blkxo[ib].-dx[ib]*0.5 blkyo[ib]+blkwidth[ib].-dx[ib]*0.5; blkxo[ib].-dx[ib]*0.5 blkyo[ib].-dx[ib]*0.5];
                
                Printf.@printf(io,"%f\t%f\n%f\t%f\n%f\t%f\n%f\t%f\n%f\t%f\n>>\n",blkxo[ib].-dx[ib]*0.5, blkyo[ib].-dx[ib]*0.5,blkxo[ib]+blkwidth[ib].-dx[ib]*0.5, blkyo[ib].-dx[ib]*0.5,blkxo[ib]+blkwidth[ib].-dx[ib]*0.5, blkyo[ib]+blkwidth[ib].-dx[ib]*0.5, blkxo[ib].-dx[ib]*0.5, blkyo[ib]+blkwidth[ib].-dx[ib]*0.5,blkxo[ib].-dx[ib]*0.5,blkyo[ib].-dx[ib]*0.5)
                
            end
        end


    end


    function plotvar(file,var,step,figname;region=(NaN,NaN,NaN,NaN),zrange=(NaN,NaN), plotblock=true, plotid=false, cpt=:jet, colorunit=:m)
        #

        levmax=NetCDF.ncgetatt(file, "Global", "maxlevel");
        levmin=NetCDF.ncgetatt(file, "Global", "minlevel");

        xmin=isnan(region[1]) ? NetCDF.ncgetatt(file, "Global", "xmin") : region[1];
        xmax=isnan(region[2]) ? NetCDF.ncgetatt(file, "Global", "xmax") : region[2];
        ymin=isnan(region[3]) ? NetCDF.ncgetatt(file, "Global", "ymin") : region[3];
        ymax=isnan(region[4]) ? NetCDF.ncgetatt(file, "Global", "ymax") : region[4];


        global zmin=9999.0
        global zmax=-9999.0
        dz=100.0
        #zrange[2]

        if any([isnan(zrange[1]) isnan(zrange[2])])
            zmin,zmax=scanminmax(file,var,step)
        else
            zmin=zrange[1];
            zmax=zrange[2];

            if (length(zrange)>2)
                dz=zrange[3];
            else
                dz=(zmax-zmin)/100.0
            end

        end

        if zmax<=zmin
            zmax=zmin+0.1
            dz=0.1
        end

       

        #regplot=(xmin,xmax,ymin,ymax)

        topo = GMT.makecpt(color=cpt, range=(zmin,zmax,dz), continuous=true)
        GMT.basemap(proj=:linear, region=(xmin,xmax,ymin,ymax), frame=(annot=:auto, ticks=:auto))
        #basemap(proj=:linear, figsize=(24.5,1.0), region=(xmin,xmax,ymin,ymax))
        for level=levmin:levmax

            if level<0
                stlv="N";
            else
                stlv="P";
            end

            GMT.grdimage!(file*"?"*var*"_"*stlv*string(abs(level))*"["*string(step)*"]", frame=:a, color=topo, Q=true)
        end
        if plotblock
            plotblocks(file,var,step,showid=plotid);
        end


        GMT.colorbar!(pos=(anchor=:TC,length=(15.5,0.6), horizontal=true, offset=(0,1.0)),
                  color=topo, frame=(annot=:auto,ylabel=colorunit,),nolines=true,savefig=figname*".png" , fmt=:png, show=true)





    end


 

end
