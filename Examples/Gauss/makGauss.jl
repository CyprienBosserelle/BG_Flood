using GMT

xo=-5;
yo=-5;

nx=16*16;
ny=16*16;

len=10.0;

dx=len/(nx-1);

xx=collect(xo:len/(nx-1):(len+xo));
yy=collect(yo:len/(ny-1):(len+yo));


hh=zeros(nx,ny);
zb=zeros(nx,ny);
for i=1:nx
    for j=1:ny
        hh[i,j] = 1.0.+ 1.0.*exp.(-1.0*(xx[i].*xx[i] .+ yy[j].*yy[j]));
        #hh[i,j] =
    end
end



G = mat2grid(transpose(hh), 1,[xx[1] xx[end] yy[1] yy[end] minimum(hh) maximum(hh) 1 dx dx])
cmap = grd2cpt(G);      # Compute a colormap with the grid's data range
grdimage(G, lw=:thinnest, color=cmap, fmt=:png, show=true)

gmtwrite("gauss.asc", G; id="ef");
gmtwrite("gauss.nc", G);
gmt("grdmath gauss.nc 1.0 MUL = gauss_zs.nc?zs");

G = mat2grid(transpose(zb), 1,[xx[1] xx[end] yy[1] yy[end] minimum(zb) maximum(zb) 1 dx dx])
#cmap = grd2cpt(G);      # Compute a colormap with the grid's data range
#grdimage(G, lw=:thinnest, color=cmap, fmt=:png, show=true)

gmtwrite("bathy.asc", G; id="ef");
#gmtwrite("Flat_bottom.nc", G);
