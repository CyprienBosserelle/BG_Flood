
nx=512;
ny=256;

dx=0.5;

xx=0:(nx-1);
yy=0:(ny-1);
xx=xx.*dx;
yy=yy.*dx;
zs=zeros(nx,ny)-5;

zs(150:155,:)=5;

zs(:,1:2)=5;

zs(:,end-1:end)=5;
zs(150:155,126:130)=-5;

pcolor(xx,yy,zs');shading flat


write2nc(xx,yy,zs,'Slit_bathy.nc')

