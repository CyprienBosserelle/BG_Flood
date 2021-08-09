### A Pluto.jl notebook ###
# v0.12.12

using Markdown
using InteractiveUtils

# ╔═╡ 9f6a4682-dd4b-11eb-29b9-0311850f7de6
using GMT, DelimitedFiles, Printf, StatsBase

# ╔═╡ 949761a0-dd16-11eb-32de-4b235b067893
md"""
# Monai Valley Example
This example replicates the "Tsunami runup onto a complex three-dimensional beach; Monai Valley" benchmerk from NOAA.

This notebook is presented to show how I use the model and also how the model performs against standart benchmarks.

_Cyprien Bosserelle 07/07/2021_

## Some background
This lab benchmark is an experimental setup to replicate the extreme runup observed during the Hokkaido-Nansei-Oki tsunami of 1993 that struck Okushiri Island, Japan. The extreme runup was located on a small cove (Monai Cove) surrounded by a complex bathymetry.

Details of the experiements and the benchmark data are available at the  [NOAA webpage](https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/index.html) and are referenced in [Synolakis et al. 2007](https://nctr.pmel.noaa.gov/benchmark/SP_3053.pdf) 

## To run this notebook
This notebook requires **GMT 6.2** installed as well as the GMT.jl package. Downloads is part of base in Julia version 1.6 but may need to be added. also please add Printf for writing neat looking txt files and StatsBase.

While I tried to make this notebook as portatble as I could I couldn't resolve the issue that the gauge data is provided online in a _defunct_ excel format. I have left a copy of in txt in the repository.

"""


# ╔═╡ 3c725fe0-dea9-11eb-25e3-03d2f01c07f8
md"""
## Where is BG\_flood
list below the folder where BG_flood (and associated dlls) resides:
"""

# ╔═╡ 25d689d0-dea1-11eb-3446-cfa9d339a899
bgfloodbin=`C:\\Users\\bosserellec\\source\\repos\\BG_Flood_Cleanup\\x64\\Release\\BG_Flood_Cleanup.exe`

# ╔═╡ a81a6580-dd4b-11eb-2c2c-1f7e0e684d59
md"""
##  Experimental data
We download the bathymetry, input wave and gauge reading from the internet and use GMT to make a NetCDF grid.
"""

# ╔═╡ 92e31b39-e80c-4dbb-8daa-b624abcf18bb
md"""
### Bathymetry
"""

# ╔═╡ dda506d2-f026-44c5-aed1-47f7b174d533
download("https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/MonaiValley_Bathymetry.txt","bathy.xyz")

# ╔═╡ 1a056de8-5984-4e93-95f5-ef384b03a367
Bathy=xyz2grd("bathy.xyz",region=(0.0,5.488,0.0,3.402), inc=0.014)

# ╔═╡ 4ac6b170-adf8-46cb-8ece-6d7783229eb9
md"""
This bathy is _positive down_. This is not an issue for BG_Flood but we need to keep that in mind. Before ploting we create a custom colormap. In the plot we will also include the gauges locations.
"""

# ╔═╡ c5f6c40a-b497-476c-8fb0-8a453b919e91
gaugelocs=[4.521 1.196; 4.521 1.696; 4.521 2.196]

# ╔═╡ 79fc30ee-1de4-455b-ab4e-f99fe8899f25
cpt=makecpt(cmap=:geo, inverse=true, range=(-0.2,0.13));

# ╔═╡ f99d1880-dd4b-11eb-3def-57d4ec93c373
begin
	grdimage(Bathy, cmap=cpt);
	scatter!(gaugelocs,G=:black)
	text!(text_record(gaugelocs, ["Gauge 1", "Gauge 2", "Gauge 3"]),justify=:MR, offset=(shift=(-0.2,0.0)), show=true)
	
end

# ╔═╡ c1ef1d6f-367a-499a-81cf-5933af1202cf
md"""
#### Modifying the bathy
We need to modify slighly the bathymetry so that when BG_Flood creates the mesh, it stricly preserves the shape of the lab tank
##### Why do we need to do that?
BG\_Flood creates a mesh by assembling blocks of 16x16 cells together. When the input DEM/Bathy extent or a manualy specified extend does quite fit in the layout of blocks, BG\_Flood extends the last row/column to fill the space. In real world application tha's not a big deal because you can ask BG\_Flood to make a computational domain slightly smaller then your DEM but here in a small tank that will mean a bit more water. Although probably not a huge deal for teh experiment we need to get it right so we put _Physical_ walls on the top and bottom side.
"""

# ╔═╡ 5590a60d-54f2-4c7d-8bc6-e053c48420b7
BathyMod=xyz2grd("bathy.xyz",region=(0.0,5.488,-0.014,3.418), inc=0.014);

# ╔═╡ 6459595b-fa47-4e69-8c72-d38d612abafe
BathyModB=grdmath("? -0.125 DENAN =",BathyMod);

# ╔═╡ df3b7f1a-54e3-42a4-8605-f48d1cef7de2
md"""
Now save the grid to a netcdf file. BG\_Flood likes GMT netcdf files.
"""

# ╔═╡ 456f0e8e-0873-4343-bb38-4f449f64abf4
bathyfile="Monai_BGflood_Bathy.nc";

# ╔═╡ ed7606e2-f0ff-4716-ae36-eef367578a2c
gmtwrite(bathyfile,BathyModB)

# ╔═╡ 53645d59-53c1-441f-b31c-15e841c77413
md"""
### Input wave
"""

# ╔═╡ acb05c81-db01-42b2-ae3f-48bdae935a36
download("https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/MonaiValley_InputWave.txt","InputWave.txt")

# ╔═╡ 94e8ff56-ac70-4b9c-9c49-bbec52f2fb33
md"""
The file has a non-standart hearder and non consistent delimiter so we better read it and rewrite it nicely so not to confuse BG_flood. 
"""

# ╔═╡ 9476801d-8f0a-4f5c-afea-6d195da60e43

inputdata=readdlm("InputWave.txt", '\t',  skipstart=3)


# ╔═╡ 75e1fa4e-a598-4aa4-a736-c3caab94a2b8
plot(inputdata,show=true)

# ╔═╡ fb59a99c-4776-4f4c-83ff-36cb656a3469
md"""
Actually this file is not too bad and you could manually redo the first 2 lines and add a # to the front of the header line. However, the file only provides 22.5s of input so if we want to run the model for longer we need to pad some zeros at the end.
BG\_flood is pretty good with this because we don't need the time axis to be monotonically increasing (i.e. constant time step) and so we just need to add just one or 2 lines of zero padding. Below we rewite the file, add a 0.o at the 25.0s and another at 100.0 s. This will alow us to run the model for 100.0 s but we only really care about the first wave ( 25--30 s). 

"""

# ╔═╡ b868d013-eb8e-4dac-9e66-eea5bbd47ad3
inputfile="Input_Wave_clean.txt";

# ╔═╡ 42f760b3-93c5-4aa7-8855-991318dd0b2e
open(inputfile,"w") do io
	Printf.@printf(io,"%f,%f\n",0.0,0.0);
     for i=1:first(size(inputdata))
         Printf.@printf(io,"%f,%f\n",inputdata[i,1],inputdata[i,2]);

     end
	# Add the zero padding
	Printf.@printf(io,"%f,%f\n",25.0,0.0);
	Printf.@printf(io,"%f,%f\n",100.0,0.0);

end


# ╔═╡ 38788e11-be78-42bf-9b10-50ee8a0b4d8c
md"""

### Gauge data
The gauge data is stored in a 95-97 xls workbook. These are not considered _safe_ files so we will instead use a txt file we provide in the repository. 

"""

# ╔═╡ e5bab32b-31dc-4513-b62f-252d1072f5f5
GaugeData=readdlm("./data/MonaiValley_WaveGages.txt", '\t',  skipstart=1)

# ╔═╡ a05b1574-d37a-4635-9984-bf070e2f2b62
download("https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/MonaiValley_WaveGages.xls","WaveGages.xls")

# ╔═╡ df570f11-20bf-4066-820a-267303668c4c
plot(GaugeData[:,1],GaugeData[:,2:4],show=true)

# ╔═╡ 4e26c913-ded0-44d7-9d49-3c96d3eeab70
md"""
## Prepare the model
### Write the BG_param.txt file
BG\_param.txt is the file that controles what BG\_flood does. for this simulation we will keep most of the parameters to default, but we still need to tell the model about what and where we want outputs.

#### First create a new file and add a description for what it is meant to be.
Note the # tag in front of the description. this tells BG_Flood to ignore the line. Thus, this is totally optional.
"""

# ╔═╡ 7941e544-fa23-42a2-82c9-73d9e80d8e8e
open("BG_param.txt","w") do io
	Printf.@printf(io,"# Monai Valley demo\n\n");
end

# ╔═╡ fede8135-d098-4778-859d-c16e589fe5a3
md"""
#### Specify whether you have a GPU or not.
on my Laptop I do not have a suitable GPU so I run the CPU mode:

`gpudevice=-1`
"""

# ╔═╡ 1ae54e3f-1c0b-4681-962b-f0d5b7930e4b
open("BG_param.txt","a") do io
	Printf.@printf(io,"gpudevice = %d;\n\n",1);
	
end

# ╔═╡ 75b33ef4-e19c-485d-9d9b-1a5de201be86
md"""
#### Add the bathymetry grid
BG\_flood likes GMT netcdf format but we still need to tell BG\_flood which variable in the netcdf file to read. This is done by appending a `?` followed by the variable name e.g:

`myfile.nc?z`

Remember that the bathymetry is positive down. That is not the default in BG\_Flood so we need to tell BG_Flood how to deal with it:

`posdown = 1;`
"""

# ╔═╡ 8f5063ca-8ffb-479c-adef-49107c73f7ec
open("BG_param.txt","a") do io
	Printf.@printf(io,"bathy = %s;\n",bathyfile*"?z");
	Printf.@printf(io,"posdown = %d;\n\n",1);
end

# ╔═╡ e77de767-4686-4522-a65c-b754aa7fa570
md"""
#### Add the boundary forcing
In this experiment the forcing wave comes from the left hand side of the grid.so the keyword is `left`. When forcing a boundary you will need the type of boundary and if necessary the forcing file. In this case we will be forcing with a Dirichlet type boundary(type 2) so we need to write":

`left = myfile.txt,2;`

Also the right boundary is a wall?

`right = 0;`
"""

# ╔═╡ fb6e55b9-b6af-4fa4-a0f3-cb20dc196253
open("BG_param.txt","a") do io
	Printf.@printf(io,"left = %s,%d;\n\n",inputfile,2);
	Printf.@printf(io,"right = %d;\n\n",0);
end

# ╔═╡ 4b39e332-59c2-435f-a0b8-6b2b2118efe0
md"""
#### Add the model end time.
by default, BG\_flood will run for as long as there is forcing in the bounary files. With our custom made boundary we can run the model for up to 100s but we really care about the first bit so we will run only for 50s.

`endtime=50.0;`
"""

# ╔═╡ 76fb8b86-996b-425d-af41-2a8eef6eef53
open("BG_param.txt","a") do io
	Printf.@printf(io,"endtime = %f;\n\n",50.0);
end

# ╔═╡ 0a1dbdc6-d4dc-4416-a004-59806fb62535
md"""
#### Add timeseries output
There are 3 gauges locations where we can compare the model output and the experiement. we define these with the keyword `TSOutput`. then we need to specify the coordinate of the output and an output file.

`TSOutput = Thisfile.txt,0.5,0.5;`

because there are 3 locations we need to repeat the keyword for each location.but because we are in a programing environment 3 or 3 million output time series is the same effort.

"""

# ╔═╡ b8766fe8-c9fe-4e7c-aa40-d8a34f3afc8d
open("BG_param.txt","a") do io
	
	n=first(size(gaugelocs));
	for i=1:n
		Printf.@printf(io,"TSOutput = %s,%f,%f;\n","Monai_BG_Gauge_"*string(i)*".txt",gaugelocs[i,1],gaugelocs[i,2])
	end
	
	
end

# ╔═╡ bb119cde-669b-4cc2-b4d5-f1ae048c654e
md"""
#### Add some map output
We can use that to compare the runup and make pretty figures.

`outfile=myfile.nc`

We will also specify which variables we want to output:

* _zs_: water level (i.e. snapshot at each step)
* _zsmax_: maximumum water level (i.e. maximum for each step so far )
* _hmax_: Maximum water depth

`outvars=zs,zsmax,hmax`

Finally we need to specify how often we want output maps (in seconds). We will run the model for 30 s so 1.0 s output will give us 50 slides which should be nice to watch

`outputtimestep = 1.0`

By default, BG\_flood save the output as ascaled up short integer that most other NetCDF software are capable of rescaling on the fly (COARDS compliance). This make much smaller output files and while this is pratical for real life models it doesn't work well for lab experiment where scales are quite small. so let's switch this option off.

`smallnc=0`

"""

# ╔═╡ afbb832f-cc6f-4b24-af38-952ca22afd9e
outfile="Output_Monai_BG.nc"

# ╔═╡ 0a83e5bc-f3d8-4e8b-a51f-24d19d9d9b54
open("BG_param.txt","a") do io
	Printf.@printf(io,"outfile = %s;\n",outfile);
	Printf.@printf(io,"outvars = zs,zsmax,hmax;\n");
	Printf.@printf(io,"outputtimestep = %f;\n",1.0);
	Printf.@printf(io,"smallnc = %d;\n",0.0);
end

# ╔═╡ 576c4515-2066-420f-9bc9-2fe93ecfec02
md"""
## Run the model
To run the model you will need the executable (and dlls) to be present in the folder. 
"""

# ╔═╡ f2e52db8-a76c-49b2-b643-53017824eaf1
begin
	rm(outfile,force=true)
	run(bgfloodbin)
end 

# ╔═╡ 0e653452-42b0-4dae-bc5c-ad46ba8e5372
md"""
## Results
Let's check the results.

### Runup map

"""

# ╔═╡ b6a96982-888a-4ee2-a7f3-1e8e2939e4ab
step=50;

# ╔═╡ 65ea5121-5cb9-4d7e-9dc4-ac0d8d231123
zsmaxvar = "zsmax_P0";

# ╔═╡ 1aaef1db-59f3-4991-b658-0657f9801d66
hmaxvar = "hmax_P0";

# ╔═╡ 3da84b7f-8865-4f0e-9ecb-ed8227874cc2
zsmaxstgmt=outfile*"?"*zsmaxvar*"["*string(step)*"]";

# ╔═╡ 6795175b-0646-4904-978e-24fdadec3f68
hmaxstgmt=outfile*"?"*hmaxvar*"["*string(step)*"]";

# ╔═╡ 3e58715e-a360-452b-908b-1a4abf1b332b
zsmax=gmtread(zsmaxstgmt);

# ╔═╡ 53625fa5-92d4-481c-8f9d-ea72f79ee611
hmax=gmtread(hmaxstgmt);

# ╔═╡ 7d2b30c0-afe4-4f63-a572-ece53e40ceb7
Runup=grdmath("? 0.001 GT 0 NAN ? MUL =",hmax,zsmax);

# ╔═╡ 1a784c29-e2bf-4a23-a243-2f668b35ce51
cwave=makecpt(cmap=:panoply, range=(0.0,0.05));

# ╔═╡ ed88675d-5362-42d8-b281-1c2c88ea26cc
begin
	grdimage(Bathy, cmap=cpt, region=(2.5,5.4,0.8,3.0));
	grdimage!(Runup, cmap=cwave, nan_alpha=true)
	scatter!(gaugelocs,G=:black, show=true)
end

# ╔═╡ 2f11c250-de13-11eb-33cf-3580fab90aa2
md"""
### Gauges output
Read the simulated gauge output.
"""

# ╔═╡ c0f8d550-de13-11eb-0fe8-69be1dcd7792
begin
	BGG1=readdlm("Monai_BG_Gauge_"*string(1)*".txt", '\t',  skipstart=1);
	BGG2=readdlm("Monai_BG_Gauge_"*string(2)*".txt", '\t',  skipstart=1);
	BGG3=readdlm("Monai_BG_Gauge_"*string(3)*".txt", '\t',  skipstart=1);
end

# ╔═╡ fd1c0930-de13-11eb-2eaf-4f04b2bf6604
begin
	plot(GaugeData[:,1],GaugeData[:,2],pen=(1.0,:black), region=(0,25,-1,5), xlabel="time [s]", ylabel="water level [cm]",title="Gauge 1")
	plot!(BGG1[:,1],BGG1[:,2].*100.0,pen=(1.0,:red),show=true)
	
end

# ╔═╡ 045167b0-de17-11eb-000a-fd7463f0598c
begin
	plot(GaugeData[:,1],GaugeData[:,3],pen=(1.0,:black), region=(0,25,-1,5), xlabel="time [s]", ylabel="water level [cm]",title="Gauge 2")
	plot!(BGG2[:,1],BGG2[:,2].*100.0,pen=(1.0,:red),show=true)
	
end

# ╔═╡ 07d9a410-de17-11eb-2aed-d7ced49e5460
begin
	plot(GaugeData[:,1],GaugeData[:,4],pen=(1.0,:black), region=(0,25,-1,5), xlabel="time [s]", ylabel="water level [cm]",title="Gauge 3")
	plot!(BGG3[:,1],BGG3[:,2].*100.0,pen=(1.0,:red),show=true)
	
end

# ╔═╡ ab1227fe-de17-11eb-11b6-0d9773718991
md"""
#### Measuring the skill of the model
Plots are great to have a visual comparison and really show how well the model works it is also nice to have a measure too.

##### Define some useful functions and a skill function
`interp1()` is a 1d linear interpolation function to interpolate the time axis so we can have the data and simulation at exactly the same time for comparison.

`skills()` is a function that calculate a handfull of model validation stats: RMS, bias, Index of agreement (Willmott 2002) and skill score.
"""


# ╔═╡ c5aeca00-de18-11eb-0a58-39fc9c5d7c10
function interp1(xA, yA, newx)
        # Interpolate to time nx even if xA is not monotonic (although it has to be increasing)
        # This is a naive implementation and could do with improvements.
        # This function deals with extrapolation by padding the first/last known value
        #

        #selcasttimedep[indnonan], error[indnonan], ttt

        #xA=selcasttimedep[indnonan]
        #yA=error[indnonan]
        #newx=copy(ttt)

		xA=vec(xA);
	    yA=vec(yA);


        y = zeros(length(newx))
        for n = 1:length(newx)
            index = findfirst(xA .> newx[n]);
            if index==nothing
                if newx[n] <= xA[1]
                    index = 1;
                elseif newx[n] >= xA[end]
                    index = length(xA)+1;
                end
            end
            prev = yA[max(index[1] - 1, 1)];
            next = yA[min(index[1],length(xA))];

            time = newx[n] - xA[max(index[1] - 1, 1)];
            timenext = xA[max(min(index[1],length(xA)), 1)] - xA[max(index[1] - 1, 1)];

            if max(min(index[1],length(xA)), 1) == max(index[1] - 1, 1)

                y[n] = yA[max(min(index[1],length(xA)), 1)];
            else
                y[n] = prev + (time) / (timenext) * (next - prev);

            end
        end
        return y;
    end

# ╔═╡ c668e7b0-de17-11eb-2a96-2b4eedc665a3
function skills(tmeas,xmeas,tmodel,xmodel)
        # Calculate RMS. Bias, Index of agreement, and skill score
        #

        #crop non overlapping
        index=(tmodel.>=tmeas[1]) .& (tmodel.<=tmeas[end]);

        tmod=tmodel[index];
        xmod=xmodel[index];

        indxmea=(tmeas.>=tmod[1]) .& (tmeas.<=tmod[end]);
        tmea=tmeas[indxmea];
        xmea=xmeas[indxmea];


        xint=interp1(tmod,xmod,tmea);

		PmO=sum(abs.(xint.-xmea));
		OmOb=sum(abs.(xmea.-StatsBase.mean(xmea)))
		Cwcorr=2.0;
		Wcorr=0.0;

		if(PmO<=(Cwcorr*OmOb))
			Wcorr=1.0-(PmO/(Cwcorr*OmOb));
		else
			Wcorr=((Cwcorr*OmOb)/PmO)-1.0;
		end

        RMS=sqrt(StatsBase.mean((xint.-xmea).^2));
        Bias=StatsBase.mean(xint)-StatsBase.mean(xmea);
        #Wcorr=1-(sum((xmea-xint).^2)/(sum((abs.(xmea.-StatsBase.mean(xint)).+abs.(xint.-StatsBase.mean(xint))).^2)));

		Bss=1-(StatsBase.var((xmea-xint))/(StatsBase.var((xint.-StatsBase.mean(xint)))));

        return RMS,Bias,Wcorr,Bss
    end

# ╔═╡ f2c51590-dea3-11eb-17e3-21fd60e6f304
md"""
##### Skills
Now we calculate the skills for each gauge separately. We are only calculating the skill of the model for the first 25 s of the model.
"""

# ╔═╡ 452c9370-de19-11eb-1deb-157e713d4964
indxt=GaugeData[:,1] .<25.0;

# ╔═╡ 0511e6a0-de19-11eb-08d9-fdc5480f4254
skillG1=skills(GaugeData[indxt,1],GaugeData[indxt,2],BGG1[:,1],BGG1[:,2].*100.0)

# ╔═╡ ad111a5e-de19-11eb-0925-03015ce5836a
skillG2=skills(GaugeData[indxt,1],GaugeData[indxt,3],BGG2[:,1],BGG2[:,2].*100.0)

# ╔═╡ af034d70-de19-11eb-3800-7ff4270631bc
skillG3=skills(GaugeData[indxt,1],GaugeData[indxt,4],BGG3[:,1],BGG3[:,2].*100.0)

# ╔═╡ f9d87f12-dea4-11eb-0095-e598384dbc95
md"""
Overall the model is doing pretty good with index of agreement above $(minimum([skillG1[3] skillG2[3] skillG3[3]])).

## What's next ?

there are a bunch of things that we could try to further improve the model.

### Single vs double precision
By default, BG\_flood runs with single precision math which makes things much faster. But, is the results much better in double precision?

### Default friction
By default BG_flood runs with a sinple low quadratic friction. For most applications this is not good enough but you can try a Manning friction or a Smart friction. any difference?

### Adaptive mesh
Can we run in adaptive mode to better resolve the wave at/near the gauges? See `monai_adapt.jl` notebook. 
"""

# ╔═╡ Cell order:
# ╟─949761a0-dd16-11eb-32de-4b235b067893
# ╠═9f6a4682-dd4b-11eb-29b9-0311850f7de6
# ╟─3c725fe0-dea9-11eb-25e3-03d2f01c07f8
# ╠═25d689d0-dea1-11eb-3446-cfa9d339a899
# ╟─a81a6580-dd4b-11eb-2c2c-1f7e0e684d59
# ╟─92e31b39-e80c-4dbb-8daa-b624abcf18bb
# ╠═dda506d2-f026-44c5-aed1-47f7b174d533
# ╠═1a056de8-5984-4e93-95f5-ef384b03a367
# ╟─4ac6b170-adf8-46cb-8ece-6d7783229eb9
# ╠═c5f6c40a-b497-476c-8fb0-8a453b919e91
# ╠═79fc30ee-1de4-455b-ab4e-f99fe8899f25
# ╠═f99d1880-dd4b-11eb-3def-57d4ec93c373
# ╟─c1ef1d6f-367a-499a-81cf-5933af1202cf
# ╠═5590a60d-54f2-4c7d-8bc6-e053c48420b7
# ╠═6459595b-fa47-4e69-8c72-d38d612abafe
# ╟─df3b7f1a-54e3-42a4-8605-f48d1cef7de2
# ╠═456f0e8e-0873-4343-bb38-4f449f64abf4
# ╠═ed7606e2-f0ff-4716-ae36-eef367578a2c
# ╟─53645d59-53c1-441f-b31c-15e841c77413
# ╠═acb05c81-db01-42b2-ae3f-48bdae935a36
# ╟─94e8ff56-ac70-4b9c-9c49-bbec52f2fb33
# ╠═9476801d-8f0a-4f5c-afea-6d195da60e43
# ╠═75e1fa4e-a598-4aa4-a736-c3caab94a2b8
# ╟─fb59a99c-4776-4f4c-83ff-36cb656a3469
# ╠═b868d013-eb8e-4dac-9e66-eea5bbd47ad3
# ╠═42f760b3-93c5-4aa7-8855-991318dd0b2e
# ╟─38788e11-be78-42bf-9b10-50ee8a0b4d8c
# ╠═e5bab32b-31dc-4513-b62f-252d1072f5f5
# ╠═a05b1574-d37a-4635-9984-bf070e2f2b62
# ╠═df570f11-20bf-4066-820a-267303668c4c
# ╟─4e26c913-ded0-44d7-9d49-3c96d3eeab70
# ╠═7941e544-fa23-42a2-82c9-73d9e80d8e8e
# ╟─fede8135-d098-4778-859d-c16e589fe5a3
# ╠═1ae54e3f-1c0b-4681-962b-f0d5b7930e4b
# ╟─75b33ef4-e19c-485d-9d9b-1a5de201be86
# ╠═8f5063ca-8ffb-479c-adef-49107c73f7ec
# ╟─e77de767-4686-4522-a65c-b754aa7fa570
# ╠═fb6e55b9-b6af-4fa4-a0f3-cb20dc196253
# ╟─4b39e332-59c2-435f-a0b8-6b2b2118efe0
# ╠═76fb8b86-996b-425d-af41-2a8eef6eef53
# ╟─0a1dbdc6-d4dc-4416-a004-59806fb62535
# ╠═b8766fe8-c9fe-4e7c-aa40-d8a34f3afc8d
# ╟─bb119cde-669b-4cc2-b4d5-f1ae048c654e
# ╠═afbb832f-cc6f-4b24-af38-952ca22afd9e
# ╠═0a83e5bc-f3d8-4e8b-a51f-24d19d9d9b54
# ╟─576c4515-2066-420f-9bc9-2fe93ecfec02
# ╠═f2e52db8-a76c-49b2-b643-53017824eaf1
# ╟─0e653452-42b0-4dae-bc5c-ad46ba8e5372
# ╠═b6a96982-888a-4ee2-a7f3-1e8e2939e4ab
# ╠═65ea5121-5cb9-4d7e-9dc4-ac0d8d231123
# ╠═1aaef1db-59f3-4991-b658-0657f9801d66
# ╠═3da84b7f-8865-4f0e-9ecb-ed8227874cc2
# ╠═6795175b-0646-4904-978e-24fdadec3f68
# ╠═3e58715e-a360-452b-908b-1a4abf1b332b
# ╠═53625fa5-92d4-481c-8f9d-ea72f79ee611
# ╠═7d2b30c0-afe4-4f63-a572-ece53e40ceb7
# ╠═1a784c29-e2bf-4a23-a243-2f668b35ce51
# ╠═ed88675d-5362-42d8-b281-1c2c88ea26cc
# ╟─2f11c250-de13-11eb-33cf-3580fab90aa2
# ╠═c0f8d550-de13-11eb-0fe8-69be1dcd7792
# ╠═fd1c0930-de13-11eb-2eaf-4f04b2bf6604
# ╠═045167b0-de17-11eb-000a-fd7463f0598c
# ╠═07d9a410-de17-11eb-2aed-d7ced49e5460
# ╟─ab1227fe-de17-11eb-11b6-0d9773718991
# ╟─c5aeca00-de18-11eb-0a58-39fc9c5d7c10
# ╟─c668e7b0-de17-11eb-2a96-2b4eedc665a3
# ╟─f2c51590-dea3-11eb-17e3-21fd60e6f304
# ╠═452c9370-de19-11eb-1deb-157e713d4964
# ╠═0511e6a0-de19-11eb-08d9-fdc5480f4254
# ╠═ad111a5e-de19-11eb-0925-03015ce5836a
# ╠═af034d70-de19-11eb-3800-7ff4270631bc
# ╟─f9d87f12-dea4-11eb-0095-e598384dbc95
