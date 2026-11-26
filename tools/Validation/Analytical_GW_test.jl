### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 546f34d0-5125-11f1-1dce-c79ac1688d87
using Pkg

# ╔═╡ 64553d93-ad06-42bd-ab2c-b1aaa7231189
Pkg.activate()

# ╔═╡ 78ddbf7b-31a7-4232-9173-eb6fdf799189
using GMT

# ╔═╡ 9804ea50-575d-471e-a3a9-86eeb08a8bec
using SpecialFunctions  # For the erf() function


# ╔═╡ 952f9644-540e-4e66-b6e7-dcae8abaf18b

using QuadGK            # For numerical integration of the S* function

# ╔═╡ 1558fa3e-d026-4475-881c-ac9899fa9878
using NetCDF

# ╔═╡ 4e1d57f1-cb04-4a63-ab20-b205f9055e98
"""
    boussinesq_mound_1d(x, t, h0, Hm, L, K, Sy, h_bar; max_n=151)

Calculates the analytical linearized Boussinesq solution for a decaying groundwater mound.

# Arguments
- `x`: Position vector or scalar (m)
- `t`: Time since release (s)
- `h0`: Background water table height (m)
- `Hm`: Initial mound height above h0 (m)
- `L`: Total width of the domain (m)
- `K`: Hydraulic conductivity (m/s)
- `Sy`: Specific yield (dimensionless)
- `h_bar`: Average saturated thickness (m), typically h0 + Hm/2
- `max_n`: Maximum harmonic term to compute (must be an odd integer)
"""
function boussinesq_mound_1d(x, t, h0, Hm, L, K, Sy, h_bar; max_n=51)
    # Initialize the sum matching the type of x (handles scalars or vectors)
    sum_terms = zeros(eltype(x), size(x))
    
    # Pre-calculate constant groups for performance
    spatial_factor = 4.0 * Hm / pi
    temporal_factor = (pi^2 * K * h_bar) / (Sy * L^2)
    
    # Loop over odd harmonics only
    for n in 1:2:max_n
        # Spatial component: sin(n * pi * x / L) / n
        spatial = sin.((n * pi .* x) ./ L) ./ n
        
        # Temporal component: exp(-n^2 * factor * t)
        decay = exp(-n^2 * temporal_factor * t)
        
        # Accumulate the term
        sum_terms .+= spatial .* decay
	end
    
    # Scale the accumulated terms and add background head
    return h0 .+ (spatial_factor .* sum_terms)
end

# ╔═╡ 97a74f48-b83f-420e-bc0e-7f408cab31c0
function boussinesq_mound_1d_b(x, t, h0, Hm, L, K, Sy, h_bar; max_n=51)
    # Initialize the sum matching the type of x (handles scalars or vectors)
    sum_terms = zeros(eltype(x), size(x))
    
    # Pre-calculate constant groups
    spatial_factor = 4.0 * Hm / pi
    temporal_factor = (pi^2 * K * h_bar) / (Sy * L^2)
    
    # Corrected loop structure
    for n in 1:2:max_n
        # Spatial component
        spatial = sin.((n * pi .* x) ./ L) ./ n
        
        # Temporal decay component
        decay = exp(-n^2 * temporal_factor * t)
        
        # Accumulate
        sum_terms .+= spatial .* decay
    end # <--- Corrected: was a stray '}'

    # Final result: h0 + (4*Hm/pi) * Sum
    return h0 .+ (spatial_factor .* sum_terms)
end

# ╔═╡ b9652acf-58fa-4b58-873f-886b22f448a6
L  = 256.0                # Domain length (m)

# ╔═╡ a691b620-ce52-443f-afc0-e55540a97d15

h0 = 5.0                  # Background head (m)


# ╔═╡ e96bd7af-495c-45da-bbdb-c8652c4fdb22
Hm = 0.06                  # Mound height (m)

# ╔═╡ 94dcd744-d774-4963-9fbf-d4cbb3c1f5cf
K  = 1500.0 / 86400.0       # Convert 10 m/day to m/s


# ╔═╡ 2e694e54-1958-4180-a664-7c957c110713
Sy = 0.2                  # Specific yield

# ╔═╡ dc49fe58-e9ff-45fa-a3b1-69a136bb0860
h_bar = h0 + (Hm / 2.0)   # Average thickness

# ╔═╡ 6a3232d9-6520-4cf7-8d33-71b4e04c323e
x_cells = collect(0.5:1.0:150)

# ╔═╡ 1a86a8ff-b74d-405b-a95a-86052025758f
 y10h = boussinesq_mound_1d(x_cells, 100, h0, Hm, L, K, Sy, h_bar)

# ╔═╡ 2aaba249-ea9c-43cd-aafb-8a8aaba7a9bd
plot(x_cells,y10h,show=true)

# ╔═╡ 55939618-bb99-413f-9636-62fbe5bf6fef


"""
    hantush_mound_cell(x, y, t, h0, w, K, Sy, h_bar, a, b)

Calculates the hydraulic head at a single coordinate (x,y) at time t 
using the Hantush (1967) unconfined rectangular mounding equation.
Assumes x and y are measured from the center of the rectangular basin.
"""
function hantush_mound_cell(x, y, t, h0, w, K, Sy, h_bar, a, b)
    if t <= 0.0
        return h0
    end
    
    nu = (K * h_bar) / Sy  # Diffusivity parameter
    
    # Define the integrand for Hantush's S* function
    integrand = tau -> begin
        if tau == 0.0
            return 0.0
        end
        sqrt_tau = sqrt(tau)
        
        # X-direction error functions
        term_x = erf((a + x) / (sqrt(4 * nu * t) * sqrt_tau)) + 
                 erf((a - x) / (sqrt(4 * nu * t) * sqrt_tau))
                 
        # Y-direction error functions
        term_y = erf((b + y) / (sqrt(4 * nu * t) * sqrt_tau)) + 
                 erf((b - y) / (sqrt(4 * nu * t) * sqrt_tau))
                 
        return term_x * term_y
    end
    
    # Integrate from 0 to 1 using Gauss-Kronrod quadrature
    S_star, _ = quadgk(integrand, 0.0, 1.0, rtol=1e-8)
    
    # Calculate head squared
    h2 = h0^2 + (w / K) * nu * t * S_star
    return sqrt(h2)
end


# ╔═╡ 02f5cb39-8c69-4131-a450-530099572583

"""
    hantush_mound_2d(x_vec, y_vec, t, h0, w, K, Sy, h_bar, a, b)

Generates a full 2D grid matrix of heads matching the coordinates in x_vec and y_vec.
"""
function hantush_mound_2d(x_vec, y_vec, t, h0, w, K, Sy, h_bar, a, b)
    h_matrix = zeros(length(x_vec), length(y_vec))
    for i in 1:length(x_vec)
        for j in 1:length(y_vec)
            # x_vec and y_vec are absolute coordinates, 
            # so we assume center of domain is the center of the mound
            x_rel = x_vec[i] - (maximum(x_vec)/2.0)
            y_rel = y_vec[j] - (maximum(y_vec)/2.0)
            
            h_matrix[i, j] = hantush_mound_cell(x_rel, y_rel, t, h0, w, K, Sy, h_bar, a, b)
        end
    end
    return h_matrix
end

# ╔═╡ d01f55ce-dc51-4f7a-ab7e-0b1170097af9
hantush_mound_cell(0,0, 3600, 2, 1.4e-5, 500/24, 0.2, 2, 10, 10)

# ╔═╡ 9bd30b8b-6cf1-4f8d-aaaa-7ecb923ddc08
"""
    hantush_mound_cell(x, y, t, h0, w, K, Sy, h_bar, a, b, t_off)

Calculates the hydraulic head at a single coordinate (x,y) at time t.
Handles both the growth phase (t <= t_off) and decay phase (t > t_off).
"""
function hantush_mound_cell_to(x, y, t, h0, w, K, Sy, h_bar, a, b, t_off)
    if t <= 0.0
        return h0
    end
    
    nu = (K * h_bar) / Sy  # Aquifer diffusivity
    
    # Helper function to compute the S* integral for a given time duration
    function compute_S_star(time_duration)
        if time_duration <= 0.0
            return 0.0
        end
        
        integrand = tau -> begin
            if tau == 0.0; return 0.0; end
            sqrt_tau = sqrt(tau)
            
            term_x = erf((a + x) / (sqrt(4 * nu * time_duration) * sqrt_tau)) + 
                     erf((a - x) / (sqrt(4 * nu * time_duration) * sqrt_tau))
                     
            term_y = erf((b + y) / (sqrt(4 * nu * time_duration) * sqrt_tau)) + 
                     erf((b - y) / (sqrt(4 * nu * time_duration) * sqrt_tau))
                     
            return term_x * term_y
        end
        
        S_star, _ = quadgk(integrand, 0.0, 1.0, rtol=1e-8)
        return S_star
    end

    # Phase 1: Mound is actively growing (Rain is still on)
    if t <= t_off
        S_star_1 = compute_S_star(t)
        h2 = h0^2 + (0.5*w / K) * nu * t * S_star_1
        
    # Phase 2: Mound is decaying (Rain stopped at t_off)
    else
        S_star_1 = compute_S_star(t)
        S_star_2 = compute_S_star(t - t_off)
        
        # Superposition: Positive rain since t=0 MINUS negative rain since t_off
        term1 = t * S_star_1
        term2 = (t - t_off) * S_star_2
        
        h2 = h0^2 + (0.5*w / K) * nu * (term1 - term2)
    end
    
    return sqrt(max(0.0, h2)) # Guard against minor negative floating-point issues
end

# ╔═╡ 0566ebda-133d-4a3f-a647-519174e327a5
"""
    hantush_mound_2d(x_vec, y_vec, t, h0, w, K, Sy, h_bar, a, b)

Generates a full 2D grid matrix of heads matching the coordinates in x_vec and y_vec.
"""
function hantush_mound_2d_to(x_vec, y_vec, t, h0, w, K, Sy, h_bar, a, b,toff)
    h_matrix = zeros(length(x_vec), length(y_vec))
    for i in 1:length(x_vec)
        for j in 1:length(y_vec)
            # x_vec and y_vec are absolute coordinates, 
            # so we assume center of domain is the center of the mound
            x_rel = x_vec[i]# - (maximum(x_vec)/2.0)
            y_rel = y_vec[j]# - (maximum(y_vec)/2.0)
            
            h_matrix[i, j] = hantush_mound_cell_to(x_rel, y_rel, t, h0, w, K, Sy, h_bar, a, b,toff)
        end
    end
    return h_matrix
end

# ╔═╡ c5a95c3a-277d-4cec-b130-6b1eb8d4e9d5
tvec=collect(0:600:36000)

# ╔═╡ 51d60340-e4c1-4df0-9032-8295b727d8d7
hcent=hantush_mound_cell_to.(126 .- 125, 0, tvec, 2.0, 1.3888e-5, 500/24/3600, 0.3, 2.0, 25, 25, 3600)

# ╔═╡ 215c0e9c-d212-441f-ac42-6fca77d32479
riverrate=(50/1000/3600)*50*50


# ╔═╡ 932821ea-a771-49a9-9f92-848583902a64
w=riverrate/(50*50)

# ╔═╡ 76de7206-6eea-4739-9a7d-e454df6b29d5


# ╔═╡ a43906ae-9c04-49f3-a2d2-8dabc5a5d3fa
BGff="groundwatertest_40.nc"

# ╔═╡ 50072ad0-8b44-49ad-945f-141899313176
Tscentre=dropdims(ncread(BGff,"h_gw_P0",start=[126,125,1],count=[1,1,-1]),dims=(1,2))

# ╔═╡ 05d674af-b56e-4d45-a09b-929b14c099f6
tttmodel=ncread(BGff,"time")

# ╔═╡ a63a6737-cffd-42dd-8232-24a6b0abcd81
begin
	plot(tvec,hcent,label="Hantush (1967) ", title="BG_Flood vs analytical solution at the center of the recharge", xlabel="time [s]",ylabel="hgw [m] ")
	plot!(tttmodel,Tscentre,lc=:red,label="BG_Flood",show=true)
end

# ╔═╡ 15dcd1de-0a68-46ca-a73a-b27d53c2ab68
xvecAS=collect(-100:1:100)

# ╔═╡ c8aa04a1-2dcb-47c7-90b4-0d6fea64257b
xvecASB=collect(-200:5:200)

# ╔═╡ 2e8fcab2-824c-4c9d-a9ee-3be24f3117db
hxs=hantush_mound_cell_to.(xvecAS, 0, 12*600, 2.0, 1.388e-5, 500/24/3600, 0.3, 2.0, 25, 25, 3600)

# ╔═╡ 237aa5f4-e94e-42d0-a9f9-8dec89eb56fa
modelxs=dropdims(ncread(BGff,"h_gw_P0",start=[1,125,13],count=[-1,1,1]),dims=(2,3))

# ╔═╡ f7689747-2208-4491-897f-f122d1bfd512
xxsmod=ncread(BGff,"xx_P0")

# ╔═╡ e74447c1-8fb5-4375-b3db-edd7d3ace23a
hallAS=hantush_mound_2d_to(xvecASB, xvecASB, 12*600, 2.0, 1.388e-5*.5, 1500/24/3600, 0.3, 2.0, 25, 25, 3600)

# ╔═╡ d129ceab-f1ca-4fa3-9b60-5a6350374379
grdimage(hallAS,show=true)

# ╔═╡ 3e0ba7c3-3204-4c85-b2cb-563341f21de5
totalVolAS=sum(hallAS.-2.0)*0.3

# ╔═╡ f7eb1299-134d-4a97-b28b-e2192f955db0
modelhw=dropdims(ncread(BGff,"h_gw_P0",start=[1,1,13],count=[-1,-1,1]),dims=(3))

# ╔═╡ 50adec77-dbf6-4f18-b487-9a10bf1cb594
modelwaterVol=sum(modelhw.-2.0)*0.3

# ╔═╡ 258346fa-461a-48e7-a6ee-f6ca676d635f


# ╔═╡ 05b20dfa-e506-4da3-ae31-7ed4e2f737b6
injectVol=0.03472*3600

# ╔═╡ be541022-c138-4a98-b679-968442b24d58
begin
	plot(xvecAS,hxs,label="Hantush (1967)",title="BG_Flood vs analytical solution X-Section through the center at t=6600s",xlabel="x [m]",ylabel="hgw [m] ")
	plot!(xxsmod.-125.0,modelxs,lc=:red,label="BG_Flood",show=true)
end

# ╔═╡ e3214624-b2e1-4223-ace6-b645be23ac88
500/24/3600


# ╔═╡ 2576223d-d890-4db0-bede-054718087ac0
md"""
# Groundwater infiltration and exfiltration
"""

# ╔═╡ 3c838a69-4c07-44e9-805b-a6fa9e117f59
test1BG="C:\\Project\\BG_Flood\\Investigation\\Groundwater_Test1\\groundwatertest1_2.nc"

# ╔═╡ 9557ca3f-4c27-465f-bd02-97cf3df593a1


# ╔═╡ 38a939e6-e234-40f9-a0c3-a1c69da53598


# ╔═╡ 50a28e8f-3ef8-4d04-b8cd-efa2e85c1fca
function plotgwnh(step)
	test1x=ncread(test1BG,"xx_P0")
	Test1zb=dropdims(ncread(test1BG,"zb_P0",start=[1,1,step],count=[-1,-1,1]),dims=(3))
	Test1zs=dropdims(ncread(test1BG,"zs_P0",start=[1,1,step],count=[-1,-1,1]),dims=(3))
	Test1zgw=dropdims(ncread(test1BG,"zs_gw_P0",start=[1,1,step],count=[-1,-1,1]),dims=(3))
	region=(0,250,0,1.1)
	plot(test1x,Test1zgw[:,115],region=region,lc=:green,label="groundwater")
	plot!(test1x,Test1zs[:,115],label="surface",lc=:blue)
	plot!(test1x,Test1zb[:,115],label="land",show=true)
end

# ╔═╡ 02b93d46-4b97-4b5d-bc3e-2690459a7953
plotgwnh(1)

# ╔═╡ 0b3ac4c7-e3f1-42ef-a529-eae4ec031742
plotgwnh(4)

# ╔═╡ b4a72f9f-48df-416c-bfba-5dbd843c9632
plotgwnh(8)

# ╔═╡ 2723f1c0-4237-4058-8cd4-30c4fd3a9209
plotgwnh(12)

# ╔═╡ 1b212431-d7c5-4567-9cf3-69ee4e720ca9
plotgwnh(16)

# ╔═╡ d5b1ab8a-9f05-40fd-a3e3-c3dfc8deb013
plotgwnh(20)

# ╔═╡ 424a4cbd-46ec-433b-b280-6a98b884c18d
plotgwnh(24)

# ╔═╡ 7c74c3f0-fed2-4b97-b7db-05865d792339
plotgwnh(28)

# ╔═╡ 65c62260-eb64-47f7-980a-446b25071d11
plotgwnh(32)

# ╔═╡ 72ecb147-3e51-49cb-875a-2de99f386229
plotgwnh(36)

# ╔═╡ 6b97c49f-f4ae-4531-9e65-de5d09e4affc
plotgwnh(40)

# ╔═╡ f3fde2a3-820a-49f1-bc52-448dbab103cd
plotgwnh(44)

# ╔═╡ c57cfbd3-7c48-4894-9056-d8063180ff34
plotgwnh(48)

# ╔═╡ b6422ade-e6a1-4989-8c61-74b591c8b8cc
plotgwnh(52)

# ╔═╡ b48d6f9d-54bf-4fce-aa0a-e4a95ad762cc
plotgwnh(56)

# ╔═╡ 841172f9-05c8-43ac-86c3-286207749adc
plotgwnh(60)

# ╔═╡ Cell order:
# ╠═546f34d0-5125-11f1-1dce-c79ac1688d87
# ╠═64553d93-ad06-42bd-ab2c-b1aaa7231189
# ╠═78ddbf7b-31a7-4232-9173-eb6fdf799189
# ╠═4e1d57f1-cb04-4a63-ab20-b205f9055e98
# ╠═97a74f48-b83f-420e-bc0e-7f408cab31c0
# ╠═b9652acf-58fa-4b58-873f-886b22f448a6
# ╠═a691b620-ce52-443f-afc0-e55540a97d15
# ╠═e96bd7af-495c-45da-bbdb-c8652c4fdb22
# ╠═94dcd744-d774-4963-9fbf-d4cbb3c1f5cf
# ╠═2e694e54-1958-4180-a664-7c957c110713
# ╠═dc49fe58-e9ff-45fa-a3b1-69a136bb0860
# ╠═6a3232d9-6520-4cf7-8d33-71b4e04c323e
# ╠═1a86a8ff-b74d-405b-a95a-86052025758f
# ╠═2aaba249-ea9c-43cd-aafb-8a8aaba7a9bd
# ╠═9804ea50-575d-471e-a3a9-86eeb08a8bec
# ╠═952f9644-540e-4e66-b6e7-dcae8abaf18b
# ╠═55939618-bb99-413f-9636-62fbe5bf6fef
# ╠═02f5cb39-8c69-4131-a450-530099572583
# ╠═d01f55ce-dc51-4f7a-ab7e-0b1170097af9
# ╠═9bd30b8b-6cf1-4f8d-aaaa-7ecb923ddc08
# ╠═0566ebda-133d-4a3f-a647-519174e327a5
# ╠═c5a95c3a-277d-4cec-b130-6b1eb8d4e9d5
# ╠═51d60340-e4c1-4df0-9032-8295b727d8d7
# ╠═215c0e9c-d212-441f-ac42-6fca77d32479
# ╠═932821ea-a771-49a9-9f92-848583902a64
# ╠═76de7206-6eea-4739-9a7d-e454df6b29d5
# ╠═1558fa3e-d026-4475-881c-ac9899fa9878
# ╠═a43906ae-9c04-49f3-a2d2-8dabc5a5d3fa
# ╠═50072ad0-8b44-49ad-945f-141899313176
# ╠═05d674af-b56e-4d45-a09b-929b14c099f6
# ╠═a63a6737-cffd-42dd-8232-24a6b0abcd81
# ╠═15dcd1de-0a68-46ca-a73a-b27d53c2ab68
# ╠═c8aa04a1-2dcb-47c7-90b4-0d6fea64257b
# ╠═2e8fcab2-824c-4c9d-a9ee-3be24f3117db
# ╠═237aa5f4-e94e-42d0-a9f9-8dec89eb56fa
# ╠═f7689747-2208-4491-897f-f122d1bfd512
# ╠═e74447c1-8fb5-4375-b3db-edd7d3ace23a
# ╠═d129ceab-f1ca-4fa3-9b60-5a6350374379
# ╠═3e0ba7c3-3204-4c85-b2cb-563341f21de5
# ╠═f7eb1299-134d-4a97-b28b-e2192f955db0
# ╠═50adec77-dbf6-4f18-b487-9a10bf1cb594
# ╠═258346fa-461a-48e7-a6ee-f6ca676d635f
# ╠═05b20dfa-e506-4da3-ae31-7ed4e2f737b6
# ╠═be541022-c138-4a98-b679-968442b24d58
# ╠═e3214624-b2e1-4223-ace6-b645be23ac88
# ╟─2576223d-d890-4db0-bede-054718087ac0
# ╠═3c838a69-4c07-44e9-805b-a6fa9e117f59
# ╠═9557ca3f-4c27-465f-bd02-97cf3df593a1
# ╠═38a939e6-e234-40f9-a0c3-a1c69da53598
# ╠═50a28e8f-3ef8-4d04-b8cd-efa2e85c1fca
# ╠═02b93d46-4b97-4b5d-bc3e-2690459a7953
# ╠═0b3ac4c7-e3f1-42ef-a529-eae4ec031742
# ╠═b4a72f9f-48df-416c-bfba-5dbd843c9632
# ╠═2723f1c0-4237-4058-8cd4-30c4fd3a9209
# ╠═1b212431-d7c5-4567-9cf3-69ee4e720ca9
# ╠═d5b1ab8a-9f05-40fd-a3e3-c3dfc8deb013
# ╠═424a4cbd-46ec-433b-b280-6a98b884c18d
# ╠═7c74c3f0-fed2-4b97-b7db-05865d792339
# ╠═65c62260-eb64-47f7-980a-446b25071d11
# ╠═72ecb147-3e51-49cb-875a-2de99f386229
# ╠═6b97c49f-f4ae-4531-9e65-de5d09e4affc
# ╠═f3fde2a3-820a-49f1-bc52-448dbab103cd
# ╠═c57cfbd3-7c48-4894-9056-d8063180ff34
# ╠═b6422ade-e6a1-4989-8c61-74b591c8b8cc
# ╠═b48d6f9d-54bf-4fce-aa0a-e4a95ad762cc
# ╠═841172f9-05c8-43ac-86c3-286207749adc
