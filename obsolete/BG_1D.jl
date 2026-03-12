# BG_1D

function minmod2(theta, s0, s1, s2)
    #
    if (s0 < s1 && s1 < s2) {
		d1 = theta*(s1 - s0);
		d2 = (s2 - s0) / 2.0;
		d3 = theta*(s2 - s1);
		if (d2 < d1) d1 = d2;
		return min(d1, d3);
	}
	if (s0 > s1 && s1 > s2) {
		d1 = theta*(s1 - s0)
		d2 = (s2 - s0) / 2.0
		d3 = theta*(s2 - s1);
		if (d2 > d1) d1 = d2;
		return max(d1, d3);
	}
	return 0.0;
end

function gradient(nx, theta, delta, a, dadx)
		ix=collect(1:nx);
		xplus = min.(ix + 1, nx);
		xminus = max.(ix - 1, 1);



		as=a;
		al=a[xminus];
		ar=a[xplus];
		dadx = minmod2.(theta, al, a, ar) ./ delta;

end


function kurganov()
	#
	ix=collect(1:nx);

	dx=0.5*delta

	xplus = min.(ix + 1, nx);
	xminus = max.(ix - 1, 1);

	dhdxmin = dhdx[xminus];
	hi=h;
	hn=h[xminus];

	wet=(hi.>eps) .| (hn.>eps)

	zi=zs.-hi;
	zl = zi .- dx.*(dzsdx .- dhdx);
	zn = zs[xminus] .- hn;
	zr = zn .+ dx.*(dzsdx[xminus] .- dhdxmin);

	zlr = max(zl, zr);

	hl = hi .- dx.*dhdx;
	up = uu .- dx.*dudx;
	hp = max.(0.0, hl .+ zl .- zlr);
	hr = hn .+ dx.*dhdxmin;
	um = uu[xminus] + dx*dudx[xminus];
	hm = max(0.0, hr + zr - zlr);
	### To be continued

end

function loop(h,zs,u,v)
    # Calculate gradients for h zs u v
	gradient(nx, theta, delta, h, dhdx);
	gradient(nx, theta, delta, zs, dzsdx);
	gradient(nx, theta, delta, u, dudx);
	gradient(nx, theta, delta, v, dvdx)

    # Kurganov scheme


    # Reduce dtmax

    # Update step 1

    # Advance 0.5*dt

    # Calculate gradient on advance

    # Kurganov scheme

    # update step 2

    # Advance full step

    # Clean up

end
