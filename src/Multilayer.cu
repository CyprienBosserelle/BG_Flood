#include "Multilayer.h"

template <class T> void calcAbaro()
{

	T gmetric = (2. * fm.x[i] / (cm[i] + cm[i - 1]))

	a_baro[i] (G*gmetric*(eta[i-1] - eta[i])/Delta)
}

template <class T> void CalcfaceVal()
{

	T gmetric = (2. * fm.x[i] / (cm[i] + cm[i - 1]));

	T ax = (G * gmetric * (eta[i - 1] - eta[i]) / Delta);

	T H = 0.;
	T um = 0.;
	T Hr = 0.;
	T Hl = 0.;

	//foreach_layer() {
	{
		Hr += h[], Hl += h[-1];
		T hl = h[-1] > dry ? h[-1] : 0.;
		T hr = h[] > dry ? h[] : 0.;


		hu.x[] = hl > 0. || hr > 0. ? (hl * u.x[-1] + hr * u.x[]) / (hl + hr) : 0.;
		double hff;
#if DRYSTEP
		if (Hl <= dry)
			hff = fmax(fmin(zb[] + Hr - zb[-1], h[]), 0.);
		else if (Hr <= dry)
			hff = fmax(fmin(zb[-1] + Hl - zb[], h[-1]), 0.);
		else
#endif // DRYSTEP
		{
			double un = pdt * (hu.x[] + pdt * ax) / Delta, a = sign(un);
			int i = -(a + 1.) / 2.;
			double g = h.gradient ? h.gradient(h[i - 1], h[i], h[i + 1]) / Delta :
				(h[i + 1] - h[i - 1]) / (2. * Delta);
			hff = h[i] + a * (1. - a * un) * g * Delta / 2.;
		}
		hf.x[] = fm.x[] * hff;

		if (fabs(hu.x[]) > um)
			um = fabs(hu.x[]);

		hu.x[] *= hf.x[];
		ha.x[] = hf.x[] * ax;

		H += hff;
	}

	if (H > dry) {
		double c = um / CFL + sqrt(G * (hydrostatic ? H : Delta * tanh(H / Delta))) / CFL_H;
		if (c > 0.) {
			double dt = min(cm[], cm[-1]) * Delta / (c * fm.x[]);
			if (dt < dtmax)
				dtmax = dt;
		}
	}
	pdt = dt = dtnext(dtmax);
}
