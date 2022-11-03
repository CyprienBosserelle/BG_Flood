# Infiltration Model: Initial Loss - Continuous Loss implementation

The Initial Loss - Continuous Loss (ILCL) is a very basic model for infiltration of surface water in the soil.
It requires the input of two maps, based on the soil properties: one containing an initial loss coefficient $il$ in mm, the second containing a continuous loss coefficient $cl$ in mm/s.

In this model, the initial and continuous loss are applied directly on the water elevation computed on each cell (and not by modifying the rain input).
The value of the initial loss $il$ is estimated to be the total of water infiltrating in the ground before the beginning of the surface runoff, whereas the continuous loss $cl$ is the loss that occurs, on wet cells, from the begining of the surface runoff to the end of the simulation.
The water absorbed in the ground will be tracked using the ground water elevation variable $hgw$ but wont be reintroduced to the surface flow through the computation process.

On each cell, at each simulation step, we can express the quantity of water absorbed in the ground $ha_{i,j,t}$ using:
$$
\begin{equation}
  ha_{i,j,t} =
    \begin{cases}
      h_{i,j,t} & \text{if  } hgw_{i,j} + h_{i,j,t} < il_{i,j}\\
      cl_{i,j} & \text{if  } hgw_{i,j} > il_{i,j}\\
      \min((il_{i,j} - hgw_{i,j,t-1}) + cl_{i,j}, h_{i,j,t}) & \text{otherwise}
    \end{cases}       
\end{equation}
$$

where $il_{i,j}$ and $cl_{i,j}$ are respectively the initial loss and continuous loss coefficient at a given $(i,j)$ cell location, and $hgw_{i,j,t}$ is the accumulated ground water accumulated at this cell location since the begining of the simulation.

The water absorbed is then added to the ground water tracking variable:
$$hgw_{i,j,t}=hgw_{i,j,t-1} + ha_{i,j,t}$$
 and removed from the surface water height and the surface water elevation (not shown here):
 $$h_{i,j,t} = h_{i,j,t} - ha_{i,j,t}$$

The following figure shows a representation of the initial loss - continuing loss model with $il = 10 mm$ and $cl = 1 mm/s$ :

![Initial loss and continuing loss reprensentation during a cell-wetting event](./RainLosses.png)

*Initial loss and continuing loss reprensentation during a cell-wetting event*

