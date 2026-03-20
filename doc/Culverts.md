# Culverts

Cluverts are modelled in the code based on [HEC-RAS model](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.3/modeling-culverts).
The model of culvert include a large variety of linear structures,
such as pipes with pump, clapped or one-way only limited culverts, and "classical" two-way culverts.
Different shapes can be supported, such as ellipsoides (and cylindique), box and arch culverts.

## Calculation of the flow throuh a culvert
For the calculation of the flow through a culvert, different types of headlosses are considered given the geometry of the culvert and the flow conditions.
The flow throught the culvert can be piloted by the openning capacity (Inlet control headwater) or by the culvert area and/or downstream conditions (Outlet controle headwater). Basically, if the flow became supercritical in the culvert, it will be inlet controled (mainly short, steep culverts); if it stay subcritical, it will be outlet controled (longer, rough culverts, full barrel flow).

LINK TO IMAGE from HEC-RAS: https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.3/modeling-culverts/culvert-hydraulics/flow-analysis-for-culverts

To identify the appropriated flow regime for computing the flow through the culvert, both inlet and outlet control headwater conditions are calculated.
Then the minimum flow between the two conditions is retained (the concept of "minimum
performance", https://transportation.ky.gov/Highway-Design/Drainage%20Manual/HDS-05%282012-4%29.pdf, p83) to determin the flow through the culvert.

### Inlet control headwater
#### In HECRAS 

##### Unsubmerged inlet

$$ \frac{HW_{inlet}}{D} = \frac{H_c}{D} + K \left[\frac{Q}{AD^{0.5}}\right]^{M} -0.5*S $$

$$ \frac{HW_{inlet}}{D} = K \left[\frac{Q}{AD^{0.5}}\right]^{M} $$


where $D$ is the culvert rise ("vertical width of the culvert"), $H_c$ the critical head, $K$ and $M$ are culvert type coefficients, $A$ the cross section area of the culvert, $S$ the culvert slope._
##### Submerged inlet
$$ \frac{HW_{inlet}}{D} = c \left[\frac{Q}{AD^{0.5}}\right]^2 = 0.5*S $$


#### Simplified version
##### Unsubmerged inlet
$$ Q=K A \sqrt{2 g H_w} $$
where $K$ is the inlet loss coefficient, $A$ the full cross section area of the culvert, $H_w$ the headwater above the culvert inlet.
$K$ can be set based on culvert type (~0.06 for sharpe edge, up to ~1.5 for round edge) (use a default of 0.6 for square edge)

##### Submerged inlet
$$ Q = C_d A \sqrt{2 g H_w} $$
\bold{check if not H_inlet-H_outlet}
where $C_d$ is the discharge coefficient (~0.62 for square section to 1.0 for round entrance), A the cross section area of the culvert, $H_w$ the headwater above the culvert inlet. (use a default of 0.8 for square orifice)


### Outlet control headwater

The outlet control headwater is calculated based on the Bernoulli equation between the outlet and the inlet of the culvert.
$$ Z_{s2} + \frac{V_2^2}{2g} = Z_{s1} + \frac{V_1^2}{2g} + H_L $$
Where $H_L$ are the hydraulic losses throught he culvert, $Z_{s*}$ elevation at culvert extremity.

$H_L$ contains the friction loss calculated based on the Manning equation, the entrance and the exit loss.
$$h_f= \frac{n^2 V^2 L}{R^{4/3}}$$

The head loss is then related to the velocity as:
$$ H_L = h_{en} + h_f + h_{ex} = k_{en} \frac{V^2}{2g} + \frac{n^2 V^2 L}{R^{4/3}} + k_{ex} \frac{V^2 - V_2^2}{2g} $$
Where $h_{en}$ is the entrance loss, $h_f$ the friction loss and $h_{ex}$ the exit loss, $V$ the velocity in the culvert.

#### Implemented version
The velocity is calculated based of the head losses (entrance, exit and linear along the pipe). It is then related to the wetted area using the normal flow hypothesis (and Manning equation). This is done with a straight relation for square culverts but need iterations for circles ones. Then the discharge through the culvert is calculated. 


### Flow variable set-up
#### Normal depth of the flow in the culvert
This is the water depth of an uniform flow in an open channel (infinite channel in lenght, cst flow rate).
Done by satisfying the Manning equation through iterative method:
$$Q = \frac{1}{n}\; A_{wet}\; R_{wet}^{2/3}\; S^{1/2}\,*\,1.49 $$


#### Critical depth of the flow in the culvert
This is the water depth for which the specific energy is minimum for a given flow rate.
It is used at the outlet of outlet controled culverts with low tail water conditions and influence inlet control headwater for unsubmerged conditions.
Done by satisfying the equation by iterative method:
$$ Q^2 / g = A^3 / T $$
where $T$ is the top width of the flow section at the critical depth.

For a rectangular section:
$$ y_c = \left(\frac{Q^2}{g b^2}\right)^{1/3} $$
where $y_c$ is the critical depth.

#### Entrance loss coefficient
The entrance loss coefficient $K$ is a function of the entrance velocity and culvert type coefficient.
It is caluclated using: 
$$ h_{en} = K_{en} \left(\frac{V_{en}^2}{2g}\right) $$
A default entrance loss coefficient is set to 0.5. See tables there: https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.3/modeling-culverts/culvert-data-and-coefficients/entrance-loss-coefficient

#### Exit loss coefficient
The exit loss coefficient is a function of the exit velocity just inside the culvert and the velocity of the flow just downstream from the culvert.
It is caluclated using:
$$ h_{ex} = K_{ex} \left(\frac{V_{ex}^2}{2g} - \frac{V_{ds}^2}{2g}\right) $$
A default exit loss coefficient is set to 1.0 (1.0 for sudden expansion of the flow to 0.3 if transition is less abrupt). See tables there: https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.3/modeling-culverts/culvert-data-and-coefficients/exit-loss-coefficient

#### Culvert slope
The culvert slope is the mean slope of the flow line along the culvert.
The culvert slope S is calculated as:
$$ S = \frac{Z_{inlet} - Z_{outlet}}{(L^2 -(Z_{inlet} - Z_{outlet})^2)^{1/2}} $$
Where $L$ is the length of the culvert, $Z_{inlet}$ and $Z_{outlet}$ are the invert elevations of the culvert at inlet and outlet respectively.

On the contrary to HEC-RAS, no weir flow is modelled in the culvert model, it is left to the 2D surface flow model to take care of it.