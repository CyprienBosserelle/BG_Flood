# Halo and gradient
Using multi-resolution in BG_Flood, we need to keep track of how blocks with different resolution talk to each other. This page tries to explain how this is done.

## Why halo
In BUQ grid we keep track of a ring of cells on every edge of the block. SO that when we need a neighbor value we do not need to look at it from another block. It also makes it cleaner and avoid repeating the costly calculation needed when two blocks are at a different level of refinment.

## restriction and prolongation
In Basilisk the operation to calculate a value from one level to another are called restriction and prolongation depending whether you calculate from coarse to fine or fine to coarse.

### Prolongation
prolongation is the action of extending a value from a coarse cell to a finer cell. Often this is done by using the gradient value. e.g.:
child = parent + Gp*dx*0.5

### Restriction
Restriction is where we calculate the value of a coarse cell from values of fine cells. This is usually done with cell average.
parent = 0.25*(child1+child2+child3+child4)

## Implication for gradients
For gradient the story is a little more complex. First because there is a chicken and egg situation with prolongation. But also because we can calculate the gradient at the halo using the normal gradient operation or use the restriction/prolongation to calculate gradient.

## New calculations

## Conserving elevation
When using prolongation at the wet/dry interface can lead to inconsistencies between h and zs. to limit the inconsistency zs is calculated from h after a prolongation calculation (see refine_linear). While this conserves mass, it, however, leads to a violation of the lake-at-rest resulting in (small) spurious velocity at that interface. To remove the instability and preserve the elevation of the water level (rather than mass) we use a conserve elevation option (conserveelevation = true). This gets rid of the instability and preserves the elevation of the water level but then violates the mass conservation.
