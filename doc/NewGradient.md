# Halo and gradient
## Why halo
In BUQ grid we keep track of a ring of cells on every edge of the block. SO that when we need a neighbour value we do not need to look at it from another block. It also makes it cleaner and avoid repeating the costly calculation needed when two blocks are at a different level of refinment.

## restriction and prolongation
In Basilisk the operation to calculate a value from one level to another are called restriction and prolongation depending whether you calculate from coarse to fine or fine to coarse.

### Prolongation
prolongation is teh action of extending a value from a coarse cell to a finer cell. Often this is done by using the gradient value. e.g.:
child = parent + Gp*dx*0.5

### Restriction
Restictrin is where we calculate the value of a coasrse cell from values of fine cells. Thsi si usually done with cell average.
parent = 0.25*(child1+child2+child3+child4)

## Implication for gradients
For gradient the story is a little more complex. First because there is a chicken and egg situation with prolongation. But also because we can calculate the gradient at the halo using the normal gradient operation or use the restriction/prolongation to calculate gradient.

## New calculations

## Conserving elevation
I suspect conserving elevation is not working properly at the moment.
