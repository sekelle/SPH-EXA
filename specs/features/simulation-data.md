# Feature: SimulationData Split (Phase 2)

Refactor `SimulationData` to hold basic particle fields directly and SPH-specific fields
in a dedicated member. `SimulationData` becomes the propagator template parameter.

## Data member classification

### Basic fields (in SimulationData or BasicParticlesData member)
FieldVectors: keys, x, y, z, x_m1, y_m1, z_m1, m, h, vx, vy, vz, ax, ay, az, rung, id, type

General scalars: iteration, numParticlesGlobal, numParticlesGlobalPrev, ttot, minDt, minDt_m1,
etot, ecin, eint, egrav, linmom, angmom, g, eps, etaAcc, ng0, ngmax, maxDtIncrease,
removeUnconvergedParticles, totalNeighbors, localNeighbors, maxHalos, stackUsedNc, stackUsedGravity

Data structures: neighbors, treeView, traversalStack, allocGrowthRate_

I/O: outputFieldIndices, outputFieldNames

### SPH fields (in SphFields or HydroData member)
FieldVectors: rho, temp, u, du, du_m1, p, prho, tdpdTrho, c, cv, mue, mui, divv, curlv,
c11, c12, c13, c22, c23, c33, alpha, xm, kx, gradh, dV11, dV12, dV13, dV22, dV23, dV33,
nc, ugrav, wh, whd

SPH scalars: gamma, polytropic_index, polytropic_const, eosChoice, muiConst, soundSpeedConst,
Kcour, Krho, minDtCourant, minDtRho, alphamin, alphamax, decay_constant, Atmin, Atmax, ramp,
sincIndex, kernelChoice, K

## Scenarios

### Scenario: SimulationData provides basic field access
Given: a SimulationData instance
When: basic fields (x, y, z, h, m, type) are accessed
Then: they are available directly without going through a hydro member
Test: `TEST(SimulationData, providesBasicFieldAccess)`

### Scenario: SimulationData provides SPH field access
Given: a SimulationData instance
When: SPH fields (rho, p, u, c) are accessed
Then: they are available through an SPH-specific member
Test: `TEST(SimulationData, providesSphFieldAccess)`

### Scenario: Basic and SPH fields have uniform sizes
Given: a SimulationData instance
When: both basic and SPH fields are allocated and resized to N
Then: all FieldVectors have size N (INV-13)
Test: `TEST(SimulationData, uniformFieldSizes)`

### Scenario: Propagator template parameter is SimulationData
Given: any propagator (HydroVeProp, NbodyProp, HydroDarkProp)
When: it is instantiated
Then: it accepts SimulationData as its DataType template parameter
Test: `TEST(SimulationData, propagatorTemplateParameter)` (compile-time)

### Scenario: HydroVeProp works after split
Given: a HydroVeProp instantiated with new SimulationData
When: a Sedov blast simulation is run for 10 steps
Then: results match the pre-refactor output within floating-point tolerance
Test: `TEST_F(SimulationDataRegression, hydroVePropSedov)`

### Scenario: HydroProp works after split
Given: a HydroProp (standard SPH) instantiated with new SimulationData
When: a Sedov blast simulation is run for 10 steps
Then: results match the pre-refactor output within floating-point tolerance
Test: `TEST_F(SimulationDataRegression, hydroPropSedov)`

### Scenario: NbodyProp works without SPH fields
Given: an NbodyProp instantiated with new SimulationData
When: SPH fields are not activated
Then: the propagator runs correctly using only basic fields + gravity
Test: `TEST_F(SimulationDataRegression, nbodyPropGravityOnly)`

### Scenario: Domain sync works with new structure
Given: a SimulationData with basic + SPH fields
When: `domain.syncGrav()` is called with all active fields
Then: particles are correctly distributed, halos exchanged, all field values preserved
Test: `TEST_F(SimulationDataMpi, domainSyncPreservesFields)` (MPI, 2+ ranks)

### Scenario: Field activation works per propagator
Given: a SimulationData instance
When: HydroDarkProp activates basic fields as conserved and SPH fields as dependent
Then: basic fields are always allocated, SPH fields allocated on demand
Test: `TEST(SimulationData, fieldActivationPerPropagator)`

## Invariants tested
- INV-13 (uniform array sizes)
- INV-1 (particle count preserved through domain sync)
- INV-3 (mass conservation)
