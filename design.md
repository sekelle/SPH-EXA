# Implementation of particle types in SPH-EXA

## Overview
The goal is to implement different types of particles in SPH-EXA. Examples of different types are
gas particles, dark matter particles or star particles. Currently, all particles are treated as gas particles.
Different types of particles are subject to different types of forces.

## Design principles

### Types of forces
We support:
  - gravitational forces
  - hydrodynamical (SPH) forces

### Types of fields
Fields are the variables associated with a particle.

* Basic fields: all particles require them for time-step integration. They are:
    - keys
    - x
    - y
    - z
    - x_m1
    - y_m1
    - z_m1
    - m
    - h
    - vx
    - vy
    - vz
    - ax
    - ay
    - az
    - rung
    - id
    - type (new field to be implemented, can be uint8_t to support 256 different types)

* SPH fields: required by gas particles subject to hydrodynamical forces:
    - rho
    - temp
    - u
    - prho
    - tdpdTrho
    - c
    - cv
    - mue
    - mui
    - divv
    - curlv
    - c11, c12, c13, c22, c23, c33 
    - alpha
    - xm
    - kx
    - gradh
    - dV11, dV12, dV13, dV22, dV23, dV33 

These are the sets of possible fields that must be supported. However, the specific Propagator class
decides which of these fields it will activate and use.

### Desired particle types

We will add these particle types
* Dark matter particles: they only have the basic fields
* Gas particles: they have the basic plus the SPH fields

Extensibility: Further types to be added in the future will have the basic fields plus extra fields
specific to their type

### Domain synchronization

We enforce the same size for all `FieldVectors`, effectively leaving the extra SPH fields unused for dark matter particles.
This means that we can call `domain.sync` or `domain.syncGrav` as is and pass all fields in use by the propagator.

### Particle ordering

Currently, particles are ordered by their SFC key. To support multiple particle types, we need to support two particle
orderings:
* Order-1: sorted by SFC key, same as existing ordering
* Order-2: sorted first by type, then by SFC key

## Implementation plan

### Splitting the ParticlesData class
We need to split this class that currently holds all fields into one class for the basic fields and one that
holds the SPH fields. We need to iterate on the design to resolve which of the parameter go to the basic fields class
and which go to SPH.

### Implementing a new propagator that activates different particle types
We build on top of `HydroProp` to create a new `HydroDarkProp` that will activate the new type field.
