#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt

import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert

###############################################################################
# Create a measurement scheme for 51 electrodes, spacing 1
scheme = ert.createERTData(
    elecs=np.linspace(start=0, stop=50, num=51),
    schemeName='dd'
)
m = scheme['m']
n = scheme['n']
scheme['m'] = n
scheme['n'] = m

scheme.set('k', [1 for x in range(scheme.size())])

###############################################################################
# Mesh generation
world = mt.createWorld(
    start=[-55, 0], end=[105, -80], worldMarker=True)

conductive_anomaly = mt.createCircle(
    pos=[10, -7], radius=5, marker=2
)

polarizable_anomaly = mt.createCircle(
    pos=[40, -7], radius=5, marker=3
)

plc = mt.mergePLC((world, conductive_anomaly, polarizable_anomaly))

# local refinement of mesh near electrodes
for s in scheme.sensors():
    plc.createNode(s + [0.0, -0.2])

mesh_coarse = mt.createMesh(plc, quality=33)
# additional refinements
mesh = mesh_coarse.createH2()

pg.show(plc, marker=True)
pg.show(plc, markers=True)
pg.show(mesh)
###############################################################################
# Prepare the model parameterization
# We have two markers here: 1: background 2: circle anomaly
# Parameters must be specified as a complex number, here converted by the
# utility function :func:`pygimli.utils.complex.toComplex`.
rhomap = [
    [1, pg.utils.complex.toComplex(100, 0 / 1000)],
    # Magnitude: 50 ohm m, Phase: -50 mrad
    [2, pg.utils.complex.toComplex(50, 0 / 1000)],
    [3, pg.utils.complex.toComplex(100, -50 / 1000)],
]

###############################################################################
# Do the actual forward modeling
data = ert.simulate(
    mesh,
    res=rhomap,
    scheme=scheme,
    # noiseAbs=0.0,
    # noiseLevel=0.0,
)

###############################################################################
# Visualize the modeled data
# Convert magnitude and phase into a complex apparent resistivity
rho_a_complex = data['rhoa'].array() * np.exp(1j * data['phia'].array())
np.savetxt(
    'data_rre_rim.dat', np.hstack((rho_a_complex.real, rho_a_complex.imag))
)
