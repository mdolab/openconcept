# DATA FOR TBM T80
# Collected from various sources
# including SOCATA pilot manual
from __future__ import division

data = dict()
ac = dict()
# ==AERO==================================
aero = dict()
aero['CLmax_TO']   = {'value' : 2.0}
aero['Vstall_land'] = {'value': 100, 'units':'kn'}
aero['Cl_max'] = {'value': 1.1}

polar = dict()
polar['e']              = {'value' : 0.801}
polar['CD0_TO']         = {'value' : 0.03}
polar['CD0_cruise']     = {'value' : 0.01925}

aero['polar'] = polar
ac['aero'] = aero

# ==GEOMETRY==============================
geom = dict()
wing = dict()
wing['S_ref']           = {'value': 124.6, 'units': 'm**2'}
wing['AR']              = {'value': 9.45}
wing['c4sweep']         = {'value': 25.0, 'units': 'deg'}
wing['taper']           = {'value': 0.159}
wing['toverc']          = {'value': 0.12}
geom['wing'] = wing

hstab = dict()
hstab['S_ref']          = {'value': 32.78, 'units': 'm**2'}
hstab['c4_to_wing_c4']  = {'value': 17.9, 'units': 'm'}
hstab['AR']             = {'value': 6.16}
hstab['taper']          = {'value': 0.203}
hstab['c4sweep']        = {'value': 30, 'units':'deg'}
geom['hstab'] = hstab

vstab = dict()
vstab['S_ref']          = {'value': 26.44, 'units': 'm**2'}
vstab['c4sweep']        = {'value': 35, 'units': 'deg'}
vstab['AR']             = {'value': 1.94}
vstab['toverc']         = {'value': 0.12}
geom['vstab'] = vstab

nosegear = dict()
nosegear['length'] = {'value': 4, 'units': 'ft'}
nosegear['num_wheels'] = {'value': 2}
geom['nosegear'] = nosegear

maingear = dict()
maingear['length'] = {'value': 5.9, 'units': 'ft'}
maingear['num_wheels'] = {'value': 4}
maingear['num_shock_struts'] = {'value': 2}
geom['maingear'] = maingear

fuselage = dict()
fuselage['S_wet']       = {'value': 859, 'units': 'm**2'}
fuselage['width']       = {'value': 3.76, 'units': 'm'}
fuselage['length']      = {'value': 39.12, 'units': 'm'}
fuselage['height']      = {'value': 4.01, 'units': 'm'}
geom['fuselage'] = fuselage

ac['geom'] = geom

# ==WEIGHTS========================
weights = dict()
weights['MTOW']         = {'value': 79002, 'units': 'kg'}
weights['OEW']          = {'value': 0.530*79002, 'units': 'kg'}
weights['W_fuel_max']   = {'value': 0.266*79002, 'units': 'kg'}
weights['MLW']          = {'value': 66349, 'units': 'kg'}
weights['max_payload']          = {'value': 44640, 'units': 'lb'}

ac['weights'] = weights

# ==PROPULSION=====================
propulsion = dict()
engine = dict()
engine['rating']        = {'value': 27000, 'units': 'lbf'}
engine['BPR']           = {'value': 11}
engine['weight']           = {'value': 5000, 'units':'lb'}
propulsion['engine']    = engine
propulsion['num_engines'] = {'value': 2}

ac['propulsion'] = propulsion

# Some additional parameters needed by the empirical weights tools
ac['num_passengers_max'] = {'value': 189}
ac['num_flight_deck_crew'] = {'value': 2}
ac['num_cabin_crew'] = {'value': 4}
ac['cabin_pressure'] = {'value': 8.95, 'units': 'psi'}
ac['q_cruise'] = {'value': 212.662, 'units': 'lb*ft**-2'}
data['ac'] = ac