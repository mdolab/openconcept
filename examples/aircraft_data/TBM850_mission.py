from __future__ import division

data = dict()
takeoff = dict()
takeoff['h'] = {'value': 0, 'units': 'ft'}
data['takeoff'] = takeoff

landing = dict()
landing['h'] = takeoff['h']
data['landing'] = landing

climb = dict()
climb['h0'] = takeoff['h']
climb['vs'] = {'value': 1500, 'units': 'ft/min'}
climb['time'] = {'value': 28000 / 1500 * 60, 'units': 's'}
climb['Ueas'] = {'value': 124, 'units': 'kn'}
data['climb'] = climb

cruise = dict()
cruise['h'] = {'value': 28000, 'units': 'ft'}
cruise['h0'] = cruise['h']

cruise['Ueas'] = {'value': 201, 'units': 'kn'}
data['cruise'] = cruise

descent = dict()
descent['h0'] = cruise['h0']
descent['hf'] = landing['h']
descent['vs'] = {'value': -600, 'units': 'ft/min'}
descent['time'] = {'value': 28000 / 600 * 60, 'units': 's'}

descent['Ueas'] = {'value': 140, 'units': 'kn'}
data['descent'] = descent

data['range'] = {'value': 1250, 'units': 'NM'}
data['payload'] = {'value': 1000, 'units': 'lb'}
data['design_range'] = {'value': 1250, 'units': 'NM'}
