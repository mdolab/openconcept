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
climb['vs'] = {'value': 850, 'units': 'ft/min'}
climb['Ueas'] = {'value': 104, 'units': 'kn'}
climb['time'] = {'value': 18000 / 850 * 60, 'units': 's'}
data['climb'] = climb

cruise = dict()
cruise['h'] = {'value': 18000, 'units': 'ft'}
cruise['h0'] = cruise['h']
cruise['Ueas'] = {'value': 129, 'units': 'kn'}
data['cruise'] = cruise

descent = dict()
descent['h0'] = cruise['h0']
descent['hf'] = landing['h']
descent['vs'] = {'value': -400, 'units': 'ft/min'}
descent['time'] = {'value': 18000 / 400 * 60, 'units': 's'}
descent['Ueas'] = {'value': 100, 'units': 'kn'}
data['descent'] = descent

data['range'] = {'value': 250, 'units': 'NM'}
data['design_range'] = {'value': 250, 'units': 'NM'}
data['payload'] = {'value': 3500, 'units': 'lb'}
