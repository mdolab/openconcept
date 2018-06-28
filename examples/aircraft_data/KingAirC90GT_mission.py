data = dict()
mission = dict()
takeoff = dict()
takeoff['h'] = {'value': 0, 'units': 'ft'}
mission['takeoff'] = takeoff

landing = dict()
landing['h'] = takeoff['h']
mission['landing'] = landing

climb = dict()
climb['vs'] = {'value': 1500, 'units': 'ft/min'}
climb['Ueas'] = {'value': 124, 'units': 'kn'}
mission['climb'] = climb

cruise = dict()
cruise['h'] = {'value': 29000, 'units': 'ft'}
cruise['Ueas'] = {'value': 170, 'units': 'kn'}
mission['cruise'] = cruise

descent = dict()
descent['vs'] = {'value': -600, 'units': 'ft/min'}
descent['Ueas'] = {'value': 140, 'units': 'kn'}
mission['descent'] = descent

mission['range'] = {'value': 1000, 'units': 'NM'}
mission['payload'] = {'value': 1000, 'units': 'lb'}
data['mission'] = mission
