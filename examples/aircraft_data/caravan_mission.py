data = dict()
mission = dict()
takeoff = dict()
takeoff['h'] = {'value': 0, 'units': 'ft'}
mission['takeoff'] = takeoff

landing = dict()
landing['h'] = takeoff['h']
mission['landing'] = landing

climb = dict()
climb['vs'] = {'value': 850, 'units': 'ft/min'}
climb['Ueas'] = {'value': 104, 'units': 'kn'}
mission['climb'] = climb

cruise = dict()
cruise['h'] = {'value': 18000, 'units': 'ft'}
cruise['Ueas'] = {'value': 129, 'units': 'kn'}
mission['cruise'] = cruise

descent = dict()
descent['vs'] = {'value': -400, 'units': 'ft/min'}
descent['Ueas'] = {'value': 100, 'units': 'kn'}
mission['descent'] = descent

mission['range'] = {'value': 250, 'units': 'NM'}
mission['payload'] = {'value': 3500, 'units': 'lb'}
data['mission'] = mission
