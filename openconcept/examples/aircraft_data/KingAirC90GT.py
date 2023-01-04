# DATA FOR King Air C90GT
# Collected from AOPA Pilot article
# and rough photogrammetry

data = dict()
ac = dict()
# ==AERO==================================
aero = dict()
aero["CLmax_TO"] = {"value": 1.52}

polar = dict()
polar["e"] = {"value": 0.80}
polar["CD0_TO"] = {"value": 0.040}
polar["CD0_cruise"] = {"value": 0.022}

aero["polar"] = polar
ac["aero"] = aero

# ==GEOMETRY==============================
geom = dict()
wing = dict()
wing["S_ref"] = {"value": 27.308, "units": "m**2"}
wing["AR"] = {"value": 8.5834}
wing["c4sweep"] = {"value": 1.0, "units": "deg"}
wing["taper"] = {"value": 0.397}
wing["toverc"] = {"value": 0.19}
geom["wing"] = wing

fuselage = dict()
fuselage["S_wet"] = {"value": 41.3, "units": "m**2"}
fuselage["width"] = {"value": 1.6, "units": "m"}
fuselage["length"] = {"value": 10.79, "units": "m"}
fuselage["height"] = {"value": 1.9, "units": "m"}
geom["fuselage"] = fuselage

hstab = dict()
hstab["S_ref"] = {"value": 8.08, "units": "m**2"}
hstab["c4_to_wing_c4"] = {"value": 5.33, "units": "m"}
geom["hstab"] = hstab

vstab = dict()
vstab["S_ref"] = {"value": 3.4, "units": "m**2"}
geom["vstab"] = vstab

nosegear = dict()
nosegear["length"] = {"value": 0.95, "units": "m"}
geom["nosegear"] = nosegear

maingear = dict()
maingear["length"] = {"value": 0.88, "units": "m"}
geom["maingear"] = maingear

ac["geom"] = geom

# ==WEIGHTS========================
weights = dict()
weights["MTOW"] = {"value": 4581, "units": "kg"}
weights["W_fuel_max"] = {"value": 1166, "units": "kg"}
weights["MLW"] = {"value": 4355, "units": "kg"}
weights["W_battery"] = {"value": 100, "units": "kg"}

ac["weights"] = weights

# ==PROPULSION=====================
propulsion = dict()
engine = dict()
engine["rating"] = {"value": 750, "units": "hp"}
propulsion["engine"] = engine

propeller = dict()
propeller["diameter"] = {"value": 2.28, "units": "m"}
propulsion["propeller"] = propeller

motor = dict()
motor["rating"] = {"value": 527.2, "units": "hp"}
propulsion["motor"] = motor

generator = dict()
generator["rating"] = {"value": 1083.7, "units": "hp"}
propulsion["generator"] = generator

ac["propulsion"] = propulsion

# Some additional parameters needed by the empirical weights tools
ac["num_passengers_max"] = {"value": 8}
ac["q_cruise"] = {"value": 98, "units": "lb*ft**-2"}
ac["num_engines"] = {"value": 2}
data["ac"] = ac
