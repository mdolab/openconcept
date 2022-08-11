# https://www.globalair.com/aircraft-for-sale/Specifications?specid=1273
# http://www.airliners.net/aircraft-data/cessna-208-caravan-i-grand-caravan-cargomaster/158
# http://b.org.za/fly/C208Bproc.pdf

data = dict()
ac = dict()
# ==AERO==================================
aero = dict()
aero["CLmax_TO"] = {"value": 2.25}

polar = dict()
polar["e"] = {"value": 0.8}
polar["CD0_TO"] = {"value": 0.033}
polar["CD0_cruise"] = {"value": 0.027}

aero["polar"] = polar
ac["aero"] = aero

# ==GEOMETRY==============================
geom = dict()
wing = dict()
wing["S_ref"] = {"value": 26.0, "units": "m**2"}
wing["AR"] = {"value": 9.69}
wing["c4sweep"] = {"value": 1.0, "units": "deg"}
wing["taper"] = {"value": 0.625}
wing["toverc"] = {"value": 0.19}
geom["wing"] = wing

fuselage = dict()
fuselage["S_wet"] = {"value": 490, "units": "ft**2"}
fuselage["width"] = {"value": 1.7, "units": "m"}
fuselage["length"] = {"value": 12.67, "units": "m"}
fuselage["height"] = {"value": 1.73, "units": "m"}
geom["fuselage"] = fuselage

hstab = dict()
hstab["S_ref"] = {"value": 6.93, "units": "m**2"}
hstab["c4_to_wing_c4"] = {"value": 7.28, "units": "m"}
geom["hstab"] = hstab

vstab = dict()
vstab["S_ref"] = {"value": 3.34, "units": "m**2"}
geom["vstab"] = vstab

nosegear = dict()
nosegear["length"] = {"value": 0.9, "units": "m"}
geom["nosegear"] = nosegear

maingear = dict()
maingear["length"] = {"value": 0.92, "units": "m"}
geom["maingear"] = maingear

ac["geom"] = geom

# ==WEIGHTS========================
weights = dict()
weights["MTOW"] = {"value": 3970, "units": "kg"}
weights["W_fuel_max"] = {"value": 1018, "units": "kg"}
weights["MLW"] = {"value": 3358, "units": "kg"}

ac["weights"] = weights

# ==PROPULSION=====================
propulsion = dict()
engine = dict()
engine["rating"] = {"value": 675, "units": "hp"}
propulsion["engine"] = engine

propeller = dict()
propeller["diameter"] = {"value": 2.1, "units": "m"}
propulsion["propeller"] = propeller

ac["propulsion"] = propulsion

# Some additional parameters needed by the empirical weights tools
ac["num_passengers_max"] = {"value": 2}
ac["q_cruise"] = {"value": 56.9621, "units": "lb*ft**-2"}
data["ac"] = ac
