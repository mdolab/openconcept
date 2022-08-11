# DATA FOR TBM T80
# Collected from various sources
# including SOCATA pilot manual

data = dict()
ac = dict()
# ==AERO==================================
aero = dict()
aero["CLmax_TO"] = {"value": 2.0}

polar = dict()
polar["e"] = {"value": 0.801}
polar["CD0_TO"] = {"value": 0.03}
polar["CD0_cruise"] = {"value": 0.01925}

aero["polar"] = polar
ac["aero"] = aero

# ==GEOMETRY==============================
geom = dict()
wing = dict()
wing["S_ref"] = {"value": 124.6, "units": "m**2"}
wing["AR"] = {"value": 9.45}
wing["c4sweep"] = {"value": 25.0, "units": "deg"}
wing["taper"] = {"value": 0.159}
wing["toverc"] = {"value": 0.12}
geom["wing"] = wing

hstab = dict()
hstab["S_ref"] = {"value": 32.78, "units": "m**2"}
hstab["c4_to_wing_c4"] = {"value": 17.9, "units": "m"}
geom["hstab"] = hstab

vstab = dict()
vstab["S_ref"] = {"value": 26.44, "units": "m**2"}
geom["vstab"] = vstab

nosegear = dict()
nosegear["length"] = {"value": 3, "units": "ft"}
geom["nosegear"] = nosegear

maingear = dict()
maingear["length"] = {"value": 4, "units": "ft"}
geom["maingear"] = maingear

ac["geom"] = geom

# ==WEIGHTS========================
weights = dict()
weights["MTOW"] = {"value": 79002, "units": "kg"}
weights["OEW"] = {"value": 0.530 * 79002, "units": "kg"}
weights["W_fuel_max"] = {"value": 0.266 * 79002, "units": "kg"}
weights["MLW"] = {"value": 66349, "units": "kg"}

ac["weights"] = weights

# ==PROPULSION=====================
propulsion = dict()
engine = dict()
engine["rating"] = {"value": 27000, "units": "lbf"}
propulsion["engine"] = engine

ac["propulsion"] = propulsion

# Some additional parameters needed by the empirical weights tools
ac["num_passengers_max"] = {"value": 180}
ac["q_cruise"] = {"value": 212.662, "units": "lb*ft**-2"}
data["ac"] = ac
