# DATA FOR TBM T80
# Collected from various sources
# including SOCATA pilot manual

data = dict()
ac = dict()
# ==AERO==================================
aero = dict()
aero["CLmax_TO"] = {"value": 1.7}

polar = dict()
polar["e"] = {"value": 0.78}
polar["CD0_TO"] = {"value": 0.03}
polar["CD0_cruise"] = {"value": 0.0205}

aero["polar"] = polar
ac["aero"] = aero

# ==GEOMETRY==============================
geom = dict()
wing = dict()
wing["S_ref"] = {"value": 18.0, "units": "m**2"}
wing["AR"] = {"value": 8.95}
wing["c4sweep"] = {"value": 1.0, "units": "deg"}
wing["taper"] = {"value": 0.622}
wing["toverc"] = {"value": 0.16}
geom["wing"] = wing

fuselage = dict()
fuselage["S_wet"] = {"value": 392, "units": "ft**2"}
fuselage["width"] = {"value": 4.58, "units": "ft"}
fuselage["length"] = {"value": 27.39, "units": "ft"}
fuselage["height"] = {"value": 5.555, "units": "ft"}
geom["fuselage"] = fuselage

hstab = dict()
hstab["S_ref"] = {"value": 47.5, "units": "ft**2"}
hstab["c4_to_wing_c4"] = {"value": 17.9, "units": "ft"}
geom["hstab"] = hstab

vstab = dict()
vstab["S_ref"] = {"value": 31.36, "units": "ft**2"}
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
weights["MTOW"] = {"value": 3353, "units": "kg"}
weights["W_fuel_max"] = {"value": 2000, "units": "lb"}
weights["MLW"] = {"value": 7000, "units": "lb"}

ac["weights"] = weights

# ==PROPULSION=====================
propulsion = dict()
engine = dict()
engine["rating"] = {"value": 850, "units": "hp"}
propulsion["engine"] = engine

propeller = dict()
propeller["diameter"] = {"value": 2.31, "units": "m"}
propulsion["propeller"] = propeller

ac["propulsion"] = propulsion

# Some additional parameters needed by the empirical weights tools
ac["num_passengers_max"] = {"value": 6}
ac["q_cruise"] = {"value": 135.4, "units": "lb*ft**-2"}
data["ac"] = ac
