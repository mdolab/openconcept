# Data for PC-12 NGX Hydrogen, a hydrogen fuel cell powered version of the PC-12 NGX.
# From Pilot's Information Manual PC-12 NGX and the Master Thesis MSE Aviation:
# - Feasibility of fuel cell propulsion for small commuter aircraft

data = dict()
ac = dict()
# ==AERO==================================
aero = dict()
aero["CLmax_TO"] = {"value": 1.52}

polar = dict()
polar["e"] = {"value": 0.7632}
polar["CD0_TO"] = {"value": 0.0247}
polar["CD0_cruise"] = {"value": 0.0274}
aero["polar"] = polar
ac["aero"] = aero

# ==GEOMETRY==============================
geom = dict() 
wing = dict()
wing["S_ref"] = {"value": 25.8, "units": "m**2"}
wing["AR"] = {"value": 10.27}
wing["c4sweep"] = {"value": 1.0, "units": "deg"}
wing["taper"] = {"value": 0.5}
wing["toverc"] = {"value": 0.14}
geom["wing"] = wing

fuselage = dict()
fuselage["S_wet"] = {"value": 60, "units": "m**2"}
fuselage["width"] = {"value": 1.69, "units": "m"}
fuselage["length"] = {"value": 14.01, "units": "m"}
fuselage["height"] = {"value": 1.83, "units": "m"}
geom["fuselage"] = fuselage

hstab = dict()
hstab["S_ref"] = {"value": 5.8, "units": "m**2"}
hstab["c4_to_wing_c4"] = {"value": 5.33, "units": "m"}
geom["hstab"] = hstab

vstab = dict()
vstab["S_ref"] = {"value": 3.7, "units": "m**2"}
geom["vstab"] = vstab

nosegear = dict()
nosegear["length"] = {"value": 1, "units": "m"}
geom["nosegear"] = nosegear

maingear = dict()
maingear["length"] = {"value": 0.9, "units": "m"}
geom["maingear"] = maingear

ac["geom"] = geom

# ==WEIGHTS========================
weights = dict()
weights["MTOW"] = {"value": 4740, "units": "kg"}
weights["W_fuel_max"] = {"value": 500, "units": "kg"} # Es wird LH2 als Treibstoff verwendet
weights["MLW"] = {"value": 4500, "units": "kg"}
weights["W_LH2_tank"] = {"value": 264.08, "units": "kg"}

ac["weights"] = weights

# ==PROPULSION=====================
propulsion = dict()
engine = dict()
engine["rating"] = {"value": 1200, "units": "hp"}
propulsion["engine"] = engine

propeller = dict()
propeller["diameter"] = {"value": 2.67, "units": "m"}
propulsion["propeller"] = propeller

motor = dict()
motor["rating"] = {"value": 800, "units": "hp"}
propulsion["motor"] = motor

thermal = dict()
hx = dict()
hx["n_wide_cold"] = {"value": 650, "units": None}
hx["n_long_cold"] = {"value": 3, "units": None}
hx["n_tall"] = {"value": 24, "units": None}
hx["channel_width_cold"] = {"value": 1.2, "units": "mm"}
hx["channel_height_cold"] = {"value": 12.0, "units": "mm"}
hx["fin_length_cold"] = {"value": 6.0, "units": "mm"}
hx["channel_width_hot"] = {"value": 1.0, "units": "mm"}
hx["channel_height_hot"] = {"value": 1.0, "units": "mm"}
hx["fin_length_hot"] = {"value": 6.0, "units": "mm"}
hx["fin_thickness"] = {"value": 0.15, "units": "mm"}
hx["plate_thickness"] = {"value": 0.25, "units": "mm"}
hx["case_thickness"] = {"value": 2.0, "units": "mm"}
thermal["hx"] = hx
propulsion["thermal"] = thermal

generator = dict()
generator["rating"] = {"value": 1083.7, "units": "hp"}
propulsion["generator"] = generator

ac["propulsion"] = propulsion

# Some additional parameters needed by the empirical weights tools
ac["num_passengers_max"] = {"value": 9}
ac["q_cruise"] = {"value": 73.5, "units": "lb*ft**-2"}
ac["num_engines"] = {"value": 1}
data["ac"] = ac
