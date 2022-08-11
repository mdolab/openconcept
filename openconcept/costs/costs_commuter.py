"""
This file contains a module to calculate operating costs for airplanes in the general turboprop / Part 23 range. It does not consider crew
costs, so mission optimization routines will tend to favor slower, longer-time routes than they  probably should (could fix in future).
"""

from openmdao.api import ExplicitComponent


class TurbopropOperatingCost(ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "n_components",
            default=1,
            desc='Number of propulsion components, e.g. engines, motors, generators. Inputs will be numbered "component_1" thru n ',
        )
        self.options.declare(
            "n_batteries",
            default=None,
            desc='Number of batteries. These should NOT be counted as components as they are not to be subtracted from OEW. Numbered "battery_1" through n',
        )
        self.options.declare(
            "airframe_cost_per_kg",
            default=266.0,
            desc="Cost of each kg of airplane empty weight MINUS ENGINES/MOTORS/GEN/BATTERIES",
        )
        self.options.declare(
            "fuel_cost_per_kg", default=2.5 / 3.08, desc="Cost of each KG of fuel; divide USD/gal by 3.08kg/gal"
        )
        self.options.declare(
            "electricity_cost_per_MJ", default=0.01, desc="Electricity cost per MJ. Divide MWh cost by 3600 MJ/MWh"
        )
        self.options.declare("aircraft_life_years", default=15, desc="Useful aircraft life in years")
        self.options.declare("aircraft_daily_cycles", default=5, desc="Number of mission cycles per day (average).")
        self.options.declare(
            "battery_replacement_cycles", default=1500, desc="Number of battery cycles until replacement"
        )
        self.options.declare(
            "OEM_premium",
            default=1.1,
            desc="Multiplier on the airframe cost as profit margin for the OEM. 10% margin = 1.1",
        )

    def setup(self):
        n_components = self.options["n_components"]
        n_batteries = self.options["n_batteries"]
        battery_replacement_cycles = self.options["battery_replacement_cycles"]
        af_cost_kg = self.options["airframe_cost_per_kg"]
        aircraft_daily_cycles = self.options["aircraft_daily_cycles"]
        aircraft_life_years = self.options["aircraft_life_years"]
        fuel_cost_per_kg = self.options["fuel_cost_per_kg"]
        electricity_cost_per_MJ = self.options["electricity_cost_per_MJ"]
        OEM_premium = self.options["OEM_premium"]

        for i in range(n_components):
            self.add_input("component_" + str(i + 1) + "_weight", units="kg", desc="Component weight")
            self.add_input("component_" + str(i + 1) + "_NR_cost", units="USD", desc="Component cost")
            self.declare_partials(
                ["airframe_NR_cost", "total_NR_cost"],
                ["component_" + str(i + 1) + "_weight"],
                val=-af_cost_kg * OEM_premium,
            )
            self.declare_partials(
                ["trip_direct_operating_cost"],
                ["component_" + str(i + 1) + "_weight"],
                val=-af_cost_kg * OEM_premium / aircraft_daily_cycles / 365 / aircraft_life_years,
            )
            self.declare_partials(["total_NR_cost"], ["component_" + str(i + 1) + "_NR_cost"], val=OEM_premium)
            self.declare_partials(
                ["trip_direct_operating_cost"],
                ["component_" + str(i + 1) + "_NR_cost"],
                val=OEM_premium / aircraft_daily_cycles / 365 / aircraft_life_years,
            )

        if n_batteries is not None:
            self.add_output("trip_battery_cost", units="USD")
            self.add_output("electricity_cost", units="USD")
            for i in range(n_batteries):
                self.add_input("battery_" + str(i + 1) + "_NR_cost", units="USD", desc="Battery purchase cost")
                self.add_input(
                    "battery_" + str(i + 1) + "_energy_used", units="MJ", desc="Battery energy used for mission"
                )
                self.declare_partials(
                    ["electricity_cost", "trip_energy_cost", "trip_direct_operating_cost"],
                    ["battery_" + str(i + 1) + "_energy_used"],
                    val=electricity_cost_per_MJ,
                )
                self.declare_partials(
                    ["trip_battery_cost", "trip_direct_operating_cost"],
                    ["battery_" + str(i + 1) + "_NR_cost"],
                    val=1 / battery_replacement_cycles,
                )

        self.add_input("fuel_burn", units="kg", desc="Fuel used for mission")
        self.add_input("OEW", units="kg", desc="Operating empty weight")

        self.add_output("fuel_cost", units="USD")
        self.add_output("airframe_NR_cost", units="USD")
        self.add_output("total_NR_cost", units="USD")
        self.add_output("trip_energy_cost", units="USD")
        self.add_output("trip_direct_operating_cost", units="USD")

        self.declare_partials(["airframe_NR_cost", "total_NR_cost"], ["OEW"], val=af_cost_kg * OEM_premium)
        self.declare_partials(
            ["trip_direct_operating_cost"],
            ["OEW"],
            val=af_cost_kg * OEM_premium / aircraft_daily_cycles / 365 / aircraft_life_years,
        )
        self.declare_partials(
            ["fuel_cost", "trip_energy_cost", "trip_direct_operating_cost"], ["fuel_burn"], val=fuel_cost_per_kg
        )

    def compute(self, inputs, outputs):
        # compute empty weight less the number of propulsion components
        # OEW - w_component for n components
        n_components = self.options["n_components"]
        n_batteries = self.options["n_batteries"]
        battery_replacement_cycles = self.options["battery_replacement_cycles"]
        af_cost_kg = self.options["airframe_cost_per_kg"]
        aircraft_daily_cycles = self.options["aircraft_daily_cycles"]
        aircraft_life_years = self.options["aircraft_life_years"]
        fuel_cost_per_kg = self.options["fuel_cost_per_kg"]
        electricity_cost_per_MJ = self.options["electricity_cost_per_MJ"]
        OEM_premium = self.options["OEM_premium"]

        adj_OEW = inputs["OEW"]
        components_NR_cost = 0

        for i in range(n_components):
            adj_OEW = adj_OEW - inputs["component_" + str(i + 1) + "_weight"]
            components_NR_cost = components_NR_cost + inputs["component_" + str(i + 1) + "_NR_cost"]

        outputs["airframe_NR_cost"] = adj_OEW * af_cost_kg * OEM_premium
        outputs["total_NR_cost"] = outputs["airframe_NR_cost"] + components_NR_cost * OEM_premium

        NR_contrib_trip_cost = outputs["total_NR_cost"] / aircraft_daily_cycles / 365 / aircraft_life_years

        outputs["fuel_cost"] = inputs["fuel_burn"] * fuel_cost_per_kg

        total_elec = 0
        total_batt_cost = 0
        if n_batteries is not None:
            for i in range(n_batteries):
                total_elec = total_elec + inputs["battery_" + str(i + 1) + "_energy_used"]
                total_batt_cost = total_batt_cost + inputs["battery_" + str(i + 1) + "_NR_cost"]
            outputs["electricity_cost"] = total_elec * electricity_cost_per_MJ
            outputs["trip_battery_cost"] = total_batt_cost / battery_replacement_cycles
            outputs["trip_energy_cost"] = outputs["electricity_cost"] + outputs["fuel_cost"]
            outputs["trip_direct_operating_cost"] = (
                outputs["trip_energy_cost"] + NR_contrib_trip_cost + outputs["trip_battery_cost"]
            )
        else:
            outputs["trip_energy_cost"] = outputs["fuel_cost"]
            outputs["trip_direct_operating_cost"] = outputs["trip_energy_cost"] + NR_contrib_trip_cost

    # def compute_partials(self, inputs, J):
    #     n_components = self.options['n_components']
    #     n_batteries = self.options['n_batteries']
    #     battery_replacement_cycles = self.options['battery_replacement_cycles']
    #     af_cost_kg = self.options['airframe_cost_per_kg']
    #     aircraft_daily_cycles = self.options['aircraft_daily_cycles']
    #     aircraft_life_years = self.options['aircraft_life_years']
    #     fuel_cost_per_kg = self.options['fuel_cost_per_kg']
    #     electricity_cost_per_MJ = self.options['electricity_cost_per_MJ']
    #     OEM_premium = self.options['OEM_premium']

    #     J['airframe_NR_cost','OEW'] = af_cost_kg * OEM_premium
    #     J['total_NR_cost','OEW'] = J['airframe_NR_cost','OEW']
    #     J['trip_direct_operating_cost','OEW'] = J['total_NR_cost','OEW'] / aircraft_daily_cycles / 365 / aircraft_life_years

    #     for i in range(n_components):
    #         J['airframe_NR_cost','component_'+str(i+1)+'_weight'] = - af_cost_kg * OEM_premium
    #         J['total_NR_cost','component_'+str(i+1)+'_weight'] = J['airframe_NR_cost','component_'+str(i+1)+'_weight']
    #         J['total_NR_cost','component_'+str(i+1)+'_NR_cost'] = OEM_premium
    #         J['trip_direct_operating_cost','component_'+str(i+1)+'_weight'] = J['total_NR_cost','component_'+str(i+1)+'_weight'] / aircraft_daily_cycles / 365 / aircraft_life_years
    #         J['trip_direct_operating_cost','component_'+str(i+1)+'_NR_cost'] = J['total_NR_cost','component_'+str(i+1)+'_NR_cost'] / aircraft_daily_cycles / 365 / aircraft_life_years

    #     J['fuel_cost','fuel_burn'] = fuel_cost_per_kg
    #     J['trip_energy_cost','fuel_burn'] = fuel_cost_per_kg
    #     J['trip_direct_operating_cost','fuel_burn'] = fuel_cost_per_kg

    #     if n_batteries is not None:
    #         for i in range(n_batteries):
    #             J['electricity_cost','battery_'+str(i+1)+'_energy_used'] = electricity_cost_per_MJ
    #             J['trip_energy_cost','battery_'+str(i+1)+'_energy_used'] = electricity_cost_per_MJ
    #             J['trip_direct_operating_cost','battery_'+str(i+1)+'_energy_used'] = electricity_cost_per_MJ

    #             J['trip_battery_cost','battery_'+str(i+1)+'_NR_cost'] = 1 / battery_replacement_cycles
    #             J['trip_direct_operating_cost','battery_'+str(i+1)+'_NR_cost'] = 1 / battery_replacement_cycles
