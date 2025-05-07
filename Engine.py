import json
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np


class Constant:
    gravity = 0
    air_density = 0
    air_viscosity = 0
    Re_max = 0
    perfect_gaz_cst = 0
    gamma = 0
    pi = 0
    sea_pressure = 0
    sea_temp = 0
    temp_gradient = 0
    air_molar_mass = 0
    drag_coefficient = 0

    def __init__(self, constants):
        Constant.gravity = constants["g"]
        Constant.air_density = constants["air-density"]
        Constant.air_viscosity = constants["air-viscosity"]
        Constant.Re_max = constants["Re-max"]
        Constant.perfect_gaz_cst = constants["perfect-gaz-cst"]
        Constant.gamma = constants["thermiques-capacity-air"]
        Constant.pi = constants["pi"]
        Constant.sea_pressure = constants["sea-pressure"]
        Constant.sea_temp = constants["sea-temperature"]
        Constant.temp_gradient = constants["temp-gradient"]
        Constant.air_molar_mass = constants["air-molar-mass"]
        Constant.drag_coefficient = constants["drag-coefficient"]

        print("Constant: Constants init !")


class Fuel:
    def __init__(self, fuel_params):
        self.mass = fuel_params["given"]["mass"]  # kg
        self.exposed_surface_combustion = fuel_params["given"]["exposed-surface-combustion"]  # m^2
        self.chamber_volume = fuel_params["given"]["chamber-volume"]  # m^3

        default_param = fuel_params["default"]
        self.density = default_param["density"]  # kg/m^3
        self.linear_combustion_speed = default_param["linear-combustion-speed"]  # m/s
        self.gaz_heat = default_param["gaz-heat"]  # K
        self.gaz_molar_mass = default_param["gaz-molar-mass"]  # kg/mol

        # Calculated properties
        self.volume = self.mass / self.density  # m^3


def solve(f, x=1, p=1e-3):
    while f(x) < 0:
        x += p
    return x


class Rocket:
    def __init__(self, name, rocket_params, fuel_params):
        self.name = name
        self.mass = rocket_params["mass"]
        self.length = rocket_params["length"]
        self.diameter_nose = rocket_params["diameter-nose"]
        self.smallest_nozzle_diameter = rocket_params["smallest-nozzle-diameter"]  # At
        self.opening_nozzle_diameter = rocket_params["opening-nozzle-diameter"]  # Ae

        self.Ae = Constant.pi * (self.opening_nozzle_diameter / 2) ** 2
        self.At = Constant.pi * (self.smallest_nozzle_diameter / 2) ** 2
        Ae_At = self.Ae / self.At

        def mach_equation(Me):
            return (1 / Me) * ((2 / (Constant.gamma + 1)) * (1 + ((Constant.gamma - 1) / 2) * Me ** 2)) ** (
                    (Constant.gamma + 1) / (2 * (Constant.gamma - 1))) - Ae_At

        # Trouver la solution pour Me > 1 (écoulement supersonique)
        print(f"Rocket: Starting solving mach equation...")
        Me_solution = solve(mach_equation, x=1,
                            p=1 / FireRocketry.PRECISION)  # x0=2.0 est une estimation initiale
        print(f"Rocket: Finished. Me={Me_solution} !")
        self.Me = Me_solution

        # Fuel
        self.fuel = Fuel(fuel_params)


class FireRocketry:
    PRECISION = 10

    def __init__(self, params_file="./params/settings.json", rocket_name="RocketTest", precision=10):
        self.frame_in_sec = 1 / precision
        FireRocketry.PRECISION = precision
        self.params_file = params_file
        self.rocket_name = rocket_name
        self.init_constants()
        self.rocket = self.load_rocket(self.rocket_name)
        # Calcule du débit de masse (dot_m):
        self.dot_m = self.rocket.fuel.density * self.rocket.fuel.exposed_surface_combustion * self.rocket.fuel.linear_combustion_speed  # debit de masse kg/s

    def init_constants(self):
        with open(self.params_file, 'r') as file:
            constants = json.load(file)["constants"]
            Constant(constants)

    def load_rocket(self, name):
        with open(self.params_file, 'r') as file:
            j = json.load(file)
            rocket = j["rockets"][name]
            fuel = {"default": j["fuels"][rocket["fuel"]["name"]], "given": rocket["fuel"]}
        return Rocket(name, rocket, fuel)

    @staticmethod
    def calc_Patm(altitude: float):
        return Constant.sea_pressure * (1 - (Constant.temp_gradient * altitude) / Constant.sea_temp) ** (
                (Constant.gravity * Constant.air_molar_mass) / (Constant.perfect_gaz_cst * Constant.temp_gradient))

    def calculate_rocket_thrust(self, Patm: float, t, debug=False):
        if debug:
            print("Start calculating rocket thrust...")
        # Calcul de force de poussée : F(t) = dot_m * ve + (Pe(t) - Patm(h)) * Ae

        # Calcule de Pe/Pc
        Pe_Pc = (1 + ((Constant.gamma - 1) / 2) * self.rocket.Me ** 2) ** (-Constant.gamma / (Constant.gamma - 1))

        # Calcule de la vitesse d'éjection des gaz (ve):
        Ve = math.sqrt(
            (2 * Constant.gamma / (Constant.gamma - 1)) * (
                    Constant.perfect_gaz_cst / Constant.air_molar_mass) * self.rocket.fuel.gaz_heat * (
                    1 - Pe_Pc ** ((Constant.gamma - 1) / Constant.gamma)))  # m/s

        # Cherchons Pe(t) trouvons d'abord Pc(t)=(n(t)*R*Tc)/Vc(t)
        # Cherchons Vc(t) (Volume) et n(t)
        Vc_t = self.rocket.fuel.chamber_volume - (self.rocket.fuel.volume - (
                self.rocket.fuel.linear_combustion_speed * t * self.rocket.fuel.exposed_surface_combustion))  # m^3

        tau = Vc_t / (self.rocket.At * Ve)
        n_t = (self.dot_m / self.rocket.fuel.gaz_molar_mass) * tau  # mol
        Pc_t = (n_t * Constant.perfect_gaz_cst * self.rocket.fuel.gaz_heat) / Vc_t  # Pa

        Pe_t = Pe_Pc * Pc_t  # Pa

        # Calcule de F(t)
        F_t = self.dot_m * Ve + (Pe_t - Patm) * self.rocket.Ae

        mass_t = self.rocket.fuel.mass - self.dot_m * t
        if mass_t <= 0:
            F_t = 0
        return F_t, mass_t

    def calculate_drag(self, v):
        return (1 / 2) * Constant.drag_coefficient * Constant.air_density * v ** 2 * (
                Constant.pi * (self.rocket.diameter_nose / 2) ** 3)

    def get_remaining_fuel_mass(self, t):
        m0 = self.fuel.mass
        return max(0, m0 - self.dot_m * t)

    def simulation(self, yi=0, vi=0):
        y = [yi]  # altitude
        v = [vi]  # vitesse
        a = [0]  # acceleration
        dt = 0
        t_lst = [0]
        Fthr = [0]
        # init plot
        figure, axis = plt.subplots(2, 2)

        have_warn_no_fuel_left = False
        print(
            "Cette simulation repose sur des approximations peu réel ! On suppose que les gaz sont parfaits, que la fusée se dirige uniquement verticalement, que le nez est un cône, etc.")
        print("Starting simulating... Please wait !")

        i = 0
        while y[i] >= 0:
            Fthr_t, mass_fuel = self.calculate_rocket_thrust(self.calc_Patm(y[i]), dt)
            Fdrag_t = self.calculate_drag(v[i])
            mass = self.rocket.mass - self.rocket.fuel.mass + mass_fuel
            P = mass * Constant.gravity

            if mass_fuel <= 0 and not have_warn_no_fuel_left:
                print(f"NO FUEL LEFT AT {dt}SEC !")
                have_warn_no_fuel_left = True

            # Calculons Ftotal(t)
            Ftotal_t = Fthr_t - Fdrag_t - P

            # Représentons sur l'axe y l'accélération a
            ay = Ftotal_t / mass

            # Comme d_v/d_t=a alors calculons primitive de a
            vy = v[i] + ay * dt

            # Comme d_y/d_t=v alors calculons primitive de v
            Oy = y[i] + v[i] * dt + 0.5 * ay * dt ** 2

            if v[i] >= 343:
                warnings.warn(
                    "The rocket exceed Mach 1 causing model to predict bad result ! Fix the simulator or change settings !")

            t_lst.append(dt)
            dt += self.frame_in_sec
            i += 1

            y.append(Oy)
            v.append(vy)
            a.append(ay)
            Fthr.append(Fthr_t)

        print(f"Simulation finished at t={dt}sec i={i} !")
        print(mass_fuel, mass)
        # Show plot !
        # Acceleration
        t_lst = np.array(t_lst) * FireRocketry.PRECISION
        axis[0, 0].plot(t_lst, a)
        axis[0, 0].set_title("Acceleration")

        # Speed
        axis[0, 1].plot(t_lst, v)
        axis[0, 1].set_title("Speed")

        # Position
        axis[1, 0].plot(t_lst, y)
        axis[1, 0].set_title("Position")

        # Thrust
        axis[1, 1].plot(t_lst, Fthr)
        axis[1, 1].set_title("Thrust")

        plt.show()


fr = FireRocketry(precision=100)
# print(fr.calculate_rocket_thrust(fr.calc_Patm(0), 1))
# print(fr.calculate_drag(1))
fr.simulation()
