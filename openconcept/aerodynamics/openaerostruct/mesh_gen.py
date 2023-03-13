import numpy as np
import openmdao.api as om


class TrapezoidalPlanformMesh(om.ExplicitComponent):
    """
    Generate an OpenAeroStruct mesh based on basic wing design parameters.
    Resulting mesh is for a half wing (meant to use with OpenAeroStruct symmetry),
    but the input reference area is for the full wing.

    Inputs
    ------
    S: float
        full planform area (scalar, m^2)
    AR: float
        aspect ratio (scalar, dimensionless)
    taper: float
        taper ratio (must be >0 and <=1); tip chord / root chord (scalar, dimensionless)
    sweep: float
        quarter chord sweep angle (scalar, degrees)

    Outputs
    -------
    mesh: ndarray
        OpenAeroStruct 3D mesh (num_x x num_y x 3 ndarray, m)

    Options
    -------
    num_x: int
        number of points in x (streamwise) direction (scalar, dimensionless)
    num_y: int
        number of points in y (spanwise) direction (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_x", default=3, desc="Number of streamwise mesh points")
        self.options.declare("num_y", default=7, desc="Number of spanwise (half wing) mesh points")

    def setup(self):
        nx = int(self.options["num_x"])
        ny = int(self.options["num_y"])

        # Generate default mesh
        x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(-1, 0, ny), indexing="ij")
        y *= 5
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = x
        mesh[:, :, 1] = y

        self.add_input("S", val=10, units="m**2")
        self.add_input("AR", val=10)
        self.add_input("taper", val=1.0)
        self.add_input("sweep", val=10, units="deg")

        self.add_output("mesh", val=mesh, shape=(nx, ny, 3), units="m")

        self.declare_partials("mesh", "*")

    def compute(self, inputs, outputs):
        S = inputs["S"]
        AR = inputs["AR"]
        taper = inputs["taper"]
        sweep = inputs["sweep"]
        nx = int(self.options["num_x"])
        ny = int(self.options["num_y"])

        # Compute absolute dimensions from wing geometry spec
        half_span = np.sqrt(AR * S) / 2
        c_root = S / (half_span * (1 + taper))

        # Create baseline square mesh from 0 to 1 in each direction
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, 1, nx), np.linspace(-1, 0, ny), indexing="ij")

        # Morph the mesh to fit the desired wing shape
        x_mesh *= c_root
        y_mesh *= half_span  # rectangular wing with correct root chord and wingspan
        x_mesh *= np.linspace(taper, 1, ny).reshape(1, ny)  # taper wing
        x_mesh -= np.linspace(c_root * taper, c_root, ny).reshape(1, ny) / 4  # shift to quarter chord at x=0
        x_mesh += np.linspace(half_span, 0, ny).reshape(1, ny) * np.tan(
            np.deg2rad(sweep)
        )  # sweep wing at quarter chord

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = x_mesh
        mesh[:, :, 1] = y_mesh

        outputs["mesh"] = mesh

    def compute_partials(self, inputs, J):
        S = inputs["S"]
        AR = inputs["AR"]
        taper = inputs["taper"]
        sweep = inputs["sweep"]
        nx = int(self.options["num_x"])
        ny = int(self.options["num_y"])

        # Compute absolute dimensions from wing geometry spec
        half_span = np.sqrt(AR * S) / 2
        c_root = S / (half_span * (1 + taper))

        # Create baseline square mesh from 0 to 1 in each direction
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, 1, nx), np.linspace(-1, 0, ny), indexing="ij")

        # Compute derivatives in a way analogous to forward AD
        db_dS = AR / (4 * np.sqrt(AR * S))
        db_dAR = S / (4 * np.sqrt(AR * S))
        dcroot_dS = 1 / (half_span * (1 + taper)) - S / (half_span**2 * (1 + taper)) * db_dS
        dcroot_dAR = -S / (half_span**2 * (1 + taper)) * db_dAR
        dcroot_dtaper = -S / (half_span * (1 + taper) ** 2)

        dy_dS = y_mesh * db_dS
        dy_dAR = y_mesh * db_dAR

        dx_dS = x_mesh * np.linspace(taper, 1, ny).reshape(1, ny) * dcroot_dS
        dx_dS -= np.linspace(dcroot_dS * taper, dcroot_dS, ny).reshape(1, ny) / 4
        dx_dS += np.linspace(db_dS, 0, ny).reshape(1, ny) * np.tan(np.deg2rad(sweep))

        dx_dAR = x_mesh * np.linspace(taper, 1, ny).reshape(1, ny) * dcroot_dAR
        dx_dAR -= np.linspace(dcroot_dAR * taper, dcroot_dAR, ny).reshape(1, ny) / 4
        dx_dAR += np.linspace(db_dAR, 0, ny).reshape(1, ny) * np.tan(np.deg2rad(sweep))

        dx_dtaper = (
            x_mesh * c_root * np.linspace(1, 0, ny).reshape(1, ny)
            + x_mesh * np.linspace(taper, 1, ny).reshape(1, ny) * dcroot_dtaper
        )
        dx_dtaper -= (
            np.linspace(c_root, 0, ny).reshape(1, ny) / 4
            + np.linspace(dcroot_dtaper * taper, dcroot_dtaper, ny).reshape(1, ny) / 4
        )

        dx_dsweep = (
            0 * x_mesh + np.linspace(half_span, 0, ny).reshape(1, ny) / np.cos(np.deg2rad(sweep)) ** 2 * np.pi / 180.0
        )

        J["mesh", "S"] = np.dstack((dx_dS, dy_dS, np.zeros((nx, ny)))).flatten()
        J["mesh", "AR"] = np.dstack((dx_dAR, dy_dAR, np.zeros((nx, ny)))).flatten()
        J["mesh", "taper"] = np.dstack((dx_dtaper, np.zeros((nx, ny)), np.zeros((nx, ny)))).flatten()
        J["mesh", "sweep"] = np.dstack((dx_dsweep, np.zeros((nx, ny)), np.zeros((nx, ny)))).flatten()
