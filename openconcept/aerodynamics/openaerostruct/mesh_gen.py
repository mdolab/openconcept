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


class SectionPlanformMesh(om.ExplicitComponent):
    """
    Generate an OpenAeroStruct mesh based on defined section parameters, similar
    to how AVL accepts planforms. The resulting mesh is for a half wing (meant
    to use with OpenAeroStruct symmetry), but the input reference area is for the
    full wing.

    The sectional properties (x offset, y value, and chord) are nondimensional,
    but their relative magnitudes reflect the actual shape. The full planform
    is scaled to match the requested planform area.

    The mesh points are cosine spaced in the chordwise direction and in the
    spanwise direction within each trapezoidal region of the wing defined by
    pairs of section properties.

                                       ----> +y
                                      |
                                      |
                                      v +x

      _     Section 1       Section 2     _
     |                  |          _-'     |
     |                  |       _-'        |
     |                  |    _-'           |
     |                    _-'            chord_sec
    x_LE_sec           _-'                 |
     |              _-'                    |
     |           _-'   _--------------    _|
     |        _-'   _-'
     |_    _-'   _-'    |--- y_sec ---|
          |   _-'
          |_-'

    Inputs
    ------
    S: float
        full planform area (scalar, m^2)
    x_LE_sec : float
        Streamwise offset of the section's leading edge, starting with the outboard
        section (wing tip) and moving inboard toward the root (vector of length
        num_sections, dimensionless)
    y_sec : float
        Spanwise location of each section, starting with the outboard section (wing
        tip) at the MOST NEGATIVE y value and moving inboard (increasing y value)
        toward the root; the user does not provide a value for the root because it
        is always 0.0 (vector of length num_sections - 1, dimensionless)
    chord_sec : float
        Chord of each section, starting with the outboard section (wing tip) and
        moving inboard toward the root (vector of length num_sections, dimensionless)

    Outputs
    -------
    mesh: ndarray
        OpenAeroStruct 3D mesh (num_x x sum(num_y) x 3 ndarray, m)

    Options
    -------
    num_x: int
        Number of points in x (streamwise) direction (scalar, dimensionless)
    num_y: int or iterable of ints
        Number of spanwise points in the trapezoidal regions between each pair of
        adjacent sections; can be specified either as a single integer, where that
        same value is used for each region, or as an iterable of integers of length
        num_sections to enable different numbers of spanwise coordinates for each
        region (scalar or vector, dimensionless)
        number of points in y (spanwise) direction (scalar, dimensionless)
    num_sections : int
        Number of spanwise sections to define planform shape (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_x", default=3, types=int, desc="Number of streamwise mesh points")
        self.options.declare(
            "num_y",
            default=7,
            types=(int, list, tuple, np.ndarray),
            desc="Number of spanwise mesh points per trapezoidal region",
        )
        self.options.declare(
            "num_sections", default=2, types=int, desc="Number of sections along the half span to define"
        )

    def setup(self):
        # Process mesh resolution options
        self.nx = self.options["num_x"]
        self.n_sec = self.options["num_sections"]
        if self.n_sec < 2:
            raise ValueError("Must define at least two sections along the span")
        num_y = self.options["num_y"]
        if isinstance(num_y, int):
            # If it's an int, duplicate the number of sections for each region
            self.ny = [
                num_y,
            ] * (self.n_sec - 1)
        elif isinstance(num_y, (list, tuple, np.ndarray)):
            # If it is an iterable, make sure it's the right length
            if len(num_y) != self.n_sec - 1:
                raise ValueError("If specified as an iterable, the num_y option must have a length of num_sections")
            self.ny = [int(x) for x in num_y]  # cast anything that's not an integer to an integer

        self.add_input("S", units="m**2")
        self.add_input("x_LE_sec", shape=(self.n_sec,))
        self.add_input("y_sec", shape=(self.n_sec - 1,))
        self.add_input("chord_sec", shape=(self.n_sec,))

        # Generate default mesh
        ny_tot = np.sum(self.ny)
        x, y = np.meshgrid(np.linspace(0, 1, self.nx), np.linspace(-1, 0, ny_tot), indexing="ij")
        y *= 5
        mesh = np.zeros((self.nx, ny_tot, 3))
        mesh[:, :, 0] = x
        mesh[:, :, 1] = y

        self.add_output("mesh", val=mesh, shape=(self.nx, ny_tot, 3), units="m")

        self.declare_partials("mesh", "*")

    def compute(self, inputs, outputs):
        S = inputs["S"].item()
        x_sec = inputs["x_LE_sec"]
        y_sec = np.hstack((inputs["y_sec"], 0))
        c_sec = inputs["chord_sec"]

        # Iterate through the defined trapezoidal regions between sections
        y_prev = 0
        A = 0.0  # nondimensional area of existing mesh
        for i_sec in range(self.n_sec - 1):
            ny = self.ny[i_sec]
            x_mesh, y_mesh = np.meshgrid(
                cos_space(0, 1, self.nx, dtype=x_sec.dtype), cos_space(y_sec[i_sec], y_sec[i_sec + 1], ny, dtype=x_sec.dtype), indexing="ij"
            )
            x_mesh *= cos_space(c_sec[i_sec], c_sec[i_sec + 1], ny)
            x_mesh += cos_space(x_sec[i_sec], x_sec[i_sec + 1], ny)

            outputs["mesh"][:, y_prev : y_prev + ny, 0] = x_mesh
            outputs["mesh"][:, y_prev : y_prev + ny, 1] = y_mesh

            # Add the area of this trapezoidal region to the total nondimensional planform area
            A += (y_sec[i_sec + 1] - y_sec[i_sec]) * (c_sec[i_sec] + c_sec[i_sec + 1]) * 0.5

            y_prev += ny

        # Zero any z coordinates
        outputs["mesh"][:, :, 2] = 0.0

        # Scale the mesh by the reference area
        A *= 2  # we're only doing a half wing, so double to get total area
        outputs["mesh"] *= (S / A) ** 0.5


def cos_space(start, end, num, dtype=float):
    """
    Cosine spacing between start and end with num points.
    """
    return start + 0.5 * (end - start) * (1 - np.cos(np.linspace(0, np.pi, num, dtype=dtype)))


def cos_space_deriv_start(num, dtype=float):
    """
    Derivative of cosine spacing output w.r.t. the start value.
    """
    return 1 - 0.5 * (1 - np.cos(np.linspace(0, np.pi, num, dtype=dtype)))


def cos_space_deriv_end(num, dtype=float):
    """
    Derivative of cosine spacing output w.r.t. the end value.
    """
    return 0.5 * (1 - np.cos(np.linspace(0, np.pi, num, dtype=dtype)))
