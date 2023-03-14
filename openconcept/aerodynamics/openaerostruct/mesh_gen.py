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
        OpenAeroStruct 3D mesh (num_x + 1 x num_y + 1 x 3 ndarray, m)

    Options
    -------
    num_x: int
        number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y: int
        number of panels in y (spanwise) direction (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_x", default=2, desc="Number of streamwise mesh panels")
        self.options.declare("num_y", default=6, desc="Number of spanwise (half wing) mesh panels")

    def setup(self):
        # Number of coordinates is one more than the number of panels
        self.nx = int(self.options["num_x"]) + 1
        self.ny = int(self.options["num_y"]) + 1

        # Generate default mesh
        x, y = np.meshgrid(np.linspace(0, 1, self.nx), np.linspace(-1, 0, self.ny), indexing="ij")
        y *= 5
        mesh = np.zeros((self.nx, self.ny, 3))
        mesh[:, :, 0] = x
        mesh[:, :, 1] = y

        self.add_input("S", val=10, units="m**2")
        self.add_input("AR", val=10)
        self.add_input("taper", val=1.0)
        self.add_input("sweep", val=10, units="deg")

        self.add_output("mesh", val=mesh, shape=(self.nx, self.ny, 3), units="m")

        self.declare_partials("mesh", "*")

    def compute(self, inputs, outputs):
        S = inputs["S"]
        AR = inputs["AR"]
        taper = inputs["taper"]
        sweep = inputs["sweep"]

        # Compute absolute dimensions from wing geometry spec
        half_span = np.sqrt(AR * S) / 2
        c_root = S / (half_span * (1 + taper))

        # Create baseline square mesh from 0 to 1 in each direction
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, 1, self.nx), np.linspace(-1, 0, self.ny), indexing="ij")

        # Morph the mesh to fit the desired wing shape
        x_mesh *= c_root
        y_mesh *= half_span  # rectangular wing with correct root chord and wingspan
        x_mesh *= np.linspace(taper, 1, self.ny).reshape(1, self.ny)  # taper wing
        x_mesh -= np.linspace(c_root * taper, c_root, self.ny).reshape(1, self.ny) / 4  # shift to quarter chord at x=0
        x_mesh += np.linspace(half_span, 0, self.ny).reshape(1, self.ny) * np.tan(
            np.deg2rad(sweep)
        )  # sweep wing at quarter chord

        mesh = np.zeros((self.nx, self.ny, 3))
        mesh[:, :, 0] = x_mesh
        mesh[:, :, 1] = y_mesh

        outputs["mesh"] = mesh

    def compute_partials(self, inputs, J):
        S = inputs["S"]
        AR = inputs["AR"]
        taper = inputs["taper"]
        sweep = inputs["sweep"]

        # Compute absolute dimensions from wing geometry spec
        half_span = np.sqrt(AR * S) / 2
        c_root = S / (half_span * (1 + taper))

        # Create baseline square mesh from 0 to 1 in each direction
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, 1, self.nx), np.linspace(-1, 0, self.ny), indexing="ij")

        # Compute derivatives in a way analogous to forward AD
        db_dS = AR / (4 * np.sqrt(AR * S))
        db_dAR = S / (4 * np.sqrt(AR * S))
        dcroot_dS = 1 / (half_span * (1 + taper)) - S / (half_span**2 * (1 + taper)) * db_dS
        dcroot_dAR = -S / (half_span**2 * (1 + taper)) * db_dAR
        dcroot_dtaper = -S / (half_span * (1 + taper) ** 2)

        dy_dS = y_mesh * db_dS
        dy_dAR = y_mesh * db_dAR

        dx_dS = x_mesh * np.linspace(taper, 1, self.ny).reshape(1, self.ny) * dcroot_dS
        dx_dS -= np.linspace(dcroot_dS * taper, dcroot_dS, self.ny).reshape(1, self.ny) / 4
        dx_dS += np.linspace(db_dS, 0, self.ny).reshape(1, self.ny) * np.tan(np.deg2rad(sweep))

        dx_dAR = x_mesh * np.linspace(taper, 1, self.ny).reshape(1, self.ny) * dcroot_dAR
        dx_dAR -= np.linspace(dcroot_dAR * taper, dcroot_dAR, self.ny).reshape(1, self.ny) / 4
        dx_dAR += np.linspace(db_dAR, 0, self.ny).reshape(1, self.ny) * np.tan(np.deg2rad(sweep))

        dx_dtaper = (
            x_mesh * c_root * np.linspace(1, 0, self.ny).reshape(1, self.ny)
            + x_mesh * np.linspace(taper, 1, self.ny).reshape(1, self.ny) * dcroot_dtaper
        )
        dx_dtaper -= (
            np.linspace(c_root, 0, self.ny).reshape(1, self.ny) / 4
            + np.linspace(dcroot_dtaper * taper, dcroot_dtaper, self.ny).reshape(1, self.ny) / 4
        )

        dx_dsweep = (
            0 * x_mesh
            + np.linspace(half_span, 0, self.ny).reshape(1, self.ny) / np.cos(np.deg2rad(sweep)) ** 2 * np.pi / 180.0
        )

        J["mesh", "S"] = np.dstack((dx_dS, dy_dS, np.zeros((self.nx, self.ny)))).flatten()
        J["mesh", "AR"] = np.dstack((dx_dAR, dy_dAR, np.zeros((self.nx, self.ny)))).flatten()
        J["mesh", "taper"] = np.dstack(
            (dx_dtaper, np.zeros((self.nx, self.ny)), np.zeros((self.nx, self.ny)))
        ).flatten()
        J["mesh", "sweep"] = np.dstack(
            (dx_dsweep, np.zeros((self.nx, self.ny)), np.zeros((self.nx, self.ny)))
        ).flatten()


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
        OpenAeroStruct 3D mesh (num_x + 1 x sum(num_y) + 1 x 3 ndarray, m)

    Options
    -------
    num_x: int
        Number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y: int or iterable of ints
        Number of spanwise panels in the trapezoidal regions between each pair of
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
        self.nx = self.options["num_x"] + 1  # nx is now number of coordinates, not panels
        self.n_sec = self.options["num_sections"]
        if self.n_sec < 2:
            raise ValueError("Must define at least two sections along the half span")
        num_y = self.options["num_y"]
        if isinstance(num_y, int):
            # If it's an int, duplicate the number of sections for each region
            self.ny = [num_y] * (self.n_sec - 1)
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
        self.ny_tot = np.sum(self.ny) + 1
        x, y = np.meshgrid(np.linspace(0, 1, self.nx), np.linspace(-1, 0, self.ny_tot), indexing="ij")
        y *= 5
        mesh = np.zeros((self.nx, self.ny_tot, 3))
        mesh[:, :, 0] = x
        mesh[:, :, 1] = y

        self.add_output("mesh", val=mesh, shape=(self.nx, self.ny_tot, 3), units="m")

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
            ny = self.ny[i_sec] + 1  # number of coordinates in the current region (including at the ends)
            x_mesh, y_mesh = np.meshgrid(
                cos_space(0, 1, self.nx, dtype=x_sec.dtype),
                cos_space(y_sec[i_sec], y_sec[i_sec + 1], ny, dtype=x_sec.dtype),
                indexing="ij",
            )
            x_mesh *= cos_space(c_sec[i_sec], c_sec[i_sec + 1], ny)
            x_mesh += cos_space(x_sec[i_sec], x_sec[i_sec + 1], ny)

            # Because y_prev is incremented by ny - 1, rather than by ny, the furthest inboard
            # coordinates from this region will be "overwritten" (with the same values) by
            # the next region (not particularly important to know, but might be useful for debugging)
            outputs["mesh"][:, y_prev : y_prev + ny, 0] = x_mesh
            outputs["mesh"][:, y_prev : y_prev + ny, 1] = y_mesh

            # Add the area of this trapezoidal region to the total nondimensional planform area
            A += (y_sec[i_sec + 1] - y_sec[i_sec]) * (c_sec[i_sec] + c_sec[i_sec + 1]) * 0.5

            y_prev += ny - 1

        # Zero any z coordinates
        outputs["mesh"][:, :, 2] = 0.0

        # Scale the mesh by the reference area
        A *= 2  # we're only doing a half wing, so double to get total area
        outputs["mesh"] *= (S / A) ** 0.5

    def compute_partials(self, inputs, partials):
        S = inputs["S"].item()
        x_sec = inputs["x_LE_sec"]
        y_sec = np.hstack((inputs["y_sec"], 0))
        c_sec = inputs["chord_sec"]

        # Indices in flattened array
        idx_x = 3 * np.arange(self.nx * self.ny_tot)
        idx_y = idx_x + 1

        mesh = np.zeros((self.nx, self.ny_tot, 3))

        # Iterate through the defined trapezoidal regions between sections
        y_prev = 0
        A = 0.0  # nondimensional area of existing mesh
        dA_dysec = np.zeros(self.n_sec - 1, dtype=float)
        dA_dcsec = np.zeros(self.n_sec, dtype=float)
        for i_sec in range(self.n_sec - 1):
            ny = self.ny[i_sec] + 1  # number of coordinates in the current region (including at the ends)
            x_mesh, y_mesh = np.meshgrid(
                cos_space(0, 1, self.nx, dtype=x_sec.dtype),
                cos_space(y_sec[i_sec], y_sec[i_sec + 1], ny, dtype=x_sec.dtype),
                indexing="ij",
            )

            # Derivatives of this section
            dymesh_dysec = np.tile(cos_space_deriv_start(ny, dtype=x_sec.dtype), (self.nx, 1))
            dymesh_dysecnext = np.tile(cos_space_deriv_end(ny, dtype=x_sec.dtype), (self.nx, 1))
            dxmesh_dcsec = x_mesh * cos_space_deriv_start(ny)
            dxmesh_dcsecnext = x_mesh * cos_space_deriv_end(ny)
            dxmesh_dxsec = np.tile(cos_space_deriv_start(ny), (self.nx, 1))
            dxmesh_dxsecnext = np.tile(cos_space_deriv_end(ny), (self.nx, 1))

            x_mesh *= cos_space(c_sec[i_sec], c_sec[i_sec + 1], ny)
            x_mesh += cos_space(x_sec[i_sec], x_sec[i_sec + 1], ny)

            # Indices in the mesh (not x, y, z values) corresponding to the coordinates in the current region
            if i_sec == 0:
                idx_mesh = (np.tile(np.arange(y_prev, y_prev + ny), (self.nx, 1)).T + np.arange(self.nx) * self.ny_tot).T.flatten()
                partials["mesh", "y_sec"][idx_y[idx_mesh], i_sec] += dymesh_dysec.flatten()
                # No derivative w.r.t. the y value at the root because it's always zero
                if i_sec < self.n_sec - 2:
                    partials["mesh", "y_sec"][idx_y[idx_mesh], i_sec + 1] += dymesh_dysecnext.flatten()
                partials["mesh", "chord_sec"][idx_x[idx_mesh], i_sec] += dxmesh_dcsec.flatten()
                partials["mesh", "chord_sec"][idx_x[idx_mesh], i_sec + 1] += dxmesh_dcsecnext.flatten()
                partials["mesh", "x_LE_sec"][idx_x[idx_mesh], i_sec] += dxmesh_dxsec.flatten()
                partials["mesh", "x_LE_sec"][idx_x[idx_mesh], i_sec + 1] += dxmesh_dxsecnext.flatten()
            else:
                idx_mesh = (np.tile(np.arange(y_prev + 1, y_prev + ny), (self.nx, 1)).T + np.arange(self.nx) * self.ny_tot).T.flatten()
                partials["mesh", "y_sec"][idx_y[idx_mesh], i_sec] += dymesh_dysec[:, 1:].flatten()
                # No derivative w.r.t. the y value at the root because it's always zero
                if i_sec < self.n_sec - 2:
                    partials["mesh", "y_sec"][idx_y[idx_mesh], i_sec + 1] += dymesh_dysecnext[:, 1:].flatten()
                partials["mesh", "chord_sec"][idx_x[idx_mesh], i_sec] += dxmesh_dcsec[:, 1:].flatten()
                partials["mesh", "chord_sec"][idx_x[idx_mesh], i_sec + 1] += dxmesh_dcsecnext[:, 1:].flatten()
                partials["mesh", "x_LE_sec"][idx_x[idx_mesh], i_sec] += dxmesh_dxsec[:, 1:].flatten()
                partials["mesh", "x_LE_sec"][idx_x[idx_mesh], i_sec + 1] += dxmesh_dxsecnext[:, 1:].flatten()

            # Because y_prev is incremented by ny - 1, rather than by ny, the furthest inboard
            # coordinates from this region will be "overwritten" (with the same values) by
            # the next region (not particularly important to know, but might be useful for debugging)
            mesh[:, y_prev : y_prev + ny, 0] = x_mesh
            mesh[:, y_prev : y_prev + ny, 1] = y_mesh

            # Add the area of this trapezoidal region to the total nondimensional planform area
            A += (y_sec[i_sec + 1] - y_sec[i_sec]) * (c_sec[i_sec] + c_sec[i_sec + 1]) * 0.5
            dA_dysec[i_sec] += -0.5 * (c_sec[i_sec] + c_sec[i_sec + 1])
            if i_sec < self.n_sec - 2:
                dA_dysec[i_sec + 1] += 0.5 * (c_sec[i_sec] + c_sec[i_sec + 1])
            dA_dcsec[i_sec:i_sec + 2] += 0.5 * (y_sec[i_sec + 1] - y_sec[i_sec])

            y_prev += ny - 1

        # Scale the mesh by the reference area
        A *= 2  # we're only doing a half wing, so double to get total area
        dA_dysec *= 2
        dA_dcsec *= 2
        coeff = (S / A) ** 0.5
        for var in ["y_sec", "chord_sec", "x_LE_sec"]:
            partials["mesh", var] *= coeff
        partials["mesh", "y_sec"] += np.outer(-0.5 * mesh.flatten() * S**0.5 * A**(-1.5), dA_dysec)
        partials["mesh", "chord_sec"] += np.outer(-0.5 * mesh.flatten() * S**0.5 * A**(-1.5), dA_dcsec)
        partials["mesh", "S"] = 0.5 * S**(-0.5) / A**0.5 * mesh.flatten()


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
