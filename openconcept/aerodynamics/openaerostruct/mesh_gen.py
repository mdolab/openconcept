from copy import deepcopy
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
    is scaled to match the requested planform area. Optionally, this scaling
    can be turned off (by setting scale_area=False) and the planform area is
    then computed as an output.

    The mesh points are cosine spaced in the chordwise direction and in the
    spanwise direction within each trapezoidal region of the wing defined by
    pairs of section properties.

    .. code-block:: text

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
        Full planform area; only an input when scale_area is True (the default) (scalar, m^2)
    x_LE_sec : float
        Streamwise offset of the section's leading edge, starting with the outboard
        section (wing tip) and moving inboard toward the root (vector of length
        num_sections, m)
    y_sec : float
        Spanwise location of each section, starting with the outboard section (wing
        tip) at the MOST NEGATIVE y value and moving inboard (increasing y value)
        toward the root; the user does not provide a value for the root because it
        is always 0.0 (vector of length num_sections - 1, m)
    chord_sec : float
        Chord of each section, starting with the outboard section (wing tip) and
        moving inboard toward the root (vector of length num_sections, m)

    Outputs
    -------
    mesh: ndarray
        OpenAeroStruct 3D mesh (num_x + 1 x sum(num_y) + 1 x 3 ndarray, m)
    S: float
        Full planform area; only an output when scale_area is False (scalar, m^2)

    Options
    -------
    num_x: int
        Number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y: int or iterable of ints
        Number of spanwise panels in the trapezoidal regions between each pair of
        adjacent sections; can be specified either as a single integer, where that
        same value is used for each region, or as an iterable of integers of length
        num_sections - 1 to enable different numbers of spanwise coordinates for each
        region (scalar or vector, dimensionless)
    num_sections : int
        Number of spanwise sections to define planform shape (scalar, dimensionless)
    scale_area : bool
        Scale the mesh to match a planform area provided as an input
    """  # noqa

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
        self.options.declare("scale_area", default=True, types=bool, desc="Scale the planform area")

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

        self.scale = self.options["scale_area"]

        self.add_input("x_LE_sec", val=0, shape=(self.n_sec,), units="m")
        self.add_input("y_sec", val=np.linspace(-5, 0, self.n_sec)[:-1], shape=(self.n_sec - 1,), units="m")
        self.add_input("chord_sec", val=1, shape=(self.n_sec,), units="m")
        if self.scale:
            self.add_input("S", val=10, units="m**2")
            self.declare_partials("mesh", "S")
        else:
            self.add_output("S", val=10, units="m**2")
            self.declare_partials("S", ["y_sec", "chord_sec"])

        # Generate default mesh
        self.ny_tot = np.sum(self.ny) + 1
        x, y = np.meshgrid(cos_space(0, 1, self.nx), cos_space(-5, 0, self.ny_tot), indexing="ij")
        y *= 5
        mesh = np.zeros((self.nx, self.ny_tot, 3))
        mesh[:, :, 0] = x
        mesh[:, :, 1] = y

        self.add_output("mesh", val=mesh, shape=(self.nx, self.ny_tot, 3), units="m")

        # Inputs cache to know if the mesh has been updated or not
        self.inputs_cache = None
        self.dummy_outputs = {"mesh": np.zeros((self.nx, self.ny_tot, 3))}

        # Compute some values for partial derivatives that don't depend on input values
        self.dmesh_dysec_no_S_influence = np.zeros((self.nx * self.ny_tot * 3, self.n_sec - 1), dtype=float)
        self.dmesh_dcsec_no_S_influence = np.zeros((self.nx * self.ny_tot * 3, self.n_sec), dtype=float)
        self.dmesh_dxLEsec_no_S_influence = np.zeros(self.ny_tot * self.nx * 2, dtype=float)
        dmesh_dxLEsec_rows = np.zeros(self.ny_tot * self.nx * 2, dtype=int)  # rows for partial derivative sparsity
        dmesh_dxLEsec_cols = np.zeros(self.ny_tot * self.nx * 2, dtype=int)  # cols for partial derivative sparsity

        # Indices in flattened array
        idx_x = 3 * np.arange(self.nx * self.ny_tot)
        idx_y = idx_x + 1

        # Iterate through the defined trapezoidal regions between sections
        y_prev = 0
        idx_dmesh_dxLE_prev = 0
        for i_sec in range(self.n_sec - 1):
            ny = self.ny[i_sec] + 1  # number of coordinates in the current region (including at the ends)
            x_mesh = np.repeat(cos_space(0, 1, self.nx), ny).reshape(self.nx, ny)

            # Derivatives of this trapezoidal region
            cos_deriv_start = cos_space_deriv_start(ny)
            cos_deriv_end = cos_space_deriv_end(ny)
            dymesh_dysec = np.tile(cos_deriv_start, (self.nx, 1))
            dymesh_dysecnext = np.tile(cos_deriv_end, (self.nx, 1))
            dxmesh_dxsec = dymesh_dysec
            dxmesh_dxsecnext = dymesh_dysecnext
            dxmesh_dcsec = x_mesh * cos_deriv_start
            dxmesh_dcsecnext = x_mesh * cos_deriv_end

            # We must be careful not to double count the derivatives at intermediate sections,
            # since they are both a beginning and ending point of a trapezoidal region. To do this
            # only include the furthest outboard section of the region if this trapezoidal region
            # includes the wing tip
            y_idx_start = 0 if i_sec == 0 else 1
            # Indices in the mesh (not x, y, z values) corresponding to the coordinates in the current region
            idx_mesh = (
                np.tile(np.arange(y_prev + y_idx_start, y_prev + ny), (self.nx, 1)).T + np.arange(self.nx) * self.ny_tot
            ).T.flatten()
            # No derivative w.r.t. the y value at the root because it's always zero
            if i_sec < self.n_sec - 2:
                self.dmesh_dysec_no_S_influence[idx_y[idx_mesh], i_sec + 1] += dymesh_dysecnext[
                    :, y_idx_start:
                ].flatten()
            self.dmesh_dysec_no_S_influence[idx_y[idx_mesh], i_sec] += dymesh_dysec[:, y_idx_start:].flatten()
            self.dmesh_dcsec_no_S_influence[idx_x[idx_mesh], i_sec + 1] += dxmesh_dcsecnext[:, y_idx_start:].flatten()
            self.dmesh_dcsec_no_S_influence[idx_x[idx_mesh], i_sec] += dxmesh_dcsec[:, y_idx_start:].flatten()

            # Derivatives w.r.t. x_LE_sec are sparse, so track rows and cols
            idx_end = idx_dmesh_dxLE_prev + idx_mesh.size
            dmesh_dxLEsec_rows[idx_dmesh_dxLE_prev:idx_end] = idx_x[idx_mesh]
            dmesh_dxLEsec_cols[idx_dmesh_dxLE_prev:idx_end] = i_sec + 1
            self.dmesh_dxLEsec_no_S_influence[idx_dmesh_dxLE_prev:idx_end] = dxmesh_dxsecnext[:, y_idx_start:].flatten()
            idx_dmesh_dxLE_prev += idx_mesh.size

            idx_end += idx_mesh.size
            dmesh_dxLEsec_rows[idx_dmesh_dxLE_prev:idx_end] = idx_x[idx_mesh]
            dmesh_dxLEsec_cols[idx_dmesh_dxLE_prev:idx_end] = i_sec
            self.dmesh_dxLEsec_no_S_influence[idx_dmesh_dxLE_prev:idx_end] = dxmesh_dxsec[:, y_idx_start:].flatten()
            idx_dmesh_dxLE_prev += idx_mesh.size

            y_prev += ny - 1

        # Use sparsity for w.r.t. x_LE_sec but don't use sparsity for w.r.t. y_sec and chord_sec since
        # they're ~2/3 nonzero regardless of mesh size; still might be better to make y_sec and chord_sec
        # zero since derivatives of all z values are zero and the Jacobian will end up as a sparse data
        # structure under the hood
        self.declare_partials("mesh", ["y_sec", "chord_sec"])
        self.declare_partials("mesh", "x_LE_sec", rows=dmesh_dxLEsec_rows, cols=dmesh_dxLEsec_cols)

    def compute(self, inputs, outputs):
        x_sec = inputs["x_LE_sec"]
        y_sec = np.hstack((inputs["y_sec"], 0))
        c_sec = inputs["chord_sec"]

        self.inputs_cache = deepcopy(dict(inputs))

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

        # Copy the unscaled mesh for use in the derivative calculation
        self.unscaled_flattened_mesh = outputs["mesh"].copy().flatten()

        # Scale the mesh by the reference area
        A *= 2  # we're only doing a half wing, so double to get total area
        if self.scale:
            outputs["mesh"] *= (inputs["S"] / A) ** 0.5
        else:
            outputs["S"] = A

    def compute_partials(self, inputs, partials):
        y_sec = np.hstack((inputs["y_sec"], 0))
        c_sec = inputs["chord_sec"]

        # Make sure self.usncaled_mesh has been updated with the correct inputs
        for input_name in inputs.keys():
            if self.inputs_cache is None or np.any(inputs[input_name] != self.inputs_cache[input_name]):
                self.compute(inputs, self.dummy_outputs)
                break

        # Load in the initial values of the partials that are independent of the inputs;
        # this avoids recomputing them every time compute_partials is called
        partials["mesh", "y_sec"][:, :] = self.dmesh_dysec_no_S_influence[:, :]
        partials["mesh", "x_LE_sec"][:] = self.dmesh_dxLEsec_no_S_influence[:]
        partials["mesh", "chord_sec"][:, :] = self.dmesh_dcsec_no_S_influence[:, :]

        # Iterate through the defined trapezoidal regions between sections
        A = 0.0  # nondimensional area of existing mesh
        dA_dysec = np.zeros(self.n_sec - 1, dtype=float)
        dA_dcsec = np.zeros(self.n_sec, dtype=float)
        for i_sec in range(self.n_sec - 1):
            # Add the area of this trapezoidal region to the total nondimensional planform area
            A += (y_sec[i_sec + 1] - y_sec[i_sec]) * (c_sec[i_sec] + c_sec[i_sec + 1]) * 0.5
            dA_dysec[i_sec] += -0.5 * (c_sec[i_sec] + c_sec[i_sec + 1])
            if i_sec < self.n_sec - 2:
                dA_dysec[i_sec + 1] += 0.5 * (c_sec[i_sec] + c_sec[i_sec + 1])
            dA_dcsec[i_sec : i_sec + 2] += 0.5 * (y_sec[i_sec + 1] - y_sec[i_sec])

        # Scale the mesh by the reference area
        A *= 2  # we're only doing a half wing, so double to get total area
        dA_dysec *= 2
        dA_dcsec *= 2
        if self.scale:
            S = inputs["S"].item()
            coeff = (S / A) ** 0.5
            for var in ["y_sec", "chord_sec", "x_LE_sec"]:
                partials["mesh", var] *= coeff
            partials["mesh", "y_sec"] += np.outer(
                -0.5 * self.unscaled_flattened_mesh * S**0.5 * A ** (-1.5), dA_dysec
            )
            partials["mesh", "chord_sec"] += np.outer(
                -0.5 * self.unscaled_flattened_mesh * S**0.5 * A ** (-1.5), dA_dcsec
            )
            partials["mesh", "S"] = 0.5 * S ** (-0.5) / A**0.5 * self.unscaled_flattened_mesh
        else:
            partials["S", "y_sec"] = dA_dysec
            partials["S", "chord_sec"] = dA_dcsec


class ThicknessChordRatioInterp(om.ExplicitComponent):
    """
    Linearly interpolate thickness to chord ratio defined at each section.
    Take the average of the values on either side of a panel to determine
    each panel's thickness to chord ratio.

    Inputs
    ------
    section_toverc : float
        Thickness to chord ratio at each defined section, starting at the
        tip and moving to the root (vector of length num_sections, dimensionless)

    Outputs
    -------
    panel_toverc : float
        Thickness to chord ratio at every streamwise strip of panels in
        the mesh (vector of length total panels in y, dimensionless)

    Options
    -------
    num_y: int or iterable of ints
        Number of spanwise panels in the trapezoidal regions between each pair of
        adjacent sections; can be specified either as a single integer, where that
        same value is used for each region, or as an iterable of integers of length
        num_sections - 1 to enable different numbers of spanwise coordinates for each
        region (scalar or vector, dimensionless)
    num_sections : int
        Number of spanwise sections to define planform shape (scalar, dimensionless)
    cos_spacing : bool
        mesh is cosine spaced between defined sections, by default True (should be True
        if the mesh is generated by SectionPlanformMesh and False if mesh is generated
        by TrapezoidalPlanformMesh)
    """

    def initialize(self):
        self.options.declare(
            "num_y",
            default=7,
            types=(int, list, tuple, np.ndarray),
            desc="Number of spanwise mesh points per trapezoidal region",
        )
        self.options.declare(
            "num_sections", default=2, types=int, desc="Number of defined sections along the half span"
        )
        self.options.declare("cos_spacing", default=True, types=bool, desc="Mesh is cosine spaced within each region")

    def setup(self):
        # Process mesh resolution options
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
        self.ny_tot = np.sum(self.ny)  # total number of panels

        self.add_input("section_toverc", val=0.12, shape=(self.n_sec,))
        self.add_output("panel_toverc", val=0.12, shape=(self.ny_tot,))

        # Compute the partial derivatives (don't depend on inputs since this is linear)
        y_prev = 0
        idx_partial_vec = 0
        dpanel_dsection_rows = np.zeros(self.ny_tot * 2, dtype=float)
        dpanel_dsection_cols = np.zeros(self.ny_tot * 2, dtype=float)
        dpanel_dsection_vals = np.zeros(self.ny_tot * 2, dtype=float)
        for i_sec in range(self.n_sec - 1):
            ny = self.ny[i_sec]
            # First, compute the linearly-interpolated thickness to chord ratio at each y mesh point position
            if self.options["cos_spacing"]:
                dnodal_dstart = cos_space_deriv_start(ny + 1)
                dnodal_dend = cos_space_deriv_end(ny + 1)
            else:
                dnodal_dstart = np.linspace(1, 0, ny + 1)
                dnodal_dend = np.linspace(0, 1, ny + 1)

            # For each panel's t/c, take the average of the nodal t/c's on the panel's two sides
            dpanel_dsection_rows[idx_partial_vec : idx_partial_vec + ny] = np.arange(y_prev, y_prev + ny)
            dpanel_dsection_cols[idx_partial_vec : idx_partial_vec + ny] = i_sec
            dpanel_dsection_vals[idx_partial_vec : idx_partial_vec + ny] = 0.5 * (
                dnodal_dstart[:-1] + dnodal_dstart[1:]
            )
            idx_partial_vec += ny

            dpanel_dsection_rows[idx_partial_vec : idx_partial_vec + ny] = np.arange(y_prev, y_prev + ny)
            dpanel_dsection_cols[idx_partial_vec : idx_partial_vec + ny] = i_sec + 1
            dpanel_dsection_vals[idx_partial_vec : idx_partial_vec + ny] = 0.5 * (dnodal_dend[:-1] + dnodal_dend[1:])
            idx_partial_vec += ny

            y_prev += ny

        self.declare_partials(
            "panel_toverc",
            "section_toverc",
            rows=dpanel_dsection_rows,
            cols=dpanel_dsection_cols,
            val=dpanel_dsection_vals,
        )

    def compute(self, inputs, outputs):
        tc_section = inputs["section_toverc"]

        # Loop through each region
        y_prev = 0
        for i_sec in range(self.n_sec - 1):
            ny = self.ny[i_sec]
            # First, compute the linearly-interpolated thickness to chord ratio at each y mesh point position
            if self.options["cos_spacing"]:
                tc_nodal = cos_space(tc_section[i_sec], tc_section[i_sec + 1], ny + 1, dtype=tc_section.dtype)
            else:
                tc_nodal = np.linspace(tc_section[i_sec], tc_section[i_sec + 1], ny + 1, dtype=tc_section.dtype)

            # For each panel's t/c, take the average of the nodal t/c's on the panel's two sides
            outputs["panel_toverc"][y_prev : y_prev + ny] = 0.5 * (tc_nodal[:-1] + tc_nodal[1:])

            y_prev += ny


class SectionLinearInterp(om.ExplicitComponent):
    """
    Linearly interpolate a property defined at each section and
    evaluate the result at spanwise mesh points.

    Inputs
    ------
    property_sec : float
        The property to be interpolated at each section definition,
        defined from tip to root (vector of length num_sections)

    Outputs
    -------
    property_node : float
        The linearly interpolated value at each y-coordinate (vector of
        length number of different y-coordinates)

    Options
    -------
    num_y: int or iterable of ints
        Number of spanwise panels in the trapezoidal regions between each pair of
        adjacent sections; can be specified either as a single integer, where that
        same value is used for each region, or as an iterable of integers of length
        num_sections - 1 to enable different numbers of spanwise coordinates for each
        region (scalar or vector, dimensionless)
    num_sections : int
        Number of spanwise sections to define planform shape (scalar, dimensionless)
    units : str
        Units of the interpolated quantity
    cos_spacing : bool
        mesh is cosine spaced between defined sections, by default True (should be True
        if the mesh is generated by SectionPlanformMesh and False if mesh is generated
        by TrapezoidalPlanformMesh)
    """

    def initialize(self):
        self.options.declare(
            "num_y",
            default=7,
            types=(int, list, tuple, np.ndarray),
            desc="Number of spanwise mesh points per trapezoidal region",
        )
        self.options.declare(
            "num_sections", default=2, types=int, desc="Number of defined sections along the half span"
        )
        self.options.declare("units", default=None, desc="Units of interpolated quantity")
        self.options.declare("cos_spacing", default=True, types=bool, desc="Mesh is cosine spaced within each region")

    def setup(self):
        # Process mesh resolution options
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
        self.ny_tot = np.sum(self.ny) + 1  # total number of coordinates

        self.add_input("property_sec", val=0.0, shape=(self.n_sec,), units=self.options["units"])
        self.add_output("property_node", val=0.0, shape=(self.ny_tot,), units=self.options["units"])

        # Compute the partial derivatives (don't depend on inputs since this is linear)
        y_prev = 0
        idx_partial_vec = 0
        num_nonzeros = np.sum(np.asarray(self.ny) - 1) * 2 + self.n_sec
        dnodal_dsection_rows = np.zeros(num_nonzeros, dtype=float)
        dnodal_dsection_cols = np.zeros(num_nonzeros, dtype=float)
        dnodal_dsection_vals = np.zeros(num_nonzeros, dtype=float)
        for i_sec in range(self.n_sec - 1):
            ny = self.ny[i_sec]
            # Compute the linearly-interpolated property at each y mesh point position
            if self.options["cos_spacing"]:
                dnodal_dstart = cos_space_deriv_start(ny + 1)
                dnodal_dend = cos_space_deriv_end(ny + 1)
            else:
                dnodal_dstart = np.linspace(1, 0, ny + 1)
                dnodal_dend = np.linspace(0, 1, ny + 1)

            # Partials from the starting and ending section on nodal properties
            idx_start = 0 if i_sec == 0 else 1
            dnodal_dsection_rows[idx_partial_vec : idx_partial_vec + ny - idx_start] = np.arange(
                y_prev + idx_start, y_prev + ny
            )
            dnodal_dsection_cols[idx_partial_vec : idx_partial_vec + ny - idx_start] = i_sec
            dnodal_dsection_vals[idx_partial_vec : idx_partial_vec + ny - idx_start] = dnodal_dstart[idx_start:-1]
            idx_partial_vec += ny - idx_start

            dnodal_dsection_rows[idx_partial_vec : idx_partial_vec + ny] = np.arange(y_prev + 1, y_prev + ny + 1)
            dnodal_dsection_cols[idx_partial_vec : idx_partial_vec + ny] = i_sec + 1
            dnodal_dsection_vals[idx_partial_vec : idx_partial_vec + ny] = dnodal_dend[1:]
            idx_partial_vec += ny

            y_prev += ny

        self.declare_partials(
            "property_node",
            "property_sec",
            rows=dnodal_dsection_rows,
            cols=dnodal_dsection_cols,
            val=dnodal_dsection_vals,
        )

    def compute(self, inputs, outputs):
        prop_sec = inputs["property_sec"]

        # Loop through each region
        y_prev = 0
        for i_sec in range(self.n_sec - 1):
            ny = self.ny[i_sec]  # number of panels in this region

            # Compute the linearly-interpolated property at each y mesh point position
            if self.options["cos_spacing"]:
                outputs["property_node"][y_prev : y_prev + ny + 1] = cos_space(
                    prop_sec[i_sec], prop_sec[i_sec + 1], ny + 1, dtype=prop_sec.dtype
                )
            else:
                outputs["property_node"][y_prev : y_prev + ny + 1] = np.linspace(
                    prop_sec[i_sec], prop_sec[i_sec + 1], ny + 1, dtype=prop_sec.dtype
                )

            y_prev += ny


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
