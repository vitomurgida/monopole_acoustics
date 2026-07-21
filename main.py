import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib
matplotlib.use('TkAgg')
plt.close('all')

# Removed Poly3DCollection import to avoid the broken matplotlib dependency
warnings.filterwarnings("ignore", category=RuntimeWarning)


def setup_geometry(obstacles, ground_on):
    """
    Translates obstacles into pure tensors representing edges and faces.
    """
    v1_list, v2_list, z1_list, z2_list, obs_id_list = [], [], [], [], []
    num_obs = len(obstacles)

    for i, obs in enumerate(obstacles):
        verts = np.array(obs['vertices'], dtype=np.float32)
        z_start, z_end = obs['z_start'], obs['z_end']
        N = len(verts)
        for j in range(N):
            v1_list.append(verts[j])
            v2_list.append(verts[(j + 1) % N])
            z1_list.append(z_start)
            z2_list.append(z_end)
            obs_id_list.append(i)

    # Vertical edges (walls)
    E_v1 = np.array(v1_list, dtype=np.float32)
    E_v2 = np.array(v2_list, dtype=np.float32)
    E_z1 = np.array(z1_list, dtype=np.float32)
    E_z2 = np.array(z2_list, dtype=np.float32)
    E_obs_id = np.array(obs_id_list, dtype=np.int32)

    # Calculate normals for vertical faces
    dx = E_v2[:, 0] - E_v1[:, 0]
    dy = E_v2[:, 1] - E_v1[:, 1]
    lengths = np.sqrt(dx ** 2 + dy ** 2)
    n_vert = np.column_stack([dy, -dx, np.zeros_like(dx)]) / lengths[:, np.newaxis]
    p0_vert = np.column_stack([E_v1[:, 0], E_v1[:, 1], E_z1])

    # Horizontal faces (top/bottom)
    h_z = []
    h_obs_id = []
    h_n = []
    for i, obs in enumerate(obstacles):
        h_z.extend([obs['z_start'], obs['z_end']])
        h_obs_id.extend([i, i])
        h_n.extend([[0, 0, -1], [0, 0, 1]])

    H_z = np.array(h_z, dtype=np.float32)
    H_obs_id = np.array(h_obs_id, dtype=np.int32)
    H_n = np.array(h_n, dtype=np.float32)
    H_p0 = np.column_stack([np.zeros_like(H_z), np.zeros_like(H_z), H_z])

    # Combine all faces into a unified reflection tensor
    F_n = np.vstack([n_vert, H_n])
    F_p0 = np.vstack([p0_vert, H_p0])

    if ground_on:
        F_n = np.vstack([F_n, [0, 0, 1]])
        F_p0 = np.vstack([F_p0, [0, 0, 0]])

    geom = {
        'E_v1': E_v1, 'E_v2': E_v2, 'E_z1': E_z1, 'E_z2': E_z2, 'E_obs_id': E_obs_id,
        'H_z': H_z, 'H_obs_id': H_obs_id, 'num_obs': num_obs,
        'F_n': F_n, 'F_p0': F_p0, 'ground_on': ground_on
    }
    return geom


def is_inside_polygons_tensor(P_2d, geom):
    """ Vectorized ray-casting for 2D Point-in-Polygon. No for loops. """
    P_ext = P_2d[..., np.newaxis, :]
    v1 = geom['E_v1'];
    v2 = geom['E_v2']

    # Y-bounds check
    y_cond = (v1[:, 1] > P_ext[..., 1]) != (v2[:, 1] > P_ext[..., 1])
    # X-intersection
    dy = v2[:, 1] - v1[:, 1] + 1e-12
    x_int = v1[:, 0] + (P_ext[..., 1] - v1[:, 1]) * (v2[:, 0] - v1[:, 0]) / dy
    x_cond = P_ext[..., 0] < x_int

    crosses = (y_cond & x_cond).astype(np.int32)

    # Matrix multiply to sum crosses per obstacle
    M = np.zeros((len(geom['E_obs_id']), geom['num_obs']), dtype=np.int32)
    M[np.arange(len(geom['E_obs_id'])), geom['E_obs_id']] = 1
    cross_counts = crosses @ M

    return (cross_counts % 2) == 1


def check_occlusion(A, B, geom):
    """ Checks if rays A->B are occluded by ANY face in the scene. Returns boolean mask (True=Occluded). """
    dir_vec = B - A
    A_ext = A[..., np.newaxis, :]
    dir_ext = dir_vec[..., np.newaxis, :]

    # --- Check Vertical Walls ---
    n_v = geom['F_n'][:len(geom['E_v1'])]
    p0_v = geom['F_p0'][:len(geom['E_v1'])]

    den_v = np.sum(dir_ext * n_v, axis=-1)
    t_v = np.sum((p0_v - A_ext) * n_v, axis=-1) / (den_v + 1e-12)
    valid_t_v = (t_v > 1e-4) & (t_v < 1 - 1e-4)

    P_int_v = A_ext + t_v[..., np.newaxis] * dir_ext
    z_int_v = P_int_v[..., 2]
    z_valid = (z_int_v >= geom['E_z1']) & (z_int_v <= geom['E_z2'])

    edge_vec = geom['E_v2'] - geom['E_v1']
    P_int_2d = P_int_v[..., :2]
    s = np.sum((P_int_2d - geom['E_v1']) * edge_vec, axis=-1) / (np.sum(edge_vec ** 2, axis=-1) + 1e-12)
    edge_valid = (s >= 0) & (s <= 1)

    occluded_v = np.any(valid_t_v & z_valid & edge_valid, axis=-1)

    # --- Check Horizontal Caps ---
    n_h = geom['F_n'][len(geom['E_v1']):len(geom['E_v1']) + len(geom['H_z'])]
    p0_h = geom['F_p0'][len(geom['E_v1']):len(geom['E_v1']) + len(geom['H_z'])]

    den_h = np.sum(dir_ext * n_h, axis=-1)
    t_h = np.sum((p0_h - A_ext) * n_h, axis=-1) / (den_h + 1e-12)
    valid_t_h = (t_h > 1e-4) & (t_h < 1 - 1e-4)

    P_int_h = A_ext + t_h[..., np.newaxis] * dir_ext
    inside_poly = is_inside_polygons_tensor(P_int_h[..., :2], geom)

    # Map polygon inside array to horizontal faces
    H_obs_idx = geom['H_obs_id']

    # Advanced indexing: we slice the grid dimensions (...),
    # and map each horizontal face to its specific obstacle ID
    inside_h = inside_poly[..., np.arange(len(H_obs_idx)), H_obs_idx]

    occluded_h = np.any(valid_t_h & inside_h, axis=-1)

    # --- Check Ground ---
    occluded_g = np.zeros_like(occluded_v)
    if geom['ground_on']:
        t_g = -A[..., 2] / (dir_vec[..., 2] + 1e-12)
        occluded_g = (t_g > 1e-4) & (t_g < 1 - 1e-4)

    return occluded_v | occluded_h | occluded_g


def create_validity_mask(grid_pts, sources, geom):
    """
    grid_pts: (V, 3)
    sources: (S, 3)
    Returns: Validity mask (V, S, 1 + F) where index 0 is direct sound, 1:F are image sources.
    """
    V = grid_pts.shape[0]
    S = sources.shape[0]
    F = geom['F_n'].shape[0]

    # Generate Images (S, F, 3)
    s_ext = sources[:, np.newaxis, :]
    dist = np.sum((s_ext - geom['F_p0']) * geom['F_n'], axis=-1, keepdims=True)
    images = s_ext - 2 * dist * geom['F_n']

    # Validity mask shape (V, S, 1+F)
    mask = np.zeros((V, S, 1 + F), dtype=bool)

    # 1. Direct path
    for s in range(S):
        mask[:, s, 0] = ~check_occlusion(np.full((V, 3), sources[s]), grid_pts, geom)

    # 2. Image paths
    for s in range(S):
        for f in range(F):
            I = images[s, f]
            # Calculate reflection point on face f
            dir_vec = grid_pts - I
            den = np.sum(dir_vec * geom['F_n'][f], axis=-1)
            t = np.sum((geom['F_p0'][f] - I) * geom['F_n'][f]) / (den + 1e-12)
            P_refl = I + t[:, np.newaxis] * dir_vec

            # Check if P_refl is inside Face f
            valid_refl = (t > 1e-5) & (t < 1)
            if f < len(geom['E_v1']):  # Vertical
                z_valid = (P_refl[:, 2] >= geom['E_z1'][f]) & (P_refl[:, 2] <= geom['E_z2'][f])
                edge_v = geom['E_v2'][f] - geom['E_v1'][f]
                s_val = np.sum((P_refl[:, :2] - geom['E_v1'][f]) * edge_v, axis=-1) / (np.sum(edge_v ** 2) + 1e-12)
                inside = z_valid & (s_val >= 0) & (s_val <= 1)
            elif f < len(geom['E_v1']) + len(geom['H_z']):  # Horizontal
                idx_poly = f - len(geom['E_v1'])
                obs_id = geom['H_obs_id'][idx_poly]
                poly_mask = is_inside_polygons_tensor(P_refl[:, :2], geom)
                inside = poly_mask[:, obs_id]
            else:  # Ground
                inside = True

            # Check occlusions
            S_array = np.full((V, 3), sources[s])
            clear_1 = ~check_occlusion(S_array, P_refl, geom)
            clear_2 = ~check_occlusion(P_refl, grid_pts, geom)

            mask[:, s, f + 1] = valid_refl & inside & clear_1 & clear_2

    return mask, images

def compute_pressure_tensor(grid_shape, grid_pts, sources, images, mask, freqs, phases, powers, c=343.0, rho0=1.21):

    V = grid_pts.shape[0]
    S, F, _ = images.shape
    N_f = freqs.shape[1]
    N_p = phases.shape[1]

    # Combine sources and images (S, 1+F, 3)
    src_all = np.concatenate([sources[:, np.newaxis, :], images], axis=1)

    # Distance R: (V, S, 1+F)
    diff = grid_pts[:, np.newaxis, np.newaxis, :] - src_all[np.newaxis, ...]
    R = np.linalg.norm(diff, axis=-1)
    R = np.clip(R, 1e-4, None)

    # We want final shape: (V, N_f, N_p, S, 1+F)
    R_exp = R[:, np.newaxis, np.newaxis, :, :]  # (V, 1, 1, S, 1+F)

    # Convert Acoustic Power (Watts) to Peak Pressure Amplitude at 1m
    A = np.sqrt(powers * rho0 * c / (2 * np.pi)) # peak pressure, not rms
    A_exp = A.reshape(1, 1, 1, S, 1)  # Broadcast to (1, 1, 1, S, 1)

    # k is (S, N_f). Transpose to (N_f, S)
    k = (2 * np.pi * freqs / c)
    k_exp = k.T[np.newaxis, :, np.newaxis, :, np.newaxis]  # (1, N_f, 1, S, 1)

    # phases is (S, N_p). Transpose to (N_p, S)
    phi_exp = phases.T[np.newaxis, np.newaxis, :, :, np.newaxis]  # (1, 1, N_p, S, 1)

    # mask is (V, S, 1+F)
    mask_exp = mask[:, np.newaxis, np.newaxis, :, :]  # (V, 1, 1, S, 1+F)

    # Pressure computation using the calculated Amplitude (A_exp)
    P = (A_exp / R_exp) * np.exp(-1j * (k_exp * R_exp - phi_exp))
    P = P.astype(np.complex64)

    # Multiply by mask
    P *= mask_exp

    # Reshape space to Grid (Nx, Ny, Nz)
    final_shape = (*grid_shape, N_f, N_p, S, 1 + F)
    return P.reshape(final_shape)


def plot_2d_slice(tensor, x, y, z, idx, plane, title):
    # tensor shape (Nx,Ny,Nz, Nf,Np,Ns,Ni)
    # Sum over phases, sources, images -> shape (Nx, Ny, Nz, Nf)
    P_coherent = np.sum(tensor, axis=(4, 5, 6))

    # Select the slice and define physical axes (meters)
    if plane == 'z':
        # Z = constant -> plot over X-Y
        P_slice = P_coherent[:, :, idx, :]          # (Nx, Ny, Nf)
        h_arr, v_arr = x, y                         # meters
        h_label, v_label = 'X [m]', 'Y [m]'
        plane_val = z[idx]

        # After summing freq -> (Nx, Ny) which matches (x, y)
        transpose_for_plot = True  # because we will plot (Ny, Nx) as (y, x)

    elif plane == 'x':
        # X = constant -> plot over Y-Z
        P_slice = P_coherent[idx, :, :, :]          # (Ny, Nz, Nf)
        h_arr, v_arr = y, z                         # meters
        h_label, v_label = 'Y [m]', 'Z [m]'
        plane_val = x[idx]
        transpose_for_plot = True

    elif plane == 'y':
        # Y = constant -> plot over X-Z
        P_slice = P_coherent[:, idx, :, :]          # (Nx, Nz, Nf)
        h_arr, v_arr = x, z                         # meters
        h_label, v_label = 'X [m]', 'Z [m]'
        plane_val = y[idx]
        transpose_for_plot = True

    else:
        raise ValueError("plane must be one of: 'x', 'y', 'z'")

    # SPL = incoherent sum over frequencies of coherent fields
    P_rms_sq = np.sum(np.abs(P_slice) ** 2, axis=-1) / 2.0
    SPL = 10 * np.log10(P_rms_sq / (2e-5) ** 2 + 1e-12)

    # Real pressure (at t=0) across all frequencies
    P_real = np.sum(np.real(P_slice), axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # If slice degenerates to 1D, fallback to line plots with meter axis
    if P_real.shape[0] < 2 or P_real.shape[1] < 2:
        # Choose the non-singleton axis for x-axis units
        if P_real.shape[0] >= 2:
            axis_vals = h_arr
            real_line = P_real[:, 0] if P_real.ndim == 2 else P_real
            spl_line = SPL[:, 0] if SPL.ndim == 2 else SPL
        else:
            axis_vals = v_arr
            real_line = P_real[0, :] if P_real.ndim == 2 else P_real
            spl_line = SPL[0, :] if SPL.ndim == 2 else SPL

        ax1.plot(axis_vals, real_line, color='b')
        ax1.set_title(f'Real Pressure (1D) on {plane}={plane_val:.2f} m')
        ax1.set_xlabel(f'{h_label if P_real.shape[0] >= 2 else v_label}')
        ax1.grid(True)

        ax2.plot(axis_vals, spl_line, color='r')
        ax2.set_title(f'SPL (dB) (1D) on {plane}={plane_val:.2f} m')
        ax2.set_xlabel(f'{h_label if P_real.shape[0] >= 2 else v_label}')
        ax2.grid(True)

    else:
        # Standard 2D Contour Plot IN METERS:
        # P_real is shaped (H, V) but contourf expects Z as (len(v_arr), len(h_arr))
        Z_real = P_real.T if transpose_for_plot else P_real
        Z_spl = SPL.T if transpose_for_plot else SPL

        c1 = ax1.contourf(h_arr, v_arr, Z_real, levels=50, cmap='RdBu_r')
        fig.colorbar(c1, ax=ax1)
        ax1.set_title(f'Real Pressure ({plane}={plane_val:.2f} m)')
        ax1.set_xlabel(h_label)
        ax1.set_ylabel(v_label)
        ax1.set_aspect('equal', adjustable='box')

        c2 = ax2.contourf(h_arr, v_arr, Z_spl, levels=50, cmap='inferno')
        fig.colorbar(c2, ax=ax2)
        ax2.set_title(f'SPL (dB) ({plane}={plane_val:.2f} m)')
        ax2.set_xlabel(h_label)
        ax2.set_ylabel(v_label)
        ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

def plot_3d_scene(sources, receivers, geom, images, valid_mask_rec):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw obstacles as wireframes
    for obs_id in range(geom['num_obs']):
        idx = geom['E_obs_id'] == obs_id
        v1 = geom['E_v1'][idx]
        v2 = geom['E_v2'][idx]
        z1 = geom['E_z1'][idx][0]
        z2 = geom['E_z2'][idx][0]

        for pt1, pt2 in zip(v1, v2):
            # bottom edge
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [z1, z1], color='k', linewidth=1)
            # top edge
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [z2, z2], color='k', linewidth=1)
            # vertical edge
            ax.plot([pt1[0], pt1[0]], [pt1[1], pt1[1]], [z1, z2], color='k', linewidth=1)

    # Draw sources and receivers
    ax.scatter(*sources.T, color='red', s=50, label='Sources')
    ax.scatter(receivers[:, 0], receivers[:, 1], receivers[:, 2],
               color='green', s=40, label='Receivers')

    S = sources.shape[0]
    R = receivers.shape[0]
    F = images.shape[1]

    # Draw rays for each receiver
    for r in range(R):
        rec = receivers[r]

        for s in range(S):
            # Direct ray
            if valid_mask_rec[r, s, 0]:
                ax.plot([sources[s, 0], rec[0]],
                        [sources[s, 1], rec[1]],
                        [sources[s, 2], rec[2]],
                        'r--', alpha=0.35)

            # Reflected rays (single-bounce images)
            for f in range(F):
                if valid_mask_rec[r, s, f + 1]:
                    img = images[s, f]

                    # Intersection point of line (img -> rec) with plane f
                    denom = np.sum((rec - img) * geom['F_n'][f]) + 1e-12
                    t = np.sum((geom['F_p0'][f] - img) * geom['F_n'][f]) / denom
                    refl_pt = img + t * (rec - img)

                    ax.plot([sources[s, 0], refl_pt[0], rec[0]],
                            [sources[s, 1], refl_pt[1], rec[1]],
                            [sources[s, 2], refl_pt[2], rec[2]],
                            'b:', alpha=0.25)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    plt.show()

def run_simulation():
    # ---------------- USER INPUTS ----------------
    sources = np.array([[2, 2, 1.5]], dtype=np.float32) # multiple sources [[],[],...]
    freqs = np.array([[343.0, 600]]) # multiple frequencies per source [[f11,f12,...],[f21,f22,...]]
    phases = np.array([[0.0]]) # same as for sources
    powers = np.array([[0.01]]) # same as for sources
    obstacles = [
        {'vertices': [[4, 4], [6, 4], [6, 6], [4, 6]], 'z_start': 0, 'z_end': 3},
        #{'vertices': [[1, 7], [3, 7], [2, 9]], 'z_start': 1, 'z_end': 4}
    ] # at least one obstacle must be defined
    ground_on = True
    receivers = np.array([[9, 1, 1.5],[9, 2, 1.5]], dtype=np.float32)
    planes_Z = np.array([1.5, 3.0])  # Z planes on which computing and plotting
    grid_size = (100, 100, len(planes_Z))
    bounds = (0, 10, 0, 10, 0, max(planes_Z))

    print("--- Inputs ---")
    print(f"Sources:\n{sources}\nFrequencies:\n{freqs}\nPhases:\n{phases}\nGrid size: {grid_size}")

    # ---------------- EXECUTION ----------------
    x = np.linspace(bounds[0], bounds[1], grid_size[0])
    y = np.linspace(bounds[2], bounds[3], grid_size[1])
    z = np.array(planes_Z, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)

    geom = setup_geometry(obstacles, ground_on)

    # Combine receiver into grid to evaluate it simultaneously
    grid_pts_with_rec = np.vstack([receivers, grid_pts])
    R = receivers.shape[0]

    print("Computing validity masks...")
    mask, images = create_validity_mask(grid_pts_with_rec, sources, geom)

    print("Computing pressure tensor...")
    P_tensor_flat = compute_pressure_tensor(
        (grid_pts_with_rec.shape[0],), grid_pts_with_rec, sources, images, mask, freqs, phases, powers
    )

    # Receiver tensor is at indices 0..R-1
    P_rec = P_tensor_flat[:R]  # shape: (R, N_f, N_p, S, 1+F)

    # Coherent sum over phase, sources, images -> (R, N_f)
    P_rec_coh_per_f = np.sum(P_rec, axis=(2, 3, 4))

    # RMS^2: sum over frequencies of |P(f)|^2 then /2 (peak->rms)
    P_rms_sq_rec = np.sum(np.abs(P_rec_coh_per_f) ** 2, axis=-1) / 2.0  # (R,)
    SPL_rec = 10 * np.log10(P_rms_sq_rec / (2e-5) ** 2 + 1e-12)

    print("\n--- SPL at receivers ---")
    for i in range(R):
        print(f"Receiver {i}: pos={receivers[i]} -> SPL={SPL_rec[i]:.2f} dB")

    # Reshape grid tensor (excluding receiver index)
    P_tensor = P_tensor_flat[R:].reshape((*grid_size, *P_tensor_flat.shape[1:]))

    # ---------------- PLOTTING ----------------
    # Plot all Z planes
    for p_z in planes_Z:
        idx_z = np.argmin(np.abs(z - p_z))
        plot_2d_slice(P_tensor, x, y, z, idx_z, 'z', 'Z Plane')

    mask_rec = mask[:R]  # (R, S, 1+F)
    plot_3d_scene(sources, receivers, geom, images, mask_rec)


if __name__ == "__main__":
    run_simulation()
