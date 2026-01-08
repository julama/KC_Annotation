import mne
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix

def create_delta_signal(sampling_frequency, start_time, end_time, delta_time = 0, time_dim = 0):
    """
    Create a signal with a delta spike and a corresponding time array.

    Parameters:
    - sampling_frequency: The sampling frequency of the signal.
    - start_time: The start time of the signal.
    - end_time: The end time of the signal.
    - delta_time: The time at which the delta spike occurs.
    - time_dim: The dimension in the array which is the time axis.

    Returns:
    - times: The time array.
    - signal: Multidimensional array with a delta spike.
    """

    # Calculate the number of samples based on the sampling frequency and duration
    n_samples = int(np.floor(sampling_frequency * (end_time - start_time)))

    # Create the time array
    times = np.linspace(start_time, end_time, n_samples, endpoint=False)

    # Determine the index of the delta spike
    delta_index = int(np.floor((delta_time - start_time) * sampling_frequency))
    signal = np.zeros(n_samples)
    signal[delta_index] = sampling_frequency

    # Reshape the arrays to have the data in the correct dimension
    new_shape = [1] * time_dim + [-1]
    times = times.reshape(new_shape)
    signal = signal.reshape(new_shape)

    return times, signal

def calculate_phase(c, axis=0, keepdims=False):
    """
    Calculates the phase phi which maximizes norm of the real part of c / exp(1j*phi) for each element of c.

    Parameters:
    - c: Complex array.
    - axis: Axis along which the phase is calculated.
    - keepdims: If True, the output has the same number of dimensions as c.

    Returns:
    - phi: Phase array.
    """

    u = c.real
    v = c.imag

    u_sqr = np.sum(u**2, axis=axis, keepdims=keepdims)
    v_sqr = np.sum(v**2, axis=axis, keepdims=keepdims)
    uv = np.sum(u*v, axis=axis, keepdims=keepdims)

    phi = np.arctan(2*uv/(u_sqr - v_sqr))/2

    ext = u_sqr * np.cos(phi)**2 + 2*uv*np.sin(phi)*np.cos(phi) + v_sqr*np.sin(phi)**2 # extrema value
 
    phi += (ext < u_sqr)*np.pi/2

    return phi


def laea_north(xyz, radius=1.0):
    """
    Lambert azimuthal equal-area projection from 3D sphere to 2D (north pole center).
    
    Parameters
    ----------
    xyz : array-like, shape (N, 3)
        3D Cartesian coordinates on (or near) a sphere.
    radius : float, default 1.0
        Scale factor for the output coordinates.
    
    Returns
    -------
    pos : np.ndarray, shape (N, 2)
        2D projected coordinates preserving area.
    """
    xyz = np.asarray(xyz, dtype=float)
    xyz_u = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)      # normalize to unit sphere
    z = xyz_u[:, 2:3]
    k = np.sqrt(2.0 / (1.0 + z))  # denom = 1 + z
    pos = radius * k * xyz_u[:, :2] 
    return pos 


def norm(v, **kwargs):
    """
    Calculate the norm of a vector (which can be complex)

    Parameters:
    - v: Vector.

    Returns:
    - The norm of the vector.
    """
    return np.sqrt(np.sum(np.abs(v)**2, **kwargs))

def herm(A):
    """
    Calculate the conjugate transpose of a matrix.

    Parameters:
    - A: Matrix.

    Returns:
    - The conjugate transpose of the matrix.
    """
    return np.conj(A.T)


def complex_cosine_similarity(A, B, axis=0, keepdims=False, amplitude_suppression=0, eps=1e-15):
    """
    Compute the complex cosine similarity between two arrays, supporting broadcasting.

    This version is optimized for speed and numerical reliability. It handles
    arrays of different shapes according to NumPy's broadcasting rules.

    Parameters
    ----------
    A, B : np.ndarray
        Input complex arrays. Their shapes must be broadcastable to a common shape.
    axis : int, optional
        The axis along which the similarity is computed. This is the axis that gets
        summed over. Default is 0.
    keepdims : bool, optional
        If True, retains reduced dimensions. Default is False.
    amplitude_suppression : float, optional
        If non-zero, suppresses amplitude by dividing by |A|**amp and |B|**amp.
        This can help focus on phase similarity. Default is 0.
    eps : float, optional
        Small epsilon to avoid division by zero. Vectors with a norm smaller
        than eps are considered to have zero length.

    Returns
    -------
    np.ndarray
        Complex cosine similarity between A and B. The shape of the output is
        the broadcasted shape with the `axis` dimension removed.
    """
    try:
        # Ensure arrays are NumPy arrays for broadcasting
        A = np.asarray(A, dtype=np.complex128)
        B = np.asarray(B, dtype=np.complex128)
        
        # Broadcast arrays to the same shape. This is memory-efficient as it
        # creates views instead of copies where possible.
        A, B = np.broadcast_arrays(A, B)
    except ValueError:
        raise ValueError(
            f"Input arrays with original shapes {A.shape} and {B.shape} "
            "could not be broadcast together."
        )

    # 1. Amplitude suppression (optional)
    if amplitude_suppression != 0:
        abs_A = np.abs(A)
        abs_B = np.abs(B)
        
        # Suppress amplitude safely
        A = np.divide(A, abs_A**amplitude_suppression, 
                      out=np.zeros_like(A, dtype=np.complex128), 
                      where=(abs_A > eps))
        B = np.divide(B, abs_B**amplitude_suppression, 
                      out=np.zeros_like(B, dtype=np.complex128),
                      where=(abs_B > eps))

    # 2. Compute dot product and squared norms
    # These operations are now performed on the broadcasted arrays
    A_norm_sq = np.sum(A.real**2 + A.imag**2, axis=axis, keepdims=keepdims)
    B_norm_sq = np.sum(B.real**2 + B.imag**2, axis=axis, keepdims=keepdims)
    
    dot_product = np.sum(A * np.conj(B), axis=axis, keepdims=keepdims)
    cprod = np.abs(dot_product)

    # 3. Calculate similarity with protection against division by zero
    denom = np.sqrt(A_norm_sq * B_norm_sq)
    
    similarity = np.divide(cprod, denom, 
                           out=np.zeros_like(cprod, dtype=np.float64), 
                           where=(denom > eps))

    return similarity


def morlet_transform(data, sfreq, freq, n_cycles, output='complex', n_jobs=1):
    """
    Compute the Morlet wavelet transform of the input data at a specific frequency.

    Parameters:
    - data (np.ndarray): Input data array of shape (n_times, n_channels).
    - sfreq (float): Sampling frequency of the data.
    - freq (float): Frequency at which to compute the Morlet transform.
    - n_cycles (float): Number of cycles in the Morlet wavelet.
    - output (str, optional): Output type ('complex', 'power', or 'phase'). Default is 'complex'.
    - n_jobs (int, optional): Number of jobs to run in parallel. Default is 1.

    Returns:
    - np.ndarray: The Morlet transform of the data, shape (n_times, n_channels).
    """
    return mne.time_frequency.tfr_array_morlet(
        data.T[np.newaxis, :, :], sfreq=sfreq, freqs=[freq], n_cycles=n_cycles, output=output, n_jobs=n_jobs
    ).squeeze().T


def find_sphere_center(points):
    """
    Find the center of a sphere that minimizes the standard deviation of the distances
    from the given points to the center.

    Parameters:
    points (numpy.ndarray): An Nx3 matrix where each row represents a point in 3D space.

    Returns:
    numpy.ndarray: The optimal center of the sphere.
    """
    # Define the objective function
    def objective(center, points):
        # Calculate the distances from each point to the center
        distances = np.linalg.norm(points - center, axis=1)
        # Calculate the standard deviation of these distances
        std_dev = np.std(distances)
        return std_dev

    # Initial guess for the center (could be the centroid of the points)
    initial_guess = points.mean(axis=0)

    # Use scipy.optimize.minimize to find the center that minimizes the standard deviation of distances
    result = minimize(objective, initial_guess, args=(points,))

    # Extract the optimal center
    optimal_center = result.x

    return optimal_center


def interpolate_deriv(R, dR, eps = 0.01, **rbf_kwargs):
    """
    Compute the derivative of the radial basis function interpolant at the points R

    Parameters:
    - R: The points at which the derivative is computed.
    - dR: The direction in which the derivative is computed.
    - eps: The step size for the finite difference.
    - rbf_kwargs: Keyword arguments for the RBFInterpolator.

    Returns:
    - D: The derivative of the interpolant at the points R.
    """
    N, _ = R.shape
    eye = np.eye(N)
    D = np.zeros((N,N))
    for n in range(N):
        rbfi = RBFInterpolator(R, eye[n], **rbf_kwargs)
        R1 = R + eps * dR
        R2 = R - eps * dR
        D[:,n] = (rbfi(R1) - rbfi(R2)) / (2*eps)
    return D


def interpolate_deriv2(R, dR, eps = 0.01, **rbf_kwargs):
    """
    Compute the derivative of the radial basis function interpolant at the points R

    Parameters:
    - R: The points at which the derivative is computed.
    - dR: The direction in which the derivative is computed.
    - eps: The step size for the finite difference.
    - rbf_kwargs: Keyword arguments for the RBFInterpolator.

    Returns:
    - D: The derivative of the interpolant at the points R.
    """
    tri = Delaunay(R)
    interpolator = LinearNDInterpolator(tri, np.eye(len(R)))

    N, _ = R.shape
    eye = np.eye(N)
    D = np.zeros((N,N))
    for n in range(N):
        lndi = LinearNDInterpolator(tri, eye[n])
        R1 = R + eps * dR
        R2 = R - eps * dR
        D[:,n] = (lndi(R1) - lndi(R2)) / (2*eps)
    return D


def gram_schmidt_complex(vectors):
    """
    Perform Gram-Schmidt orthogonalization on a set of complex vectors.
    
    Parameters:
    vectors (ndarray): An array of complex vectors to be orthogonalized, 
                       where each column represents a vector.
    
    Returns:
    ndarray: An array of orthogonalized complex vectors.
    """
    num_vectors = vectors.shape[1]
    orthogonal_vectors = np.zeros_like(vectors, dtype=np.complex_)
    
    for i in range(num_vectors):
        # Start with the current vector
        vec = vectors[:, i]
        
        # Subtract the projections onto the previously orthogonalized vectors
        for j in range(i):
            proj = np.vdot(orthogonal_vectors[:, j], vec) / np.vdot(orthogonal_vectors[:, j], orthogonal_vectors[:, j])
            vec -= proj * orthogonal_vectors[:, j]
        
        # Normalize the resulting vector
        norm = np.linalg.norm(vec)
        if norm > 1e-10:  # Avoid division by zero for zero or near-zero vectors
            orthogonal_vectors[:, i] = vec / norm
    

def compute_eigenvalues_and_vectors(matrix):
    # Check the number of dimensions
    if matrix.ndim == 2:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        idx = eigenvalues.real.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors
    elif matrix.ndim == 3:
        
        # Initialize lists to store eigenvalues and eigenvectors
        eigenvalues_list = []
        eigenvectors_list = []
        
        # Compute eigenvalues and eigenvectors for each matrix
        for i in range(matrix.shape[2]):
            eigenvalues, eigenvectors = compute_eigenvalues_and_vectors(matrix[:, :, i])
            eigenvalues_list.append(eigenvalues)
            eigenvectors_list.append(eigenvectors)
        
        # Convert lists to numpy arrays
        eigenvalues_array = np.array(eigenvalues_list).T
        eigenvectors_array = np.array(eigenvectors_list)
        eigenvectors_array = np.moveaxis(eigenvectors_array, 0, -1)
        
        return eigenvalues_array, eigenvectors_array
    else:
        raise ValueError("The input must be either a 2D square matrix or a 3D array of square matrices.")


def phase_to_k_matrix(x, y):
    """
    Compute transformation matrix T for k-vector estimation from 2D points.
    
    Uses Delaunay triangulation to create edges, then solves the least-squares problem
    T = (R^T R)^{-1} R^T D to estimate spatial gradients from scalar field values.
    
    Parameters
    ----------
    x : array-like
        X coordinates of the points.
    y : array-like  
        Y coordinates of the points.
        
    Returns
    -------
    T : np.ndarray, shape (2, N)
        Transformation matrix where T @ theta gives [kx, ky] gradient components.
        Use as: kx, ky = T @ theta where theta is the scalar field at each point.
        
    Notes
    -----
    This is a bare-bones implementation that assumes well-behaved data with no
    edge wrapping issues. The method creates a Delaunay triangulation, extracts
    unique edges, and solves for the optimal linear transformation that maps
    scalar values to gradient estimates.
    
    Examples
    --------
    >>> T = T_from_points_bare(x, y)
    >>> kx, ky = T @ theta  # theta: scalar field values at each point
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(x)
    pts = np.column_stack([x, y])
    tri = Delaunay(pts)

    # unique undirected edges from triangles
    E = set()
    for s in tri.simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            i, j = (a, b) if a < b else (b, a)
            E.add((i, j))
    E = list(E)
    M = len(E)

    # Build D (M x N) and R (M x 2)
    rows, cols, data = [], [], []
    R = np.empty((M, 2), dtype=float)
    for e, (i, j) in enumerate(E):
        rows += [e, e]
        cols += [i, j]
        data += [1.0, -1.0]
        R[e, 0] = x[i] - x[j]
        R[e, 1] = y[i] - y[j]
    D = coo_matrix((data, (rows, cols)), shape=(M, N)).toarray()

    # T = (R^T R)^{-1} R^T D
    A = R.T @ R             # 2x2
    T = np.linalg.solve(A, R.T @ D)  # 2xN
    return T
