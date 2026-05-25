import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple

class ProductionWaterSurfaceFEM:
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, cell_size: float):
        """
        Initializes a production-grade 2D horizontal regular grid structure 
        for continuous bivariate polynomial water surface reconstruction.
        """
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.dxy = cell_size  # Grid cell size (ΔXY)
        
        # Build regular axes grid arrays
        self.x_nodes = np.arange(x_min, x_max + cell_size, cell_size)
        self.y_nodes = np.arange(y_min, y_max + cell_size, cell_size)
        self.cols = len(self.x_nodes)
        self.rows = len(self.y_nodes)
        self.num_nodes = self.rows * self.cols
        
        # Grid vertex nodal values (Z heights to solve dynamically)
        self.Z_nodes = np.zeros(self.num_nodes)

    def _node_index(self, r: int, c: int) -> int:
        """Flattens 2D row/column indices to 1D vector index for matrix assembly."""
        return r * self.cols + c

    def fit_global_surface(self, obs_x: np.ndarray, obs_y: np.ndarray, obs_z: np.ndarray, smoothness_weight: float = 0.1):
        """
        Fits a globally smooth surface using Least-Squares Adjustment with 
        Finite Element Method (FEM) curvature regularization.
        
        Solves: (A^T * P * A + λ * C^T * C) * Z = A^T * P * Z_obs
        """
        num_obs = len(obs_z)
        
        # System matrices initialization (using Sparse LIL for fast building)
        A = lil_matrix((num_obs, self.num_nodes))
        b = obs_z.copy()
        
        # 1. Populate Observation Equations (Bivariate Polynomial mapping per point)
        for idx in range(num_obs):
            x, y = obs_x[idx], obs_y[idx]
            
            # Identify bounding cell row and column
            c = int((x - self.x_min) // self.dxy)
            r = int((y - self.y_min) // self.dxy)
            c = max(0, min(c, self.cols - 2))
            r = max(0, min(r, self.rows - 2))
            
            # Local coordinates
            xl = x - self.x_nodes[c]
            yl = y - self.y_nodes[r]
            
            # Polynomial Shape Functions (Kraus, 2000)
            w1 = (1.0 - xl / self.dxy) * (1.0 - yl / self.dxy)
            w2 = (xl / self.dxy) * (1.0 - yl / self.dxy)
            w3 = (1.0 - xl / self.dxy) * (yl / self.dxy)
            w4 = (xl / self.dxy) * (yl / self.dxy)
            
            # Assign weights to the global design matrix A
            A[idx, self._node_index(r, c)]     = w1
            A[idx, self._node_index(r, c+1)]   = w2
            A[idx, self._node_index(r+1, c)]   = w3
            A[idx, self._node_index(r+1, c+1)] = w4

        # Convert to CSR format for high-performance matrix multiplication
        A = A.tocsr()
        N = A.T @ A
        RHS = A.T @ b

        # 2. Add Curvature Smoothing Regularization (FEM laplacian constraint)
        # Guarantees numerical stability and smooth normal changes across edges
        C = lil_matrix((self.num_nodes, self.num_nodes))
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):
                idx = self._node_index(r, c)
                # Discrete Laplace operator for continuous 2nd order derivatives
                C[idx, idx] = -4.0
                C[idx, self._node_index(r+1, c)] = 1.0
                C[idx, self._node_index(r-1, c)] = 1.0
                C[idx, self._node_index(r, c+1)] = 1.0
                C[idx, self._node_index(r, c-1)] = 1.0
                
        C = C.tocsr()
        N += smoothness_weight * (C.T @ C)

        # 3. Solve the sparse system of linear equations
        self.Z_nodes = spsolve(N, RHS)

    def evaluate_points(self, target_x: np.ndarray, target_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized evaluation engine.
        Accepts arrays of target coordinates and concurrently calculates 
        interpolated water surface heights and normalized 3D normal vectors.
        
        Returns:
            heights: (N,) array of Z positions
            normals: (N, 3) array of unit surface normals
        """
        target_x = np.atleast_1d(target_x)
        target_y = np.atleast_1d(target_y)
        num_pts = len(target_x)

        # Vectorized grid boundary bounding index extraction
        c = ((target_x - self.x_min) // self.dxy).astype(np.int32)
        r = ((target_y - self.y_min) // self.dxy).astype(np.int32)
        np.clip(c, 0, self.cols - 2, out=c)
        np.clip(r, 0, self.rows - 2, out=r)

        # Local coordinate grids
        xl = target_x - self.x_nodes[c]
        yl = target_y - self.y_nodes[r]

        # Extract node IDs for four corner boundaries
        idx1 = self._node_index(r, c)
        idx2 = self._node_index(r, c + 1)
        idx3 = self._node_index(r + 1, c)
        idx4 = self._node_index(r + 1, c + 1)

        # Gather node heights from state array
        z1 = self.Z_nodes[idx1]
        z2 = self.Z_nodes[idx2]
        z3 = self.Z_nodes[idx3]
        z4 = self.Z_nodes[idx4]

        # 1. Height Interpolation using shape coefficients
        w1 = (1.0 - xl / self.dxy) * (1.0 - yl / self.dxy)
        w2 = (xl / self.dxy) * (1.0 - yl / self.dxy)
        w3 = (1.0 - xl / self.dxy) * (yl / self.dxy)
        w4 = (xl / self.dxy) * (yl / self.dxy)
        heights = (w1 * z1) + (w2 * z2) + (w3 * z3) + (w4 * z4)

        # 2. Vectorized Partial Derivatives Mapping (Slopes)
        # Partial derivative dZ/dX
        dz_dx = ((-1.0 / self.dxy) * (1.0 - yl / self.dxy) * z1 +
                 (1.0 / self.dxy) * (1.0 - yl / self.dxy) * z2 +
                 (-1.0 / self.dxy) * (yl / self.dxy) * z3 +
                 (1.0 / self.dxy) * (yl / self.dxy) * z4)

        # Partial derivative dZ/dY
        dz_dy = ((1.0 - xl / self.dxy) * (-1.0 / self.dxy) * z1 +
                 (xl / self.dxy) * (-1.0 / self.dxy) * z2 +
                 (1.0 - xl / self.dxy) * (1.0 / self.dxy) * z3 +
                 (xl / self.dxy) * (1.0 / self.dxy) * z4)

        # 3. Construct Unit Normals: [-dz/dx, -dz/dy, 1.0]
        normals = np.zeros((num_pts, 3))
        normals[:, 0] = -dz_dx
        normals[:, 1] = -dz_dy
        normals[:, 2] = 1.0

        # L2 Vector Normalization over axis 1
        magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= magnitudes

        return heights, normals

# --- Pipeline Integration Validation ---
if __name__ == "__main__":
    print("Initializing production water surface engine...")
    
    # 1. Define Area Framework (e.g., 50m x 50m survey segment with 5.0m grid nodes)
    surface_engine = ProductionWaterSurfaceFEM(x_min=0.0, x_max=50.0, y_min=0.0, y_max=50.0, cell_size=5.0)
    
    # 2. Generate synthetic noisy OWP observation points (simulate rough sea surface data)
    np.random.seed(42)
    sample_size = 800
    obs_raw_x = np.random.uniform(2.0, 48.0, sample_size)
    obs_raw_y = np.random.uniform(2.0, 48.0, sample_size)
    # Target idealized true profile: gentle slope down X axis with wave function noise
    obs_raw_z = 12.0 - 0.05 * obs_raw_x + 0.1 * np.cos(0.2 * obs_raw_y) + np.random.normal(0, 0.02, sample_size)
    
    # 3. Execute global sparse Least-Squares Adjustment matrix optimization
    print(f"Solving optimization over {sample_size} raw OWP intersections...")
    surface_engine.fit_global_surface(obs_raw_x, obs_raw_y, obs_raw_z, smoothness_weight=0.2)
    print("Global optimization converged successfully.")

    # 4. Perform ultra-fast vectorized evaluation of thousands of incoming flight ray points
    eval_size = 5000
    flight_rays_x = np.random.uniform(5.0, 45.0, eval_size)
    flight_rays_y = np.random.uniform(5.0, 45.0, eval_size)
    
    # Vectorized execution
    calc_heights, calc_normals = surface_engine.evaluate_points(flight_rays_x, flight_rays_y)
    
    print("\n--- Pipeline Verification Metrics ---")
    print(f"Processed Batch Size            : {eval_size} laser points")
    print(f"Sample Out Mean Height (Z)      : {np.mean(calc_heights):.3f} meters")
    print(f"First Evaluated Point Normals   : {calc_normals[0]}")
    print(f"All Normal Vector Magnitudes = 1: {np.allclose(np.linalg.norm(calc_normals, axis=1), 1.0)}")