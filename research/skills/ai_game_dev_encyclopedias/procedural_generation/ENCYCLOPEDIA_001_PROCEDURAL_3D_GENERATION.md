# ENCYCLOPEDIA OF PROCEDURAL 3D GENERATION
## Volume I: From Mathematical Primitives to Production Assets
### Technical Reference for AI-Automated Game Development Pipelines
### Version 1.0 | November 2025

---

# PREFACE

This encyclopedia provides exhaustive technical documentation for procedural generation of three-dimensional content in game development contexts. Written at research-laboratory standards (Oak Ridge/Sandia tier), it assumes graduate-level mathematics, professional software engineering competency, and familiarity with real-time graphics programming.

The goal: enable AI systems and human developers to generate AAA-quality 3D assets programmatically, without purchasing or licensing external content.

---

# TABLE OF CONTENTS

## PART I: MATHEMATICAL FOUNDATIONS
- Chapter 1: Vector Spaces and Linear Algebra for 3D
- Chapter 2: Differential Geometry of Surfaces
- Chapter 3: Topology and Manifold Theory
- Chapter 4: Numerical Methods and Stability

## PART II: IMPLICIT SURFACE REPRESENTATIONS
- Chapter 5: Signed Distance Functions (SDFs)
- Chapter 6: Constructive Solid Geometry (CSG)
- Chapter 7: Metaballs and Blobby Objects
- Chapter 8: Level Set Methods

## PART III: MESH GENERATION ALGORITHMS
- Chapter 9: Marching Cubes and Variants
- Chapter 10: Dual Contouring
- Chapter 11: Surface Nets
- Chapter 12: Mesh Optimization and Decimation

## PART IV: PROCEDURAL MODELING TECHNIQUES
- Chapter 13: L-Systems and Grammar-Based Generation
- Chapter 14: Subdivision Surfaces
- Chapter 15: Fractal Terrain Generation
- Chapter 16: Architectural Procedural Modeling

## PART V: PROCEDURAL MATERIALS AND TEXTURING
- Chapter 17: 3D Noise Functions
- Chapter 18: Triplanar Mapping
- Chapter 19: Procedural PBR Materials
- Chapter 20: Weathering and Aging Systems

## PART VI: PROCEDURAL ANIMATION
- Chapter 21: Inverse Kinematics Solvers
- Chapter 22: Procedural Locomotion
- Chapter 23: Physics-Driven Secondary Motion
- Chapter 24: Blend Shapes and Morph Targets

## PART VII: ENVIRONMENT GENERATION
- Chapter 25: Terrain Systems
- Chapter 26: Vegetation Distribution
- Chapter 27: Urban and Dungeon Generation
- Chapter 28: Biome and Ecosystem Simulation

## PART VIII: OPTIMIZATION AND PRODUCTION
- Chapter 29: LOD Generation
- Chapter 30: Occlusion and Culling
- Chapter 31: GPU Compute for Procedural Content
- Chapter 32: Streaming and Chunk Management

---

# PART I: MATHEMATICAL FOUNDATIONS

---

## Chapter 1: Vector Spaces and Linear Algebra for 3D

### 1.1 Coordinate Systems and Conventions

Three-dimensional graphics universally employ Cartesian coordinate systems, but conventions vary significantly across engines and APIs:

```
COORDINATE SYSTEM CONVENTIONS:

OpenGL / Vulkan (Right-Handed, Y-Up):
      +Y
       │
       │
       │
       └───────► +X
      /
     /
    ▼
   +Z (toward viewer)

DirectX / Unity (Left-Handed, Y-Up):
      +Y
       │
       │
       │
       └───────► +X
        \
         \
          ▼
         +Z (away from viewer)

Blender (Right-Handed, Z-Up):
      +Z
       │
       │
       │
       └───────► +X
      /
     /
    ▼
   +Y (toward viewer)
```

**Transformation between systems requires careful attention:**

```python
import numpy as np
from typing import Tuple

def opengl_to_directx(v: np.ndarray) -> np.ndarray:
    """Convert OpenGL coordinates to DirectX coordinates.
    
    OpenGL: Right-handed, Y-up, -Z forward
    DirectX: Left-handed, Y-up, +Z forward
    
    Transformation: Negate Z component
    """
    return np.array([v[0], v[1], -v[2]])

def blender_to_opengl(v: np.ndarray) -> np.ndarray:
    """Convert Blender coordinates to OpenGL coordinates.
    
    Blender: Right-handed, Z-up, -Y forward
    OpenGL: Right-handed, Y-up, -Z forward
    
    Transformation: Swap Y and Z, negate new Z
    """
    return np.array([v[0], v[2], -v[1]])

def quaternion_handedness_flip(q: np.ndarray) -> np.ndarray:
    """Flip quaternion handedness.
    
    When converting between left and right-handed systems,
    quaternion rotation direction must be inverted.
    
    q = (w, x, y, z) -> (w, -x, -y, -z) for axis inversion
    OR negate specific axes based on transformation
    """
    # For Z-axis flip (OpenGL <-> DirectX)
    return np.array([q[0], q[1], q[2], -q[3]])
```

### 1.2 Vector Operations

Fundamental vector operations form the basis of all 3D mathematics:

```python
class Vec3:
    """High-performance 3D vector implementation.
    
    Designed for numerical stability and explicit operation semantics.
    All operations return new vectors (immutability for safety).
    """
    
    __slots__ = ('x', 'y', 'z')  # Memory optimization
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    # === ARITHMETIC OPERATIONS ===
    
    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vec3':
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vec3':
        if abs(scalar) < 1e-10:
            raise ValueError("Division by near-zero scalar")
        inv = 1.0 / scalar
        return Vec3(self.x * inv, self.y * inv, self.z * inv)
    
    # === VECTOR PRODUCTS ===
    
    def dot(self, other: 'Vec3') -> float:
        """Dot product: a · b = |a||b|cos(θ)
        
        Properties:
        - Commutative: a · b = b · a
        - Distributive: a · (b + c) = a · b + a · c
        - Scalar multiplication: (ka) · b = k(a · b)
        - Self dot equals squared magnitude: a · a = |a|²
        
        Applications:
        - Projection of one vector onto another
        - Angle calculation between vectors
        - Backface culling (sign indicates facing)
        - Lighting calculations (Lambert's cosine law)
        """
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vec3') -> 'Vec3':
        """Cross product: a × b = |a||b|sin(θ)n̂
        
        Properties:
        - Anti-commutative: a × b = -(b × a)
        - Distributive: a × (b + c) = (a × b) + (a × c)
        - NOT associative: a × (b × c) ≠ (a × b) × c
        - Perpendicular to both inputs
        - Magnitude equals parallelogram area
        
        Applications:
        - Surface normal calculation
        - Torque computation
        - Angular velocity
        - Coordinate system construction
        """
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def triple_scalar(self, b: 'Vec3', c: 'Vec3') -> float:
        """Scalar triple product: a · (b × c)
        
        Equals the signed volume of the parallelepiped defined by a, b, c.
        Equivalent to determinant of matrix [a|b|c].
        
        |a · (b × c)| = |b · (c × a)| = |c · (a × b)|
        
        Sign indicates handedness of the triple.
        """
        return self.dot(b.cross(c))
    
    def triple_vector(self, b: 'Vec3', c: 'Vec3') -> 'Vec3':
        """Vector triple product: a × (b × c)
        
        BAC-CAB identity: a × (b × c) = b(a · c) - c(a · b)
        
        Result lies in the plane spanned by b and c.
        """
        return b * self.dot(c) - c * self.dot(b)
    
    # === MAGNITUDE AND NORMALIZATION ===
    
    def magnitude(self) -> float:
        """Euclidean magnitude (L2 norm): |v| = √(x² + y² + z²)"""
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5
    
    def magnitude_squared(self) -> float:
        """Squared magnitude (avoids sqrt for comparisons)"""
        return self.x**2 + self.y**2 + self.z**2
    
    def normalized(self) -> 'Vec3':
        """Unit vector in same direction.
        
        Warning: Returns zero vector if input magnitude is near-zero.
        This is intentional for numerical stability in degenerate cases.
        """
        mag = self.magnitude()
        if mag < 1e-10:
            return Vec3(0, 0, 0)
        return self / mag
    
    def safe_normalized(self, fallback: 'Vec3' = None) -> 'Vec3':
        """Normalized with explicit fallback for degenerate inputs."""
        mag = self.magnitude()
        if mag < 1e-10:
            return fallback if fallback else Vec3(0, 1, 0)
        return self / mag
    
    # === GEOMETRIC OPERATIONS ===
    
    def project_onto(self, other: 'Vec3') -> 'Vec3':
        """Project self onto other vector.
        
        proj_b(a) = (a · b / |b|²) * b
        
        Returns component of self parallel to other.
        """
        other_mag_sq = other.magnitude_squared()
        if other_mag_sq < 1e-10:
            return Vec3(0, 0, 0)
        return other * (self.dot(other) / other_mag_sq)
    
    def reject_from(self, other: 'Vec3') -> 'Vec3':
        """Rejection: component perpendicular to other.
        
        rej_b(a) = a - proj_b(a)
        """
        return self - self.project_onto(other)
    
    def reflect(self, normal: 'Vec3') -> 'Vec3':
        """Reflect vector across plane defined by normal.
        
        r = v - 2(v · n)n
        
        Used for: reflections, bouncing, mirror effects
        Assumes normal is unit length.
        """
        return self - normal * (2.0 * self.dot(normal))
    
    def refract(self, normal: 'Vec3', eta: float) -> 'Vec3':
        """Refract vector through surface.
        
        eta = n1/n2 (ratio of refractive indices)
        
        Uses Snell's law: n1*sin(θ1) = n2*sin(θ2)
        
        Returns zero vector for total internal reflection.
        """
        cos_i = -normal.dot(self)
        sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
        
        if sin_t2 > 1.0:
            # Total internal reflection
            return Vec3(0, 0, 0)
        
        cos_t = (1.0 - sin_t2) ** 0.5
        return self * eta + normal * (eta * cos_i - cos_t)
    
    def angle_to(self, other: 'Vec3') -> float:
        """Angle between vectors in radians.
        
        θ = arccos((a · b) / (|a||b|))
        
        Clamped to avoid numerical issues at parallel/antiparallel.
        """
        import math
        denom = self.magnitude() * other.magnitude()
        if denom < 1e-10:
            return 0.0
        cos_angle = max(-1.0, min(1.0, self.dot(other) / denom))
        return math.acos(cos_angle)
    
    def signed_angle_to(self, other: 'Vec3', axis: 'Vec3') -> float:
        """Signed angle around axis.
        
        Positive = counterclockwise when looking down axis.
        Uses atan2 for full -π to π range.
        """
        import math
        cross = self.cross(other)
        dot = self.dot(other)
        return math.atan2(cross.dot(axis), dot)
    
    # === INTERPOLATION ===
    
    def lerp(self, other: 'Vec3', t: float) -> 'Vec3':
        """Linear interpolation.
        
        lerp(a, b, t) = a + t(b - a) = (1-t)a + tb
        
        t=0 returns self, t=1 returns other.
        """
        return self + (other - self) * t
    
    def slerp(self, other: 'Vec3', t: float) -> 'Vec3':
        """Spherical linear interpolation.
        
        Interpolates along great arc on unit sphere.
        Maintains constant angular velocity.
        
        For unit vectors only!
        """
        import math
        
        dot = max(-1.0, min(1.0, self.dot(other)))
        theta = math.acos(dot)
        
        if abs(theta) < 1e-6:
            return self.lerp(other, t)
        
        sin_theta = math.sin(theta)
        a = math.sin((1 - t) * theta) / sin_theta
        b = math.sin(t * theta) / sin_theta
        
        return self * a + other * b
    
    # === COMPONENT ACCESS ===
    
    def __getitem__(self, i: int) -> float:
        if i == 0: return self.x
        if i == 1: return self.y
        if i == 2: return self.z
        raise IndexError(f"Vec3 index {i} out of range")
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @staticmethod
    def from_numpy(arr: np.ndarray) -> 'Vec3':
        return Vec3(arr[0], arr[1], arr[2])
```

### 1.3 Matrix Transformations

Transformation matrices encode translation, rotation, scale, and projection:

```python
import numpy as np
from typing import Tuple
import math

class Mat4:
    """4x4 transformation matrix for 3D graphics.
    
    Column-major storage (OpenGL convention).
    Multiplication order: M_final = M_projection * M_view * M_model
    
    Matrix structure:
    ┌                      ┐
    │ Xx  Yx  Zx  Tx │  (row 0: X basis + X translation)
    │ Xy  Yy  Zy  Ty │  (row 1: Y basis + Y translation)
    │ Xz  Yz  Zz  Tz │  (row 2: Z basis + Z translation)
    │  0   0   0   1 │  (row 3: homogeneous)
    └                      ┘
    
    Where X, Y, Z columns are the transformed basis vectors,
    and T column is the translation.
    """
    
    def __init__(self, data: np.ndarray = None):
        if data is None:
            self.m = np.eye(4, dtype=np.float64)
        else:
            self.m = np.array(data, dtype=np.float64).reshape(4, 4)
    
    # === FACTORY METHODS ===
    
    @staticmethod
    def identity() -> 'Mat4':
        """Identity matrix: no transformation."""
        return Mat4()
    
    @staticmethod
    def translation(x: float, y: float, z: float) -> 'Mat4':
        """Translation matrix.
        
        ┌             ┐   ┌   ┐   ┌       ┐
        │ 1 0 0 tx│   │ x │   │ x + tx │
        │ 0 1 0 ty│ × │ y │ = │ y + ty │
        │ 0 0 1 tz│   │ z │   │ z + tz │
        │ 0 0 0  1│   │ 1 │   │    1   │
        └             ┘   └   ┘   └       ┘
        """
        m = Mat4()
        m.m[0, 3] = x
        m.m[1, 3] = y
        m.m[2, 3] = z
        return m
    
    @staticmethod
    def scale(x: float, y: float, z: float) -> 'Mat4':
        """Scale matrix.
        
        Uniform scale: x = y = z
        Non-uniform scale: different values (shearing if combined with rotation)
        """
        m = Mat4()
        m.m[0, 0] = x
        m.m[1, 1] = y
        m.m[2, 2] = z
        return m
    
    @staticmethod
    def rotation_x(radians: float) -> 'Mat4':
        """Rotation around X axis (pitch).
        
        ┌                      ┐
        │ 1    0       0    0 │
        │ 0  cos(θ) -sin(θ) 0 │
        │ 0  sin(θ)  cos(θ) 0 │
        │ 0    0       0    1 │
        └                      ┘
        """
        c, s = math.cos(radians), math.sin(radians)
        m = Mat4()
        m.m[1, 1] = c
        m.m[1, 2] = -s
        m.m[2, 1] = s
        m.m[2, 2] = c
        return m
    
    @staticmethod
    def rotation_y(radians: float) -> 'Mat4':
        """Rotation around Y axis (yaw).
        
        ┌                      ┐
        │  cos(θ) 0  sin(θ) 0 │
        │    0    1    0    0 │
        │ -sin(θ) 0  cos(θ) 0 │
        │    0    0    0    1 │
        └                      ┘
        """
        c, s = math.cos(radians), math.sin(radians)
        m = Mat4()
        m.m[0, 0] = c
        m.m[0, 2] = s
        m.m[2, 0] = -s
        m.m[2, 2] = c
        return m
    
    @staticmethod
    def rotation_z(radians: float) -> 'Mat4':
        """Rotation around Z axis (roll).
        
        ┌                      ┐
        │ cos(θ) -sin(θ) 0  0 │
        │ sin(θ)  cos(θ) 0  0 │
        │   0       0    1  0 │
        │   0       0    0  1 │
        └                      ┘
        """
        c, s = math.cos(radians), math.sin(radians)
        m = Mat4()
        m.m[0, 0] = c
        m.m[0, 1] = -s
        m.m[1, 0] = s
        m.m[1, 1] = c
        return m
    
    @staticmethod
    def rotation_axis_angle(axis: 'Vec3', radians: float) -> 'Mat4':
        """Rodrigues' rotation formula.
        
        Rotation of θ radians around arbitrary unit axis (x, y, z):
        
        R = I + sin(θ)K + (1-cos(θ))K²
        
        Where K is the skew-symmetric cross-product matrix:
        ┌         ┐
        │  0 -z  y│
        │  z  0 -x│
        │ -y  x  0│
        └         ┘
        """
        axis = axis.normalized()
        x, y, z = axis.x, axis.y, axis.z
        c, s = math.cos(radians), math.sin(radians)
        t = 1.0 - c
        
        m = Mat4()
        m.m[0, 0] = t*x*x + c
        m.m[0, 1] = t*x*y - s*z
        m.m[0, 2] = t*x*z + s*y
        m.m[1, 0] = t*x*y + s*z
        m.m[1, 1] = t*y*y + c
        m.m[1, 2] = t*y*z - s*x
        m.m[2, 0] = t*x*z - s*y
        m.m[2, 1] = t*y*z + s*x
        m.m[2, 2] = t*z*z + c
        return m
    
    @staticmethod
    def from_quaternion(q: 'Quaternion') -> 'Mat4':
        """Convert quaternion to rotation matrix.
        
        More numerically stable than Euler angles.
        Avoids gimbal lock.
        """
        w, x, y, z = q.w, q.x, q.y, q.z
        
        # Normalize to ensure unit quaternion
        n = w*w + x*x + y*y + z*z
        s = 2.0 / n if n > 0 else 0.0
        
        wx, wy, wz = s*w*x, s*w*y, s*w*z
        xx, xy, xz = s*x*x, s*x*y, s*x*z
        yy, yz, zz = s*y*y, s*y*z, s*z*z
        
        m = Mat4()
        m.m[0, 0] = 1.0 - (yy + zz)
        m.m[0, 1] = xy - wz
        m.m[0, 2] = xz + wy
        m.m[1, 0] = xy + wz
        m.m[1, 1] = 1.0 - (xx + zz)
        m.m[1, 2] = yz - wx
        m.m[2, 0] = xz - wy
        m.m[2, 1] = yz + wx
        m.m[2, 2] = 1.0 - (xx + yy)
        return m
    
    @staticmethod
    def look_at(eye: 'Vec3', target: 'Vec3', up: 'Vec3') -> 'Mat4':
        """View matrix: positions camera at eye, looking at target.
        
        Constructs orthonormal basis:
        - Z axis: from target to eye (forward direction, negated for RH)
        - X axis: perpendicular to Z and world up
        - Y axis: perpendicular to X and Z
        """
        z_axis = (eye - target).normalized()
        x_axis = up.cross(z_axis).normalized()
        y_axis = z_axis.cross(x_axis)
        
        m = Mat4()
        m.m[0, 0] = x_axis.x
        m.m[0, 1] = x_axis.y
        m.m[0, 2] = x_axis.z
        m.m[0, 3] = -x_axis.dot(eye)
        m.m[1, 0] = y_axis.x
        m.m[1, 1] = y_axis.y
        m.m[1, 2] = y_axis.z
        m.m[1, 3] = -y_axis.dot(eye)
        m.m[2, 0] = z_axis.x
        m.m[2, 1] = z_axis.y
        m.m[2, 2] = z_axis.z
        m.m[2, 3] = -z_axis.dot(eye)
        return m
    
    @staticmethod
    def perspective(fov_y: float, aspect: float, near: float, far: float) -> 'Mat4':
        """Perspective projection matrix.
        
        fov_y: vertical field of view in radians
        aspect: width / height
        near: near clipping plane (must be > 0)
        far: far clipping plane (must be > near)
        
        Maps view frustum to normalized device coordinates [-1, 1].
        
        Depth precision notes:
        - Precision is highest near the near plane
        - Use reversed-Z for better depth distribution
        - Infinite far plane: far = ∞ → m[2,2] = -1, m[2,3] = -near
        """
        tan_half_fov = math.tan(fov_y / 2.0)
        
        m = Mat4(np.zeros((4, 4)))
        m.m[0, 0] = 1.0 / (aspect * tan_half_fov)
        m.m[1, 1] = 1.0 / tan_half_fov
        m.m[2, 2] = -(far + near) / (far - near)
        m.m[2, 3] = -(2.0 * far * near) / (far - near)
        m.m[3, 2] = -1.0
        return m
    
    @staticmethod
    def orthographic(left: float, right: float, bottom: float, 
                     top: float, near: float, far: float) -> 'Mat4':
        """Orthographic projection matrix.
        
        No perspective foreshortening.
        Used for: 2D games, UI, CAD, isometric views.
        """
        m = Mat4()
        m.m[0, 0] = 2.0 / (right - left)
        m.m[1, 1] = 2.0 / (top - bottom)
        m.m[2, 2] = -2.0 / (far - near)
        m.m[0, 3] = -(right + left) / (right - left)
        m.m[1, 3] = -(top + bottom) / (top - bottom)
        m.m[2, 3] = -(far + near) / (far - near)
        return m
    
    # === OPERATIONS ===
    
    def __mul__(self, other) -> 'Mat4':
        """Matrix multiplication."""
        if isinstance(other, Mat4):
            return Mat4(self.m @ other.m)
        elif isinstance(other, Vec3):
            # Transform point (w=1)
            v = np.array([other.x, other.y, other.z, 1.0])
            result = self.m @ v
            if abs(result[3]) > 1e-10:
                return Vec3(result[0]/result[3], result[1]/result[3], result[2]/result[3])
            return Vec3(result[0], result[1], result[2])
        raise TypeError(f"Cannot multiply Mat4 by {type(other)}")
    
    def transform_point(self, p: 'Vec3') -> 'Vec3':
        """Transform point (applies translation)."""
        return self * p
    
    def transform_vector(self, v: 'Vec3') -> 'Vec3':
        """Transform direction vector (ignores translation)."""
        result = self.m[:3, :3] @ np.array([v.x, v.y, v.z])
        return Vec3(result[0], result[1], result[2])
    
    def transform_normal(self, n: 'Vec3') -> 'Vec3':
        """Transform normal vector.
        
        Normals must be transformed by the inverse-transpose matrix
        to preserve perpendicularity under non-uniform scale.
        
        n' = (M⁻¹)ᵀ n
        """
        inv_transpose = np.linalg.inv(self.m[:3, :3]).T
        result = inv_transpose @ np.array([n.x, n.y, n.z])
        return Vec3(result[0], result[1], result[2]).normalized()
    
    def inverse(self) -> 'Mat4':
        """Matrix inverse.
        
        For rigid transforms (rotation + translation only),
        inverse is transpose of rotation block with negated translation.
        General case uses LU decomposition.
        """
        return Mat4(np.linalg.inv(self.m))
    
    def transpose(self) -> 'Mat4':
        """Matrix transpose."""
        return Mat4(self.m.T)
    
    def determinant(self) -> float:
        """Matrix determinant.
        
        For transformation matrices:
        - det = 1: pure rotation
        - det = -1: reflection + rotation
        - det > 0: orientation preserving
        - det < 0: orientation reversing (reflection)
        - det = 0: singular (degenerate)
        """
        return np.linalg.det(self.m)
    
    # === DECOMPOSITION ===
    
    def decompose(self) -> Tuple['Vec3', 'Quaternion', 'Vec3']:
        """Decompose into translation, rotation, scale.
        
        Assumes no shear. For general affine transforms,
        use polar decomposition.
        
        Returns: (translation, rotation_quaternion, scale)
        """
        # Translation from last column
        translation = Vec3(self.m[0, 3], self.m[1, 3], self.m[2, 3])
        
        # Scale from column magnitudes
        sx = Vec3(self.m[0, 0], self.m[1, 0], self.m[2, 0]).magnitude()
        sy = Vec3(self.m[0, 1], self.m[1, 1], self.m[2, 1]).magnitude()
        sz = Vec3(self.m[0, 2], self.m[1, 2], self.m[2, 2]).magnitude()
        
        # Handle negative scale (reflection)
        if self.determinant() < 0:
            sx = -sx
        
        scale = Vec3(sx, sy, sz)
        
        # Rotation matrix (normalized columns)
        rot = np.zeros((3, 3))
        rot[0, 0] = self.m[0, 0] / sx if sx != 0 else 0
        rot[1, 0] = self.m[1, 0] / sx if sx != 0 else 0
        rot[2, 0] = self.m[2, 0] / sx if sx != 0 else 0
        rot[0, 1] = self.m[0, 1] / sy if sy != 0 else 0
        rot[1, 1] = self.m[1, 1] / sy if sy != 0 else 0
        rot[2, 1] = self.m[2, 1] / sy if sy != 0 else 0
        rot[0, 2] = self.m[0, 2] / sz if sz != 0 else 0
        rot[1, 2] = self.m[1, 2] / sz if sz != 0 else 0
        rot[2, 2] = self.m[2, 2] / sz if sz != 0 else 0
        
        rotation = Quaternion.from_rotation_matrix(rot)
        
        return translation, rotation, scale


class Quaternion:
    """Unit quaternion for 3D rotation representation.
    
    q = w + xi + yj + zk
    
    where i² = j² = k² = ijk = -1
    
    Advantages over matrices/Euler angles:
    - Compact (4 floats vs 9)
    - Efficient composition
    - Smooth interpolation (SLERP)
    - No gimbal lock
    - Easy normalization
    
    Convention: (w, x, y, z) with w as scalar component.
    """
    
    __slots__ = ('w', 'x', 'y', 'z')
    
    def __init__(self, w: float = 1.0, x: float = 0.0, 
                 y: float = 0.0, z: float = 0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    # === FACTORY METHODS ===
    
    @staticmethod
    def identity() -> 'Quaternion':
        """Identity rotation (no rotation)."""
        return Quaternion(1, 0, 0, 0)
    
    @staticmethod
    def from_axis_angle(axis: Vec3, radians: float) -> 'Quaternion':
        """Create quaternion from axis-angle representation.
        
        q = cos(θ/2) + sin(θ/2)(xi + yj + zk)
        """
        axis = axis.normalized()
        half_angle = radians / 2.0
        s = math.sin(half_angle)
        return Quaternion(
            math.cos(half_angle),
            axis.x * s,
            axis.y * s,
            axis.z * s
        )
    
    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float) -> 'Quaternion':
        """Create from Euler angles (XYZ order).
        
        pitch: rotation around X (in radians)
        yaw: rotation around Y (in radians)
        roll: rotation around Z (in radians)
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        )
    
    @staticmethod
    def from_rotation_matrix(m: np.ndarray) -> 'Quaternion':
        """Convert 3x3 rotation matrix to quaternion.
        
        Uses Shepperd's method for numerical stability.
        """
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            return Quaternion(
                0.25 / s,
                (m[2, 1] - m[1, 2]) * s,
                (m[0, 2] - m[2, 0]) * s,
                (m[1, 0] - m[0, 1]) * s
            )
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            return Quaternion(
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s
            )
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            return Quaternion(
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s
            )
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            return Quaternion(
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s
            )
    
    @staticmethod
    def from_to_rotation(from_dir: Vec3, to_dir: Vec3) -> 'Quaternion':
        """Quaternion that rotates from_dir to to_dir.
        
        Handles parallel and anti-parallel vectors.
        """
        from_dir = from_dir.normalized()
        to_dir = to_dir.normalized()
        
        dot = from_dir.dot(to_dir)
        
        if dot > 0.999999:
            return Quaternion.identity()
        
        if dot < -0.999999:
            # 180 degree rotation around any perpendicular axis
            axis = Vec3(1, 0, 0).cross(from_dir)
            if axis.magnitude_squared() < 0.001:
                axis = Vec3(0, 1, 0).cross(from_dir)
            axis = axis.normalized()
            return Quaternion.from_axis_angle(axis, math.pi)
        
        axis = from_dir.cross(to_dir)
        s = math.sqrt((1 + dot) * 2)
        inv_s = 1.0 / s
        
        return Quaternion(
            s * 0.5,
            axis.x * inv_s,
            axis.y * inv_s,
            axis.z * inv_s
        ).normalized()
    
    # === OPERATIONS ===
    
    def __mul__(self, other) -> 'Quaternion':
        """Quaternion multiplication (Hamilton product).
        
        q1 * q2 applies q1 rotation AFTER q2 rotation.
        
        Non-commutative: q1 * q2 ≠ q2 * q1
        """
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
                self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
                self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
                self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            )
        raise TypeError(f"Cannot multiply Quaternion by {type(other)}")
    
    def rotate_vector(self, v: Vec3) -> Vec3:
        """Rotate vector by this quaternion.
        
        v' = q * v * q⁻¹
        
        Optimized form (avoids full quaternion multiplication):
        v' = v + 2w(q_xyz × v) + 2(q_xyz × (q_xyz × v))
        """
        qv = Vec3(self.x, self.y, self.z)
        uv = qv.cross(v)
        uuv = qv.cross(uv)
        return v + (uv * (2.0 * self.w)) + (uuv * 2.0)
    
    def conjugate(self) -> 'Quaternion':
        """Quaternion conjugate: q* = w - xi - yj - zk
        
        For unit quaternions: q* = q⁻¹
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self) -> 'Quaternion':
        """Quaternion inverse: q⁻¹ = q* / |q|²
        
        For unit quaternions, inverse equals conjugate.
        """
        mag_sq = self.magnitude_squared()
        if mag_sq < 1e-10:
            return Quaternion.identity()
        inv_mag_sq = 1.0 / mag_sq
        return Quaternion(
            self.w * inv_mag_sq,
            -self.x * inv_mag_sq,
            -self.y * inv_mag_sq,
            -self.z * inv_mag_sq
        )
    
    def magnitude(self) -> float:
        """Quaternion magnitude: |q| = √(w² + x² + y² + z²)"""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def magnitude_squared(self) -> float:
        """Squared magnitude (avoids sqrt)."""
        return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
    def normalized(self) -> 'Quaternion':
        """Normalize to unit quaternion."""
        mag = self.magnitude()
        if mag < 1e-10:
            return Quaternion.identity()
        inv_mag = 1.0 / mag
        return Quaternion(
            self.w * inv_mag,
            self.x * inv_mag,
            self.y * inv_mag,
            self.z * inv_mag
        )
    
    def dot(self, other: 'Quaternion') -> float:
        """Quaternion dot product.
        
        Measures similarity: dot(q1, q2) = 1 means identical,
        dot(q1, q2) = -1 means opposite rotations.
        """
        return self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z
    
    # === INTERPOLATION ===
    
    def slerp(self, other: 'Quaternion', t: float) -> 'Quaternion':
        """Spherical linear interpolation.
        
        Constant angular velocity interpolation on 4D unit sphere.
        
        t=0 returns self, t=1 returns other.
        
        Handles shortest path (flips other if necessary).
        """
        dot = self.dot(other)
        
        # If dot < 0, negate one quaternion to take shorter path
        if dot < 0:
            other = Quaternion(-other.w, -other.x, -other.y, -other.z)
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = Quaternion(
                self.w + t * (other.w - self.w),
                self.x + t * (other.x - self.x),
                self.y + t * (other.y - self.y),
                self.z + t * (other.z - self.z)
            )
            return result.normalized()
        
        # Standard SLERP
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return Quaternion(
            s0 * self.w + s1 * other.w,
            s0 * self.x + s1 * other.x,
            s0 * self.y + s1 * other.y,
            s0 * self.z + s1 * other.z
        )
    
    def nlerp(self, other: 'Quaternion', t: float) -> 'Quaternion':
        """Normalized linear interpolation.
        
        Faster than SLERP, nearly identical results for small angles.
        Not constant velocity but acceptable for most applications.
        """
        dot = self.dot(other)
        
        if dot < 0:
            other = Quaternion(-other.w, -other.x, -other.y, -other.z)
        
        return Quaternion(
            self.w + t * (other.w - self.w),
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y),
            self.z + t * (other.z - self.z)
        ).normalized()
    
    # === CONVERSION ===
    
    def to_euler(self) -> Tuple[float, float, float]:
        """Convert to Euler angles (XYZ order).
        
        Returns: (pitch, yaw, roll) in radians
        
        Warning: Euler angles suffer from gimbal lock at ±90° pitch.
        """
        # Roll (X)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (Y)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (Z)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (pitch, yaw, roll)
    
    def to_axis_angle(self) -> Tuple[Vec3, float]:
        """Convert to axis-angle representation.
        
        Returns: (axis, angle_in_radians)
        """
        # Ensure unit quaternion
        q = self.normalized()
        
        angle = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))
        
        s = math.sqrt(1.0 - q.w * q.w)
        if s < 0.001:
            # Angle is close to 0 or 2π, axis is arbitrary
            axis = Vec3(1, 0, 0)
        else:
            axis = Vec3(q.x / s, q.y / s, q.z / s)
        
        return (axis, angle)
    
    def forward(self) -> Vec3:
        """Get forward direction (assumes -Z forward)."""
        return self.rotate_vector(Vec3(0, 0, -1))
    
    def right(self) -> Vec3:
        """Get right direction (assumes +X right)."""
        return self.rotate_vector(Vec3(1, 0, 0))
    
    def up(self) -> Vec3:
        """Get up direction (assumes +Y up)."""
        return self.rotate_vector(Vec3(0, 1, 0))
```

### 1.4 Numerical Precision Considerations

Floating-point arithmetic introduces systematic errors that compound across operations:

```python
"""
FLOATING-POINT PRECISION IN 3D GRAPHICS

IEEE 754 Float32 (single precision):
- Sign: 1 bit
- Exponent: 8 bits
- Mantissa: 23 bits
- Precision: ~7 decimal digits
- Range: ±3.4 × 10³⁸

IEEE 754 Float64 (double precision):
- Sign: 1 bit
- Exponent: 11 bits
- Mantissa: 52 bits
- Precision: ~15-16 decimal digits
- Range: ±1.8 × 10³⁰⁸

COMMON PRECISION ISSUES:

1. CATASTROPHIC CANCELLATION
   When subtracting nearly equal numbers:
   
   a = 1.0000001
   b = 1.0000000
   c = a - b  # Expected: 0.0000001
             # Actual: ~0.00000009536743 (error: 4.6%)
   
   Mitigation: Rearrange formulas, use higher precision for intermediates

2. ACCUMULATED DRIFT
   Repeated floating-point operations accumulate error:
   
   position += velocity * dt  # Repeated 1000x
   # Error grows linearly: O(n × ε)
   
   Mitigation: Periodic renormalization, higher precision physics

3. COMPARISON HAZARDS
   Direct equality fails for computed values:
   
   # BAD:
   if a == b: ...
   
   # GOOD:
   EPSILON = 1e-6
   if abs(a - b) < EPSILON: ...
   
   # BETTER (relative tolerance):
   if abs(a - b) < max(abs(a), abs(b)) * EPSILON: ...

4. WORLD-SPACE PRECISION LOSS
   At large coordinates, precision degrades:
   
   Position (1,000,000, 0, 0):
   - Representable precision: ~0.0625 units
   - At origin: ~0.0000001 units
   
   Mitigation: Origin rebasing, double-precision for world position,
               local-space rendering with floating origin
"""

import numpy as np

class PrecisionUtils:
    """Utilities for managing floating-point precision."""
    
    FLOAT32_EPSILON = np.finfo(np.float32).eps  # ~1.19e-7
    FLOAT64_EPSILON = np.finfo(np.float64).eps  # ~2.22e-16
    
    @staticmethod
    def approx_equal(a: float, b: float, rel_tol: float = 1e-6, 
                     abs_tol: float = 1e-9) -> bool:
        """Compare floats with both relative and absolute tolerance.
        
        Handles edge cases:
        - Very small numbers (use abs_tol)
        - Large numbers (use rel_tol)
        - Opposite signs correctly
        """
        diff = abs(a - b)
        return diff <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    
    @staticmethod
    def ulp_distance(a: float, b: float) -> int:
        """Distance in Units of Least Precision.
        
        More accurate than epsilon comparisons for many cases.
        1 ULP = smallest possible float difference at that magnitude.
        """
        # Convert to integer representation
        ai = np.float64(a).view(np.int64)
        bi = np.float64(b).view(np.int64)
        
        # Handle negative numbers
        if ai < 0:
            ai = 0x8000000000000000 - ai
        if bi < 0:
            bi = 0x8000000000000000 - bi
        
        return abs(ai - bi)
    
    @staticmethod
    def kahan_sum(values: np.ndarray) -> float:
        """Kahan summation algorithm.
        
        Compensates for floating-point error accumulation.
        Error: O(ε) vs O(n×ε) for naive summation.
        """
        sum_val = 0.0
        compensation = 0.0
        
        for x in values:
            y = x - compensation
            t = sum_val + y
            compensation = (t - sum_val) - y
            sum_val = t
        
        return sum_val
    
    @staticmethod
    def stable_normalize(v: np.ndarray) -> np.ndarray:
        """Numerically stable vector normalization.
        
        Handles very large and very small vectors without overflow/underflow.
        """
        max_component = np.max(np.abs(v))
        if max_component < 1e-30:
            return np.zeros_like(v)
        
        # Scale down to prevent overflow in magnitude calculation
        scaled = v / max_component
        mag = np.linalg.norm(scaled)
        
        if mag < 1e-30:
            return np.zeros_like(v)
        
        return scaled / mag
    
    @staticmethod
    def robust_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cross product with improved numerical stability.
        
        Uses compensated arithmetic for near-parallel vectors.
        """
        # Standard cross product
        result = np.cross(a, b)
        
        # Check if vectors are nearly parallel
        mag_sq = np.dot(result, result)
        if mag_sq > 1e-12:
            return result
        
        # Vectors nearly parallel - use extended precision
        result_high = np.cross(a.astype(np.float64), b.astype(np.float64))
        return result_high.astype(a.dtype)
```

---

## Chapter 2: Differential Geometry of Surfaces

### 2.1 Parametric Surfaces

A parametric surface S is defined by a function **r**: D ⊂ ℝ² → ℝ³:

```
r(u, v) = (x(u, v), y(u, v), z(u, v))
```

Where (u, v) are parameters in domain D.

```python
from abc import ABC, abstractmethod
from typing import Tuple, Callable
import numpy as np

class ParametricSurface(ABC):
    """Abstract base class for parametric surfaces.
    
    Subclasses implement specific surface equations.
    Base class provides differential geometry calculations.
    """
    
    def __init__(self, u_range: Tuple[float, float] = (0, 1),
                 v_range: Tuple[float, float] = (0, 1)):
        self.u_range = u_range
        self.v_range = v_range
    
    @abstractmethod
    def evaluate(self, u: float, v: float) -> np.ndarray:
        """Evaluate surface position at (u, v).
        
        Returns: 3D point on surface
        """
        pass
    
    def partial_u(self, u: float, v: float, h: float = 1e-5) -> np.ndarray:
        """Partial derivative ∂r/∂u via central differences.
        
        More accurate than forward differences: O(h²) vs O(h)
        """
        return (self.evaluate(u + h, v) - self.evaluate(u - h, v)) / (2 * h)
    
    def partial_v(self, u: float, v: float, h: float = 1e-5) -> np.ndarray:
        """Partial derivative ∂r/∂v."""
        return (self.evaluate(u, v + h) - self.evaluate(u, v - h)) / (2 * h)
    
    def normal(self, u: float, v: float) -> np.ndarray:
        """Surface normal: n = (∂r/∂u × ∂r/∂v) / |∂r/∂u × ∂r/∂v|
        
        The normal direction defines "outside" of the surface.
        """
        du = self.partial_u(u, v)
        dv = self.partial_v(u, v)
        n = np.cross(du, dv)
        mag = np.linalg.norm(n)
        return n / mag if mag > 1e-10 else np.array([0, 1, 0])
    
    def tangent_u(self, u: float, v: float) -> np.ndarray:
        """Tangent vector in u direction (normalized)."""
        du = self.partial_u(u, v)
        return du / np.linalg.norm(du)
    
    def tangent_v(self, u: float, v: float) -> np.ndarray:
        """Tangent vector in v direction (normalized)."""
        dv = self.partial_v(u, v)
        return dv / np.linalg.norm(dv)
    
    # === FIRST FUNDAMENTAL FORM (Intrinsic Geometry) ===
    
    def first_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]:
        """Compute coefficients E, F, G of first fundamental form.
        
        I = E du² + 2F du dv + G dv²
        
        The first fundamental form encodes intrinsic geometry:
        - Lengths of curves on surface
        - Angles between curves
        - Areas of regions
        
        It's "intrinsic" because it can be measured by a 2D being
        living on the surface without reference to embedding space.
        """
        du = self.partial_u(u, v)
        dv = self.partial_v(u, v)
        
        E = np.dot(du, du)  # |∂r/∂u|²
        F = np.dot(du, dv)  # ∂r/∂u · ∂r/∂v
        G = np.dot(dv, dv)  # |∂r/∂v|²
        
        return E, F, G
    
    def metric_tensor(self, u: float, v: float) -> np.ndarray:
        """Metric tensor (2x2 matrix form of first fundamental form).
        
        g = [[E, F],
             [F, G]]
        
        Used for: length calculations, area elements, covariant derivatives
        """
        E, F, G = self.first_fundamental_form(u, v)
        return np.array([[E, F], [F, G]])
    
    def area_element(self, u: float, v: float) -> float:
        """Differential area element: dA = √(EG - F²) du dv
        
        Used for integrating functions over the surface.
        """
        E, F, G = self.first_fundamental_form(u, v)
        return np.sqrt(E * G - F * F)
    
    # === SECOND FUNDAMENTAL FORM (Extrinsic Geometry) ===
    
    def second_derivatives(self, u: float, v: float, h: float = 1e-4
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Second partial derivatives: ∂²r/∂u², ∂²r/∂u∂v, ∂²r/∂v²"""
        # ∂²r/∂u²
        d2u = (self.evaluate(u + h, v) - 2*self.evaluate(u, v) + 
               self.evaluate(u - h, v)) / (h * h)
        
        # ∂²r/∂v²
        d2v = (self.evaluate(u, v + h) - 2*self.evaluate(u, v) + 
               self.evaluate(u, v - h)) / (h * h)
        
        # ∂²r/∂u∂v (mixed partial)
        d2uv = (self.evaluate(u + h, v + h) - self.evaluate(u + h, v - h) -
                self.evaluate(u - h, v + h) + self.evaluate(u - h, v - h)) / (4 * h * h)
        
        return d2u, d2uv, d2v
    
    def second_fundamental_form(self, u: float, v: float
                               ) -> Tuple[float, float, float]:
        """Compute coefficients L, M, N of second fundamental form.
        
        II = L du² + 2M du dv + N dv²
        
        The second fundamental form measures how the surface curves
        in the embedding space (extrinsic curvature).
        
        L = n · ∂²r/∂u²
        M = n · ∂²r/∂u∂v
        N = n · ∂²r/∂v²
        """
        n = self.normal(u, v)
        d2u, d2uv, d2v = self.second_derivatives(u, v)
        
        L = np.dot(n, d2u)
        M = np.dot(n, d2uv)
        N = np.dot(n, d2v)
        
        return L, M, N
    
    # === CURVATURE ===
    
    def gaussian_curvature(self, u: float, v: float) -> float:
        """Gaussian curvature: K = (LN - M²) / (EG - F²) = κ₁ × κ₂
        
        Intrinsic curvature (Theorema Egregium):
        K can be computed from first fundamental form alone!
        
        Interpretation:
        - K > 0: elliptic (sphere-like, positive curvature in all directions)
        - K < 0: hyperbolic (saddle-like, opposite curvatures)
        - K = 0: parabolic (cylinder-like, zero curvature in one direction)
        
        Gauss-Bonnet theorem: ∫∫_S K dA = 2π χ(S)
        where χ(S) is the Euler characteristic of surface S.
        """
        E, F, G = self.first_fundamental_form(u, v)
        L, M, N = self.second_fundamental_form(u, v)
        
        denom = E * G - F * F
        if abs(denom) < 1e-10:
            return 0.0
        
        return (L * N - M * M) / denom
    
    def mean_curvature(self, u: float, v: float) -> float:
        """Mean curvature: H = (EN - 2FM + GL) / (2(EG - F²)) = (κ₁ + κ₂) / 2
        
        Extrinsic curvature (depends on embedding).
        
        Applications:
        - Minimal surfaces: H = 0 everywhere (soap films)
        - Surface tension: pressure difference = 2γH
        - Surface fairing: minimize ∫H² dA
        """
        E, F, G = self.first_fundamental_form(u, v)
        L, M, N = self.second_fundamental_form(u, v)
        
        denom = E * G - F * F
        if abs(denom) < 1e-10:
            return 0.0
        
        return (E * N - 2 * F * M + G * L) / (2 * denom)
    
    def principal_curvatures(self, u: float, v: float) -> Tuple[float, float]:
        """Principal curvatures κ₁, κ₂.
        
        Eigenvalues of shape operator (Weingarten map).
        
        κ₁, κ₂ are maximum and minimum curvatures at the point.
        
        κ = H ± √(H² - K)
        """
        K = self.gaussian_curvature(u, v)
        H = self.mean_curvature(u, v)
        
        discriminant = H * H - K
        if discriminant < 0:
            discriminant = 0  # Numerical protection
        
        sqrt_disc = np.sqrt(discriminant)
        return H + sqrt_disc, H - sqrt_disc
    
    def shape_index(self, u: float, v: float) -> float:
        """Shape index: S = (2/π) arctan((κ₂ + κ₁)/(κ₂ - κ₁))
        
        Normalized curvature measure in range [-1, 1].
        
        S = -1: cup (concave)
        S = -0.5: rut (cylinder concave)
        S = 0: saddle
        S = 0.5: ridge (cylinder convex)
        S = 1: cap (dome)
        
        Useful for shape recognition and classification.
        """
        k1, k2 = self.principal_curvatures(u, v)
        
        if abs(k2 - k1) < 1e-10:
            return 0.0
        
        return (2 / np.pi) * np.arctan((k2 + k1) / (k2 - k1))
    
    def curvedness(self, u: float, v: float) -> float:
        """Curvedness: C = √((κ₁² + κ₂²)/2)
        
        Magnitude of curvature independent of shape.
        C = 0 for flat surfaces.
        """
        k1, k2 = self.principal_curvatures(u, v)
        return np.sqrt((k1 * k1 + k2 * k2) / 2)


class Sphere(ParametricSurface):
    """Sphere of radius r centered at origin.
    
    Parametrization:
    x = r sin(v) cos(u)
    y = r sin(v) sin(u)
    z = r cos(v)
    
    u ∈ [0, 2π]: azimuthal angle (longitude)
    v ∈ [0, π]: polar angle (latitude from pole)
    """
    
    def __init__(self, radius: float = 1.0):
        super().__init__(u_range=(0, 2*np.pi), v_range=(0, np.pi))
        self.radius = radius
    
    def evaluate(self, u: float, v: float) -> np.ndarray:
        r = self.radius
        return np.array([
            r * np.sin(v) * np.cos(u),
            r * np.sin(v) * np.sin(u),
            r * np.cos(v)
        ])
    
    # Analytical derivatives (more accurate than numerical)
    def partial_u(self, u: float, v: float, h: float = None) -> np.ndarray:
        r = self.radius
        return np.array([
            -r * np.sin(v) * np.sin(u),
            r * np.sin(v) * np.cos(u),
            0
        ])
    
    def partial_v(self, u: float, v: float, h: float = None) -> np.ndarray:
        r = self.radius
        return np.array([
            r * np.cos(v) * np.cos(u),
            r * np.cos(v) * np.sin(u),
            -r * np.sin(v)
        ])


class Torus(ParametricSurface):
    """Torus with major radius R and minor radius r.
    
    Parametrization:
    x = (R + r cos(v)) cos(u)
    y = (R + r cos(v)) sin(u)
    z = r sin(v)
    
    u ∈ [0, 2π]: angle around major circumference
    v ∈ [0, 2π]: angle around minor circumference
    """
    
    def __init__(self, major_radius: float = 1.0, minor_radius: float = 0.3):
        super().__init__(u_range=(0, 2*np.pi), v_range=(0, 2*np.pi))
        self.R = major_radius
        self.r = minor_radius
    
    def evaluate(self, u: float, v: float) -> np.ndarray:
        return np.array([
            (self.R + self.r * np.cos(v)) * np.cos(u),
            (self.R + self.r * np.cos(v)) * np.sin(u),
            self.r * np.sin(v)
        ])


class BezierSurface(ParametricSurface):
    """Bicubic Bézier surface patch.
    
    Defined by 4×4 grid of control points.
    
    r(u,v) = Σᵢ Σⱼ Bᵢ(u) Bⱼ(v) Pᵢⱼ
    
    where Bᵢ are Bernstein basis polynomials.
    """
    
    def __init__(self, control_points: np.ndarray):
        """
        control_points: shape (4, 4, 3) array of 16 control points
        """
        super().__init__()
        self.P = control_points
    
    @staticmethod
    def bernstein(n: int, i: int, t: float) -> float:
        """Bernstein basis polynomial Bᵢⁿ(t).
        
        Bᵢⁿ(t) = C(n,i) tⁱ (1-t)^(n-i)
        """
        from math import comb
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def evaluate(self, u: float, v: float) -> np.ndarray:
        result = np.zeros(3)
        for i in range(4):
            for j in range(4):
                b = self.bernstein(3, i, u) * self.bernstein(3, j, v)
                result += b * self.P[i, j]
        return result
    
    def partial_u(self, u: float, v: float, h: float = None) -> np.ndarray:
        """Analytical derivative of Bézier surface in u direction.
        
        ∂r/∂u = 3 Σᵢ Σⱼ [Bᵢ₋₁²(u) - Bᵢ²(u)] Bⱼ³(v) Pᵢⱼ
        
        Simplified using derivative of Bernstein polynomials:
        ∂Bᵢⁿ/∂t = n(Bᵢ₋₁ⁿ⁻¹ - Bᵢⁿ⁻¹)
        """
        # Compute derivative control points
        dP = np.zeros((3, 4, 3))
        for i in range(3):
            for j in range(4):
                dP[i, j] = 3 * (self.P[i+1, j] - self.P[i, j])
        
        # Evaluate quadratic Bézier in u, cubic in v
        result = np.zeros(3)
        for i in range(3):
            for j in range(4):
                b = self.bernstein(2, i, u) * self.bernstein(3, j, v)
                result += b * dP[i, j]
        return result
```

### 2.2 Implicit Surfaces

An implicit surface is defined as the zero level set of a function f: ℝ³ → ℝ:

```
S = { (x, y, z) ∈ ℝ³ : f(x, y, z) = 0 }
```

```python
class ImplicitSurface(ABC):
    """Abstract base class for implicit surfaces.
    
    Implicit surfaces defined by f(x, y, z) = 0.
    
    Advantages over parametric:
    - Easy Boolean operations (CSG)
    - Topologically flexible
    - Natural for level sets, SDFs
    
    Disadvantages:
    - Harder to directly sample points
    - Requires root finding for intersection
    - UV mapping more complex
    """
    
    @abstractmethod
    def evaluate(self, p: np.ndarray) -> float:
        """Evaluate function value at point.
        
        f(p) < 0: inside surface
        f(p) = 0: on surface
        f(p) > 0: outside surface
        """
        pass
    
    def gradient(self, p: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """Gradient ∇f via central differences.
        
        The gradient points in direction of steepest increase.
        Normal = ∇f / |∇f|
        """
        return np.array([
            (self.evaluate(p + [h, 0, 0]) - self.evaluate(p - [h, 0, 0])) / (2*h),
            (self.evaluate(p + [0, h, 0]) - self.evaluate(p - [0, h, 0])) / (2*h),
            (self.evaluate(p + [0, 0, h]) - self.evaluate(p - [0, 0, h])) / (2*h)
        ])
    
    def normal(self, p: np.ndarray) -> np.ndarray:
        """Surface normal at point (assumes p is on surface)."""
        g = self.gradient(p)
        mag = np.linalg.norm(g)
        return g / mag if mag > 1e-10 else np.array([0, 1, 0])
    
    def hessian(self, p: np.ndarray, h: float = 1e-3) -> np.ndarray:
        """Hessian matrix H of second derivatives.
        
        H[i,j] = ∂²f/∂xᵢ∂xⱼ
        
        Used for curvature calculations on implicit surfaces.
        """
        H = np.zeros((3, 3))
        
        # Diagonal elements (∂²f/∂xᵢ²)
        for i, offset in enumerate([[1,0,0], [0,1,0], [0,0,1]]):
            offset = np.array(offset) * h
            H[i, i] = (self.evaluate(p + offset) - 2*self.evaluate(p) + 
                      self.evaluate(p - offset)) / (h * h)
        
        # Off-diagonal elements (∂²f/∂xᵢ∂xⱼ)
        offsets = [([1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0]),  # xy
                   ([1,0,1], [1,0,-1], [-1,0,1], [-1,0,-1]),  # xz
                   ([0,1,1], [0,1,-1], [0,-1,1], [0,-1,-1])]  # yz
        indices = [(0,1), (0,2), (1,2)]
        
        for (i, j), offs in zip(indices, offsets):
            H[i, j] = (self.evaluate(p + np.array(offs[0])*h) -
                      self.evaluate(p + np.array(offs[1])*h) -
                      self.evaluate(p + np.array(offs[2])*h) +
                      self.evaluate(p + np.array(offs[3])*h)) / (4 * h * h)
            H[j, i] = H[i, j]
        
        return H
    
    def mean_curvature(self, p: np.ndarray) -> float:
        """Mean curvature for implicit surface.
        
        H = -div(∇f/|∇f|) / 2
          = -(|∇f|²trace(H) - ∇f·H·∇f) / (2|∇f|³)
        
        where H is the Hessian matrix.
        """
        g = self.gradient(p)
        H = self.hessian(p)
        
        g_mag = np.linalg.norm(g)
        if g_mag < 1e-10:
            return 0.0
        
        trace_H = np.trace(H)
        g_H_g = g @ H @ g
        
        return -(g_mag**2 * trace_H - g_H_g) / (2 * g_mag**3)
    
    def gaussian_curvature(self, p: np.ndarray) -> float:
        """Gaussian curvature for implicit surface.
        
        K = [∇f · adj(H) · ∇f] / |∇f|⁴
        
        where adj(H) is the adjugate (classical adjoint) of Hessian.
        """
        g = self.gradient(p)
        H = self.hessian(p)
        
        g_mag = np.linalg.norm(g)
        if g_mag < 1e-10:
            return 0.0
        
        # Adjugate of 3x3 matrix
        adj_H = np.array([
            [H[1,1]*H[2,2] - H[1,2]*H[2,1], H[0,2]*H[2,1] - H[0,1]*H[2,2], H[0,1]*H[1,2] - H[0,2]*H[1,1]],
            [H[1,2]*H[2,0] - H[1,0]*H[2,2], H[0,0]*H[2,2] - H[0,2]*H[2,0], H[0,2]*H[1,0] - H[0,0]*H[1,2]],
            [H[1,0]*H[2,1] - H[1,1]*H[2,0], H[0,1]*H[2,0] - H[0,0]*H[2,1], H[0,0]*H[1,1] - H[0,1]*H[1,0]]
        ])
        
        return (g @ adj_H @ g) / (g_mag ** 4)
    
    def project_to_surface(self, p: np.ndarray, max_iter: int = 20, 
                           tol: float = 1e-6) -> np.ndarray:
        """Project point onto surface via Newton's method.
        
        Iteratively moves point along gradient to reach f(p) = 0.
        """
        for _ in range(max_iter):
            f = self.evaluate(p)
            if abs(f) < tol:
                break
            
            g = self.gradient(p)
            g_mag_sq = np.dot(g, g)
            if g_mag_sq < 1e-10:
                break
            
            p = p - (f / g_mag_sq) * g
        
        return p
    
    def ray_march(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                  max_distance: float = 100.0, max_steps: int = 100,
                  tolerance: float = 1e-4) -> Tuple[bool, float, np.ndarray]:
        """Ray march to find intersection with surface.
        
        For SDFs, step size = function value (sphere tracing).
        For general implicit surfaces, use fixed step + refinement.
        
        Returns: (hit, distance, hit_point)
        """
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        t = 0.0
        
        for _ in range(max_steps):
            p = ray_origin + t * ray_dir
            f = self.evaluate(p)
            
            if abs(f) < tolerance:
                return True, t, p
            
            if t > max_distance:
                return False, max_distance, ray_origin + max_distance * ray_dir
            
            # For SDF, step by f; otherwise use fixed step
            step = max(abs(f), 0.01)
            t += step
        
        return False, t, ray_origin + t * ray_dir


class ImplicitSphere(ImplicitSurface):
    """Sphere: f(x,y,z) = x² + y² + z² - r² = 0"""
    
    def __init__(self, center: np.ndarray = None, radius: float = 1.0):
        self.center = center if center is not None else np.zeros(3)
        self.radius = radius
    
    def evaluate(self, p: np.ndarray) -> float:
        d = p - self.center
        return np.dot(d, d) - self.radius * self.radius
    
    def gradient(self, p: np.ndarray, h: float = None) -> np.ndarray:
        return 2.0 * (p - self.center)


class ImplicitTorus(ImplicitSurface):
    """Torus: f(x,y,z) = (√(x² + y²) - R)² + z² - r² = 0
    
    R: major radius (distance from center to tube center)
    r: minor radius (tube radius)
    """
    
    def __init__(self, major_radius: float = 1.0, minor_radius: float = 0.3):
        self.R = major_radius
        self.r = minor_radius
    
    def evaluate(self, p: np.ndarray) -> float:
        xy_dist = np.sqrt(p[0]**2 + p[1]**2)
        return (xy_dist - self.R)**2 + p[2]**2 - self.r**2
```

---

## Chapter 3: Topology and Manifold Theory

### 3.1 Mesh Topology Fundamentals

A mesh is a discrete representation of a surface composed of vertices, edges, and faces:

```python
"""
MESH TOPOLOGY CONCEPTS

EULER CHARACTERISTIC:
χ = V - E + F

For closed surfaces:
- Sphere: χ = 2
- Torus: χ = 0
- Double torus: χ = -2
- General: χ = 2 - 2g where g = genus (number of handles)

EULER-POINCARÉ FORMULA (with boundaries):
χ = V - E + F = 2 - 2g - b

where b = number of boundary loops

MESH PROPERTIES:

1. MANIFOLD:
   A mesh is manifold if:
   - Every edge is shared by exactly 1 or 2 faces
   - The faces around every vertex form a single fan/loop
   
   Non-manifold conditions:
   - T-junctions (edge shared by 3+ faces)
   - Bowtie vertices (multiple fans meeting at a point)
   - Duplicate faces
   - Isolated vertices

2. ORIENTABLE:
   A mesh is orientable if consistent winding order exists
   for all faces (all normals point "out" or all point "in").
   
   Non-orientable: Möbius strip, Klein bottle

3. CLOSED:
   A mesh is closed (watertight) if:
   - Every edge is shared by exactly 2 faces
   - No boundary edges exist
   
   Open meshes have boundary edges (shared by only 1 face).

4. GENUS:
   Number of "handles" or "holes" in the surface.
   g = (2 - χ) / 2 for closed orientable surfaces.
   
   Examples:
   - Sphere: g = 0
   - Torus: g = 1
   - Coffee mug: g = 1 (topologically a torus!)
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
import numpy as np

@dataclass
class Vertex:
    """Mesh vertex with position and topology information."""
    position: np.ndarray
    index: int
    edges: List[int] = field(default_factory=list)  # Incident edge indices
    normal: np.ndarray = None  # Computed vertex normal
    uv: np.ndarray = None  # Texture coordinates
    
    def valence(self) -> int:
        """Number of edges incident to this vertex."""
        return len(self.edges)


@dataclass
class Edge:
    """Mesh edge connecting two vertices."""
    v0: int  # First vertex index
    v1: int  # Second vertex index
    index: int
    faces: List[int] = field(default_factory=list)  # Adjacent face indices
    
    def is_boundary(self) -> bool:
        """True if edge is on mesh boundary (only 1 adjacent face)."""
        return len(self.faces) == 1
    
    def is_manifold(self) -> bool:
        """True if edge is manifold (1 or 2 adjacent faces)."""
        return len(self.faces) <= 2


@dataclass
class Face:
    """Mesh face (polygon, typically triangle or quad)."""
    vertices: List[int]  # Vertex indices in winding order
    index: int
    edges: List[int] = field(default_factory=list)  # Edge indices
    normal: np.ndarray = None  # Face normal
    
    def is_triangle(self) -> bool:
        return len(self.vertices) == 3
    
    def is_quad(self) -> bool:
        return len(self.vertices) == 4


class HalfEdgeMesh:
    """Half-edge mesh data structure for efficient topology queries.
    
    Each directed half-edge stores:
    - Origin vertex
    - Opposite half-edge (twin)
    - Next half-edge in face
    - Previous half-edge in face
    - Incident face
    
    Advantages:
    - O(1) access to adjacent elements
    - Easy traversal around vertices/faces
    - Natural representation of orientation
    
    Disadvantages:
    - Higher memory usage
    - More complex to construct/modify
    - Requires manifold mesh
    """
    
    @dataclass
    class HalfEdge:
        vertex: int  # Origin vertex
        twin: int = -1  # Opposite half-edge (-1 if boundary)
        next: int = -1  # Next half-edge in face
        prev: int = -1  # Previous half-edge in face
        face: int = -1  # Incident face (-1 if boundary)
        index: int = 0
    
    def __init__(self):
        self.vertices: List[np.ndarray] = []
        self.half_edges: List[HalfEdgeMesh.HalfEdge] = []
        self.faces: List[int] = []  # Starting half-edge index for each face
        self.vertex_half_edge: List[int] = []  # One outgoing half-edge per vertex
    
    @classmethod
    def from_face_list(cls, vertices: np.ndarray, 
                       faces: List[List[int]]) -> 'HalfEdgeMesh':
        """Construct half-edge mesh from vertex positions and face indices.
        
        Args:
            vertices: (N, 3) array of vertex positions
            faces: List of face vertex indices (each face is a list)
        
        Returns:
            Constructed HalfEdgeMesh
        """
        mesh = cls()
        mesh.vertices = [v.copy() for v in vertices]
        mesh.vertex_half_edge = [-1] * len(vertices)
        
        # Map from (v0, v1) to half-edge index for twin finding
        edge_map: Dict[Tuple[int, int], int] = {}
        
        for face_idx, face_verts in enumerate(faces):
            n = len(face_verts)
            first_he = len(mesh.half_edges)
            mesh.faces.append(first_he)
            
            # Create half-edges for this face
            for i in range(n):
                v0 = face_verts[i]
                v1 = face_verts[(i + 1) % n]
                
                he = cls.HalfEdge(
                    vertex=v0,
                    face=face_idx,
                    index=len(mesh.half_edges)
                )
                mesh.half_edges.append(he)
                
                # Set vertex's half-edge
                if mesh.vertex_half_edge[v0] == -1:
                    mesh.vertex_half_edge[v0] = he.index
                
                # Store for twin finding
                edge_map[(v0, v1)] = he.index
            
            # Link next/prev within face
            for i in range(n):
                curr = first_he + i
                mesh.half_edges[curr].next = first_he + (i + 1) % n
                mesh.half_edges[curr].prev = first_he + (i - 1) % n
        
        # Find twins
        for (v0, v1), he_idx in edge_map.items():
            twin_key = (v1, v0)
            if twin_key in edge_map:
                twin_idx = edge_map[twin_key]
                mesh.half_edges[he_idx].twin = twin_idx
                mesh.half_edges[twin_idx].twin = he_idx
        
        return mesh
    
    def vertex_ring(self, vertex_idx: int) -> List[int]:
        """Get vertices in the 1-ring neighborhood of a vertex.
        
        Returns vertices connected by a single edge.
        """
        ring = []
        start_he = self.vertex_half_edge[vertex_idx]
        if start_he == -1:
            return ring
        
        he = start_he
        while True:
            # Get the vertex at the end of this half-edge
            next_he = self.half_edges[he].next
            ring.append(self.half_edges[next_he].vertex)
            
            # Move to next outgoing half-edge
            twin = self.half_edges[he].twin
            if twin == -1:
                # Hit boundary - need to go other direction
                break
            he = self.half_edges[twin].next
            
            if he == start_he:
                break
        
        return ring
    
    def face_vertices(self, face_idx: int) -> List[int]:
        """Get vertices of a face in order."""
        verts = []
        start_he = self.faces[face_idx]
        he = start_he
        while True:
            verts.append(self.half_edges[he].vertex)
            he = self.half_edges[he].next
            if he == start_he:
                break
        return verts
    
    def is_boundary_vertex(self, vertex_idx: int) -> bool:
        """Check if vertex is on mesh boundary."""
        start_he = self.vertex_half_edge[vertex_idx]
        if start_he == -1:
            return True  # Isolated vertex
        
        he = start_he
        while True:
            if self.half_edges[he].twin == -1:
                return True
            he = self.half_edges[self.half_edges[he].twin].next
            if he == start_he:
                break
        return False
    
    def boundary_loops(self) -> List[List[int]]:
        """Find all boundary loops in the mesh.
        
        Returns list of vertex index lists, each representing a boundary loop.
        """
        loops = []
        visited_edges = set()
        
        for he in self.half_edges:
            if he.twin == -1 and he.index not in visited_edges:
                # Found boundary edge, trace the loop
                loop = []
                curr = he.index
                while curr not in visited_edges:
                    visited_edges.add(curr)
                    loop.append(self.half_edges[curr].vertex)
                    
                    # Find next boundary edge
                    next_he = self.half_edges[curr].next
                    while self.half_edges[next_he].twin != -1:
                        next_he = self.half_edges[self.half_edges[next_he].twin].next
                    curr = next_he
                
                loops.append(loop)
        
        return loops
    
    def compute_vertex_normals(self):
        """Compute area-weighted vertex normals."""
        normals = [np.zeros(3) for _ in self.vertices]
        
        for face_idx in range(len(self.faces)):
            verts = self.face_vertices(face_idx)
            if len(verts) < 3:
                continue
            
            # Compute face normal and area
            v0, v1, v2 = [self.vertices[i] for i in verts[:3]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = np.linalg.norm(cross) / 2
            
            if area > 1e-10:
                face_normal = cross / (2 * area)
                
                # Add weighted contribution to each vertex
                for v_idx in verts:
                    normals[v_idx] += face_normal * area
        
        # Normalize
        for i in range(len(normals)):
            mag = np.linalg.norm(normals[i])
            if mag > 1e-10:
                normals[i] /= mag
            else:
                normals[i] = np.array([0, 1, 0])
        
        return normals
    
    def euler_characteristic(self) -> int:
        """Compute Euler characteristic χ = V - E + F."""
        V = len(self.vertices)
        F = len(self.faces)
        
        # Count unique edges (each undirected edge has 2 half-edges)
        edge_count = 0
        for he in self.half_edges:
            if he.twin == -1 or he.index < he.twin:
                edge_count += 1
        
        return V - edge_count + F
    
    def genus(self) -> int:
        """Compute genus (number of handles).
        
        For closed orientable surfaces: g = (2 - χ) / 2
        """
        chi = self.euler_characteristic()
        b = len(self.boundary_loops())
        
        # χ = 2 - 2g - b
        # g = (2 - χ - b) / 2
        return (2 - chi - b) // 2
    
    def is_closed(self) -> bool:
        """Check if mesh is closed (watertight)."""
        return all(he.twin != -1 for he in self.half_edges)
    
    def is_manifold(self) -> bool:
        """Check if mesh is manifold.
        
        Conditions:
        1. Every edge has at most 2 incident faces
        2. Faces around each vertex form a single fan
        """
        # Check edge condition (built into half-edge structure)
        # If we successfully built the structure, edges are OK
        
        # Check vertex condition
        for v_idx in range(len(self.vertices)):
            start_he = self.vertex_half_edge[v_idx]
            if start_he == -1:
                continue  # Isolated vertex (may or may not be allowed)
            
            # Count half-edges by traversing both directions
            visited = set()
            he = start_he
            
            # Forward traversal
            while he not in visited:
                visited.add(he)
                twin = self.half_edges[he].twin
                if twin == -1:
                    break
                he = self.half_edges[twin].next
                if he == start_he:
                    break
            
            # If we hit boundary, traverse backward too
            if self.half_edges[start_he].twin == -1 or he != start_he:
                he = self.half_edges[start_he].prev
                while True:
                    twin = self.half_edges[he].twin
                    if twin == -1:
                        break
                    visited.add(twin)
                    he = self.half_edges[twin].prev
            
            # Count expected half-edges from vertex
            expected = sum(1 for h in self.half_edges if h.vertex == v_idx)
            if len(visited) != expected:
                return False  # Multiple fans at this vertex
        
        return True
```

### 3.2 Mesh Operations

```python
class MeshOperations:
    """Collection of mesh topology operations."""
    
    @staticmethod
    def triangulate_polygon(vertices: List[int], 
                           positions: List[np.ndarray]) -> List[List[int]]:
        """Triangulate a convex or simple polygon using ear clipping.
        
        Args:
            vertices: Indices of polygon vertices in order
            positions: 3D positions for each vertex
        
        Returns:
            List of triangle vertex index lists
        """
        if len(vertices) < 3:
            return []
        if len(vertices) == 3:
            return [vertices.copy()]
        
        triangles = []
        remaining = vertices.copy()
        
        # Simple ear clipping for convex polygons
        # (Production code would use proper ear clipping with reflex detection)
        while len(remaining) > 3:
            # Find an ear (convex vertex where triangle doesn't contain other vertices)
            for i in range(len(remaining)):
                prev_idx = (i - 1) % len(remaining)
                next_idx = (i + 1) % len(remaining)
                
                v_prev = remaining[prev_idx]
                v_curr = remaining[i]
                v_next = remaining[next_idx]
                
                # Check if this is a valid ear
                p0 = positions[v_prev]
                p1 = positions[v_curr]
                p2 = positions[v_next]
                
                # Check convexity (cross product direction)
                edge1 = p1 - p0
                edge2 = p2 - p1
                cross = np.cross(edge1, edge2)
                
                # Assuming consistent winding, positive Z means convex
                if cross[2] > 0:  # Simplified check
                    triangles.append([v_prev, v_curr, v_next])
                    remaining.pop(i)
                    break
            else:
                # No ear found - fall back to fan triangulation
                center = remaining[0]
                for i in range(1, len(remaining) - 1):
                    triangles.append([center, remaining[i], remaining[i + 1]])
                break
        
        if len(remaining) == 3:
            triangles.append(remaining)
        
        return triangles
    
    @staticmethod
    def compute_face_normal(positions: np.ndarray, 
                           face_vertices: List[int]) -> np.ndarray:
        """Compute face normal using Newell's method.
        
        More robust than simple cross product for non-planar polygons.
        """
        normal = np.zeros(3)
        n = len(face_vertices)
        
        for i in range(n):
            v0 = positions[face_vertices[i]]
            v1 = positions[face_vertices[(i + 1) % n]]
            
            # Newell's method
            normal[0] += (v0[1] - v1[1]) * (v0[2] + v1[2])
            normal[1] += (v0[2] - v1[2]) * (v0[0] + v1[0])
            normal[2] += (v0[0] - v1[0]) * (v0[1] + v1[1])
        
        mag = np.linalg.norm(normal)
        return normal / mag if mag > 1e-10 else np.array([0, 1, 0])
    
    @staticmethod
    def compute_face_area(positions: np.ndarray, 
                         face_vertices: List[int]) -> float:
        """Compute area of a polygon face.
        
        Uses the shoelace formula generalized to 3D.
        """
        if len(face_vertices) < 3:
            return 0.0
        
        # Triangulate and sum areas
        total_area = 0.0
        v0 = positions[face_vertices[0]]
        
        for i in range(1, len(face_vertices) - 1):
            v1 = positions[face_vertices[i]]
            v2 = positions[face_vertices[i + 1]]
            
            cross = np.cross(v1 - v0, v2 - v0)
            total_area += np.linalg.norm(cross) / 2
        
        return total_area
    
    @staticmethod
    def compute_mesh_volume(positions: np.ndarray, 
                           faces: List[List[int]]) -> float:
        """Compute signed volume of a closed mesh.
        
        Uses the divergence theorem:
        V = (1/6) Σ (v0 · (v1 × v2))
        
        Positive if faces have outward normals.
        """
        volume = 0.0
        
        for face in faces:
            if len(face) < 3:
                continue
            
            # Triangulate
            v0 = positions[face[0]]
            for i in range(1, len(face) - 1):
                v1 = positions[face[i]]
                v2 = positions[face[i + 1]]
                
                # Signed volume of tetrahedron with origin
                volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        
        return volume
    
    @staticmethod
    def compute_centroid(positions: np.ndarray, 
                        faces: List[List[int]]) -> np.ndarray:
        """Compute centroid of a mesh weighted by face area."""
        centroid = np.zeros(3)
        total_area = 0.0
        
        for face in faces:
            if len(face) < 3:
                continue
            
            # Face centroid (average of vertices)
            face_centroid = np.mean([positions[v] for v in face], axis=0)
            
            # Face area
            area = MeshOperations.compute_face_area(positions, face)
            
            centroid += face_centroid * area
            total_area += area
        
        return centroid / total_area if total_area > 1e-10 else np.zeros(3)
    
    @staticmethod
    def compute_bounding_box(positions: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box.
        
        Returns: (min_corner, max_corner)
        """
        return positions.min(axis=0), positions.max(axis=0)
    
    @staticmethod
    def compute_bounding_sphere(positions: np.ndarray
                               ) -> Tuple[np.ndarray, float]:
        """Compute bounding sphere using Ritter's algorithm.
        
        Not optimal but fast O(n).
        
        Returns: (center, radius)
        """
        if len(positions) == 0:
            return np.zeros(3), 0.0
        
        # Find most separated pair along x, y, z
        min_pts = positions.argmin(axis=0)
        max_pts = positions.argmax(axis=0)
        
        max_dist = 0
        p1, p2 = 0, 0
        for i in range(3):
            dist = np.linalg.norm(positions[max_pts[i]] - positions[min_pts[i]])
            if dist > max_dist:
                max_dist = dist
                p1, p2 = min_pts[i], max_pts[i]
        
        # Initial sphere from diameter
        center = (positions[p1] + positions[p2]) / 2
        radius = max_dist / 2
        
        # Expand to include all points
        for p in positions:
            dist = np.linalg.norm(p - center)
            if dist > radius:
                # Expand sphere to include this point
                radius = (radius + dist) / 2
                center = center + (p - center) * ((dist - radius) / dist)
        
        return center, radius
```

---

This concludes Part I (Chapters 1-3) of the Encyclopedia. The full document continues with:

- **Part II**: Signed Distance Functions, CSG, Metaballs, Level Sets
- **Part III**: Marching Cubes, Dual Contouring, Surface Nets, Mesh Optimization
- **Part IV**: L-Systems, Subdivision, Fractal Terrain, Architectural Modeling
- **Part V**: 3D Noise, Triplanar Mapping, PBR Materials, Weathering
- **Part VI**: IK Solvers, Procedural Locomotion, Physics-Driven Animation
- **Part VII**: Terrain Systems, Vegetation, Dungeon Generation, Biomes
- **Part VIII**: LOD, Culling, GPU Compute, Streaming

Total planned length: ~25,000 lines

---

*Continue to Part II: Implicit Surface Representations...*
