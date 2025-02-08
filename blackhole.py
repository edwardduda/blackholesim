import argparse
import time
import numpy as np
from numba import jit, prange
from classes.Visualizer import Visualizer

SCHWARZSCHILD_RADIUS = 4.0
EPS = 1e-6

@jit(nopython=True, cache=True)
def calculate_light_bending(r: float) -> float:
    if r < SCHWARZSCHILD_RADIUS:
        return float('inf')
    denom = r * np.sqrt(1.0 - SCHWARZSCHILD_RADIUS / (r + EPS))
    if denom < EPS:
        return float('inf')
    return 2.0 * SCHWARZSCHILD_RADIUS / denom

@jit(nopython=True, cache=True)
def calculate_doppler_shift(velocity: float, angle: float) -> float:

    v_clamped = min(velocity, 1.0 - EPS)
    gamma = 1.0 / np.sqrt(1.0 - v_clamped * v_clamped + EPS)
    return gamma * (1.0 - v_clamped * np.cos(angle))

@jit(nopython=True, cache=True)
def calculate_disk_intensity(r: float, phi: float,
                             disk_inner_radius: float,
                             disk_outer_radius: float) -> float:

    if r < disk_inner_radius or r > disk_outer_radius:
        return 0.0
    
    exponent = 2.0
    base_temp = (disk_inner_radius / (r + EPS)) ** exponent
    
    azimuth_factor = 1.0 + 0.2 * np.cos(2.0 * phi)
    
    return base_temp * azimuth_factor

@jit(nopython=True, cache=True)
def ray_trace(x: float, y: float,
              disk_inner_radius: float,
              disk_outer_radius: float,
              tilt_angle: float,
              time_value: float = 0.0) -> float:

    alpha = np.radians(tilt_angle)
    
    x_prime = x
    y_prime = y * np.cos(alpha)
    z_prime = y * np.sin(alpha)
    
    r = np.sqrt(x_prime**2 + y_prime**2 + z_prime**2)
    if r < SCHWARZSCHILD_RADIUS:
        return 0.0
    
    base_phi = np.arctan2(y_prime, x_prime)
    omega = -0.25  # Angular velocity factor
    phi = base_phi + omega * time_value  # Rotate with time
    
    b = r * np.sqrt(1.0 - SCHWARZSCHILD_RADIUS / (r + EPS))
    deflection = calculate_light_bending(r)
    disk_r = r * (1.0 + deflection)
    
    raw_v = np.sqrt(SCHWARZSCHILD_RADIUS / (2.0 * disk_r + EPS))
    velocity = -min(raw_v, 1.0 - EPS)
    
    doppler_factor = calculate_doppler_shift(-velocity, phi)
    
    intensity = calculate_disk_intensity(disk_r, phi + time_value * 0.1, 
                                         disk_inner_radius, disk_outer_radius)
    
    intensity *= doppler_factor**4
    
    r_safe = max(r, SCHWARZSCHILD_RADIUS + EPS)
    redshift_factor = np.sqrt(1.0 - SCHWARZSCHILD_RADIUS / r_safe)
    intensity *= redshift_factor
    
    return intensity

@jit(nopython=True, parallel=True, cache=True)
def generate_image(width: int, height: int, scale: float,
                   disk_inner_radius: float, disk_outer_radius: float,
                   tilt_angle: float, time_value: float) -> np.ndarray:
    image = np.zeros((height, width))
    
    for j in prange(height):
        for i in range(width):
            x = (height / 2 - j) * scale / height
            y = -(i - width / 2) * scale / width
            image[j, i] = ray_trace(x, y,
                                    disk_inner_radius,
                                    disk_outer_radius,
                                    tilt_angle,
                                    time_value)
    return image

def add_star_field(intensity: np.ndarray, prob: float = 0.008, brightness: float = 0.01) -> np.ndarray:
    h, w = intensity.shape
    rand_vals = np.random.rand(h, w)
    star_mask = rand_vals < prob
    return intensity + star_mask * brightness

def intensity_to_ascii_colored(intensity: np.ndarray, palette: np.ndarray) -> list:

    min_val = intensity.min()
    max_val = intensity.max()
    
    if np.isclose(max_val, min_val):
        return [[f"\033[38;5;232m \033[0m" for _ in range(intensity.shape[1])]
                for _ in range(intensity.shape[0])]
        
    gamma = 0.76

    norm = ((intensity - min_val) / (max_val - min_val + EPS)) ** (1 / gamma)
    
    indices = (norm * (len(palette) - 1)).astype(np.int8)
    
    ascii_2d = []
    for i, row in enumerate(indices):
        ascii_row = []
        for j, idx in enumerate(row):
            char = palette[idx]
            # Map normalized intensity to a grey value between 232 and 255.
            grey_val = 232 + int(norm[i, j] * (255 - 232))
            colored_char = f"\033[38;5;{grey_val}m{char}\033[0m"
            ascii_row.append(colored_char)
        ascii_2d.append(ascii_row)
    return ascii_2d

def ascii_render(ascii_2d: list, double_columns: bool = False) -> str:
    lines = []
    for row in ascii_2d:
        if double_columns:
            line = "".join(ch * 2 for ch in row)
        else:
            line = "".join(row)
        lines.append(line)
    return "\n".join(lines)

def main():
    visualizer = Visualizer()
    parser = argparse.ArgumentParser(description="Animated ASCII Black Hole (Terminal In-Place)")
    parser.add_argument("--width", type=int, default=visualizer.visualizer_width, help="ASCII output width")
    parser.add_argument("--height", type=int, default=visualizer.visualizer_height, help="ASCII output height")
    parser.add_argument("--frames", type=int, default=visualizer.frames, help="Total number of frames")
    parser.add_argument("--fps", type=float, default=visualizer.fps, help="Frames per second")
    parser.add_argument("--tilt_start", type=float, default=visualizer.init_tilt, help="Starting tilt angle in degrees")
    parser.add_argument("--tilt_end", type=float, default=visualizer.final_tilt, help="Ending tilt angle in degrees")
    parser.add_argument("--double_cols", action="store_true", help="Double columns for a wider aspect ratio")
    
    # Custom ASCII gradient palette (from low to high intensity)
    parser.add_argument("--palette", type=str,
                        default=" .`'-^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLOQ0AZmwqpdbkhao*#DVMW&8%B@$",
                        help="Custom ASCII gradient palette (from low to high intensity)")
    args = parser.parse_args()
    
    width = args.width
    height = args.height
    frames = args.frames
    fps = args.fps
    tilt_start = args.tilt_start
    tilt_end = args.tilt_end
    
    if args.double_cols:
        width = int(width / 1.9)
 
    # Adjust disk radii relative to Schwarzschild radius.
    visualizer.inner_radius *= SCHWARZSCHILD_RADIUS
    visualizer.outer_radius *= SCHWARZSCHILD_RADIUS
    
    palette = np.array(list(args.palette))
    
    try:
        for frame in range(frames):
            time_value = frame * 2 * np.pi / frames
            
            fraction = frame / max(frames - 1, 1)
            tilt_angle = tilt_start + (tilt_end - tilt_start) * fraction
            
            intensity = generate_image(width, height, visualizer.scale,
                                       visualizer.inner_radius, visualizer.outer_radius,
                                       tilt_angle, time_value)
            # Uncomment to overlay stars on the intensity image.
            #intensity = add_star_field(intensity, prob=0.004, brightness=0.01)
            ascii_2d = intensity_to_ascii_colored(intensity, palette)
            
            frame_str = ascii_render(ascii_2d, double_columns=args.double_cols)
            print("\x1b[2J\x1b[H", end="")
            
            # Optionally display frame information.
            print(f"Frame {frame+1}/{frames} | Tilt: {tilt_angle:.2f}°\n")
            print(frame_str)
            
            time.sleep(1.0 / fps)
    
    except KeyboardInterrupt:
        print("\nAnimation interrupted by user.")

if __name__ == "__main__":
    main()
