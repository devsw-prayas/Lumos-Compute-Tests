import torch
import numpy as np
from PIL import Image
import math

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class Ray:
    """Ray with origin and direction"""

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction / torch.norm(direction, dim=-1, keepdim=True)

    def at(self, t):
        return self.origin + t.unsqueeze(-1) * self.direction


class Camera:
    """Simple perspective camera"""

    def __init__(self, lookfrom, lookat, vup, vfov, aspect_ratio, device):
        self.device = device
        theta = vfov * math.pi / 180
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        w = (lookfrom - lookat)
        w = w / torch.norm(w)
        u = torch.cross(vup, w)
        u = u / torch.norm(u)
        v = torch.cross(w, u)

        self.origin = lookfrom
        self.horizontal = viewport_width * u
        self.vertical = viewport_height * v
        self.lower_left_corner = self.origin - self.horizontal / 2 - self.vertical / 2 - w

    def get_ray(self, u, v):
        """Get ray for given UV coordinates (batched)"""
        direction = (self.lower_left_corner +
                     u.unsqueeze(-1) * self.horizontal +
                     v.unsqueeze(-1) * self.vertical -
                     self.origin)
        origin = self.origin.expand(u.shape[0], 3)
        return Ray(origin, direction)


class Sphere:
    """Sphere primitive"""

    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray, t_min, t_max):
        """Ray-sphere intersection (batched)"""
        oc = ray.origin - self.center
        a = torch.sum(ray.direction * ray.direction, dim=-1)
        half_b = torch.sum(oc * ray.direction, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - self.radius * self.radius

        discriminant = half_b * half_b - a * c

        # Find nearest valid hit
        sqrt_d = torch.sqrt(torch.clamp(discriminant, min=0))
        root = (-half_b - sqrt_d) / a

        # Check if root is in valid range
        valid = (discriminant > 0) & (root > t_min) & (root < t_max)

        # Try other root if first doesn't work
        root2 = (-half_b + sqrt_d) / a
        valid2 = ~valid & (discriminant > 0) & (root2 > t_min) & (root2 < t_max)
        root = torch.where(valid2, root2, root)
        valid = valid | valid2

        return valid, root


class Material:
    """Base material class"""

    def __init__(self, albedo, material_type='diffuse', fuzz=0.0, ref_idx=1.5):
        self.albedo = albedo
        self.material_type = material_type
        self.fuzz = min(fuzz, 1.0)
        self.ref_idx = ref_idx


def random_in_unit_sphere(shape, device):
    """Generate random points in unit sphere"""
    # Use rejection sampling
    while True:
        p = 2.0 * torch.rand(shape, 3, device=device) - 1.0
        mask = torch.sum(p * p, dim=-1) < 1.0
        if mask.all():
            return p
        p[~mask] = 2.0 * torch.rand((~mask).sum(), 3, device=device) - 1.0


def random_unit_vector(shape, device):
    """Generate random unit vectors"""
    return random_in_unit_sphere(shape, device) / torch.norm(
        random_in_unit_sphere(shape, device), dim=-1, keepdim=True)


def reflect(v, n):
    """Reflect vector v about normal n"""
    return v - 2 * torch.sum(v * n, dim=-1, keepdim=True) * n


def refract(uv, n, etai_over_etat):
    """Refract vector through surface"""
    cos_theta = torch.clamp(-torch.sum(uv * n, dim=-1, keepdim=True), -1.0, 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -torch.sqrt(torch.abs(1.0 - torch.sum(r_out_perp * r_out_perp, dim=-1, keepdim=True))) * n
    return r_out_perp + r_out_parallel


def schlick(cosine, ref_idx):
    """Schlick's approximation for reflectance"""
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * torch.pow((1 - cosine), 5)


def ray_color(ray, world, depth, device):
    """Compute color by tracing ray through scene"""
    batch_size = ray.origin.shape[0]
    color = torch.ones(batch_size, 3, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)

    current_ray = ray

    for _ in range(depth):
        if not active.any():
            break

        # Find closest hit
        closest_t = torch.full((batch_size,), float('inf'), device=device)
        hit_anything = torch.zeros(batch_size, dtype=torch.bool, device=device)
        hit_material = [None] * batch_size
        hit_point = torch.zeros(batch_size, 3, device=device)
        hit_normal = torch.zeros(batch_size, 3, device=device)

        for obj in world:
            valid, t = obj.hit(current_ray, 0.001, closest_t)
            hit_mask = valid & active

            if hit_mask.any():
                # Update closest hits
                update = hit_mask & (t < closest_t)
                closest_t = torch.where(update, t, closest_t)
                hit_anything = hit_anything | update

                # Calculate hit points and normals for updated rays
                for i in range(batch_size):
                    if update[i]:
                        hit_material[i] = obj.material
                        p = current_ray.at(t[i:i + 1])[0]
                        hit_point[i] = p
                        hit_normal[i] = (p - obj.center) / obj.radius

        # Sky color for rays that didn't hit anything
        no_hit = ~hit_anything & active
        if no_hit.any():
            unit_direction = current_ray.direction[no_hit]
            t = 0.5 * (unit_direction[:, 1] + 1.0)
            sky_color = (1.0 - t).unsqueeze(-1) * torch.tensor([1.0, 1.0, 1.0], device=device) + \
                        t.unsqueeze(-1) * torch.tensor([0.5, 0.7, 1.0], device=device)
            color[no_hit] *= sky_color
            active[no_hit] = False

        # Process hits
        if hit_anything.any():
            new_origins = torch.zeros(batch_size, 3, device=device)
            new_directions = torch.zeros(batch_size, 3, device=device)

            for i in range(batch_size):
                if hit_anything[i] and active[i]:
                    mat = hit_material[i]
                    p = hit_point[i:i + 1]
                    normal = hit_normal[i:i + 1]

                    if mat.material_type == 'diffuse':
                        # Lambertian diffuse
                        target = p + normal + random_unit_vector((1,), device)
                        new_directions[i] = (target - p)[0]
                        new_origins[i] = p[0]
                        color[i] *= mat.albedo

                    elif mat.material_type == 'metal':
                        # Metal reflection
                        reflected = reflect(current_ray.direction[i:i + 1], normal)
                        fuzz_offset = mat.fuzz * random_in_unit_sphere((1,), device)
                        new_directions[i] = (reflected + fuzz_offset)[0]
                        new_origins[i] = p[0]

                        # Absorb if reflected below surface
                        if torch.sum(new_directions[i:i + 1] * normal) <= 0:
                            active[i] = False
                        else:
                            color[i] *= mat.albedo

                    elif mat.material_type == 'dielectric':
                        # Glass refraction
                        attenuation = torch.tensor([1.0, 1.0, 1.0], device=device)
                        color[i] *= attenuation

                        front_face = torch.sum(current_ray.direction[i:i + 1] * normal) < 0
                        normal_adj = normal if front_face else -normal
                        etai_over_etat = 1.0 / mat.ref_idx if front_face else mat.ref_idx

                        cos_theta = min(-torch.sum(current_ray.direction[i:i + 1] * normal_adj).item(), 1.0)
                        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

                        cannot_refract = etai_over_etat * sin_theta > 1.0

                        if cannot_refract or schlick(cos_theta, mat.ref_idx) > torch.rand(1, device=device).item():
                            # Reflect
                            new_directions[i] = reflect(current_ray.direction[i:i + 1], normal_adj)[0]
                        else:
                            # Refract
                            new_directions[i] = refract(current_ray.direction[i:i + 1], normal_adj, etai_over_etat)[0]

                        new_origins[i] = p[0]

            current_ray = Ray(new_origins, new_directions)

    return color


def render(width, height, samples_per_pixel, max_depth):
    """Render the scene"""
    aspect_ratio = width / height

    # Camera setup
    lookfrom = torch.tensor([13.0, 2.0, 3.0], device=device)
    lookat = torch.tensor([0.0, 0.0, 0.0], device=device)
    vup = torch.tensor([0.0, 1.0, 0.0], device=device)

    cam = Camera(lookfrom, lookat, vup, 20.0, aspect_ratio, device)

    # Create world
    world = []

    # Ground
    world.append(Sphere(
        torch.tensor([0.0, -1000.0, 0.0], device=device),
        1000.0,
        Material(torch.tensor([0.5, 0.5, 0.5], device=device), 'diffuse')
    ))

    # Three large spheres
    world.append(Sphere(
        torch.tensor([0.0, 1.0, 0.0], device=device),
        1.0,
        Material(torch.tensor([0.0, 0.0, 0.0], device=device), 'dielectric', ref_idx=1.5)
    ))

    world.append(Sphere(
        torch.tensor([-4.0, 1.0, 0.0], device=device),
        1.0,
        Material(torch.tensor([0.4, 0.2, 0.1], device=device), 'diffuse')
    ))

    world.append(Sphere(
        torch.tensor([4.0, 1.0, 0.0], device=device),
        1.0,
        Material(torch.tensor([0.7, 0.6, 0.5], device=device), 'metal', fuzz=0.0)
    ))

    # Random small spheres
    np.random.seed(42)
    for a in range(-11, 11, 2):
        for b in range(-11, 11, 2):
            choose_mat = np.random.random()
            center = torch.tensor([a + 0.9 * np.random.random(), 0.2, b + 0.9 * np.random.random()], device=device)

            if torch.norm(center - torch.tensor([4.0, 0.2, 0.0], device=device)) > 0.9:
                if choose_mat < 0.8:
                    # Diffuse
                    albedo = torch.rand(3, device=device) * torch.rand(3, device=device)
                    world.append(Sphere(center, 0.2, Material(albedo, 'diffuse')))
                elif choose_mat < 0.95:
                    # Metal
                    albedo = 0.5 * (1 + torch.rand(3, device=device))
                    fuzz = 0.5 * np.random.random()
                    world.append(Sphere(center, 0.2, Material(albedo, 'metal', fuzz=fuzz)))
                else:
                    # Glass
                    world.append(Sphere(center, 0.2,
                                        Material(torch.tensor([0.0, 0.0, 0.0], device=device), 'dielectric',
                                                 ref_idx=1.5)))

    print(f"Scene has {len(world)} objects")

    # Render
    img = torch.zeros(height, width, 3, device=device)

    batch_size = 1024  # Process rays in batches for memory efficiency
    total_pixels = width * height

    print(f"Rendering {width}x{height} image with {samples_per_pixel} samples per pixel...")

    for sample in range(samples_per_pixel):
        print(f"Sample {sample + 1}/{samples_per_pixel}")

        for batch_start in range(0, total_pixels, batch_size):
            batch_end = min(batch_start + batch_size, total_pixels)
            batch_pixels = batch_end - batch_start

            # Generate pixel coordinates
            pixel_indices = torch.arange(batch_start, batch_end, device=device)
            j = pixel_indices // width  # row
            i = pixel_indices % width  # col

            # Add random offset for anti-aliasing
            u = (i + torch.rand(batch_pixels, device=device)) / (width - 1)
            v = (j + torch.rand(batch_pixels, device=device)) / (height - 1)

            # Generate rays
            rays = cam.get_ray(u, v)

            # Trace rays
            colors = ray_color(rays, world, max_depth, device)

            # Accumulate colors
            for idx, (row, col) in enumerate(zip(j, i)):
                img[row, col] += colors[idx]

    # Average and gamma correct
    img = img / samples_per_pixel
    img = torch.sqrt(img)  # Gamma 2.0
    img = torch.clamp(img, 0.0, 1.0)

    # Convert to numpy and save
    img_np = (255.99 * img.cpu().numpy()).astype(np.uint8)
    img_np = np.flipud(img_np)  # Flip vertically

    return img_np


if __name__ == "__main__":
    # Render settings
    width = 800
    height = 450
    samples_per_pixel = 10
    max_depth = 8

    print("Starting path tracer...")
    img = render(width, height, samples_per_pixel, max_depth)

    # Save image
    output_path = "/mnt/user-data/outputs/path_traced_scene.png"
    Image.fromarray(img).save(output_path)
    print(f"Image saved to {output_path}")