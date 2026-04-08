import numpy as np
import matplotlib.pyplot as plt
from slam_project.mapping import mapper as mp
from slam_project.models.occupancy_grid import occupancy_grid
from slam_project.localization import odometry as od
from slam_project.io import load_data as ld

def slam(room_name):
    # Noise Parameters
    movement_noise = 0.025
    angle_noise = 0.05

    # Initialization (num_particles x (x,y,theta))
    num_particles = 100
    particles = np.zeros([num_particles, 3])
    particles[:, 0] = np.random.normal(0, movement_noise, num_particles)
    particles[:, 1] = np.random.normal(0, movement_noise, num_particles)
    particles[:, 2] = np.random.normal(0, angle_noise, num_particles)
    weights = np.ones(num_particles) * (1/num_particles)

    # Odometry Step
    room_name = room_name
    enc_path = f'./data/Encoders{room_name}'
    imu_path = f'./data/imu{room_name}'
    lidar_path = f'./data/Hokuyo{room_name}'
    delta_forward, delta_perp, delta_theta, ts = od.compute_body_frame_odometry_at_lidar_times(
        enc_path, imu_path, lidar_path, alpha=0.0
    )

    lidar = ld.get_lidar(lidar_path)

    best_grid = occupancy_grid(0, 0, 0.1, (-25,35), f"Room {room_name}")
    best_trajectory = []

    # For each lidar scan:
    # 1. Predict each particle pose from odometry plus noise.
    # 2. Score each particle against the current map.
    # 3. Add only the best particle's scan to the map.
    # 4. Resample if the particle set has collapsed.
    num_scan = len(lidar)
    best_grid.init_live_plot()
    for idx, scan in enumerate(lidar):
        
        d_forward = delta_forward[idx]
        d_perp = delta_perp[idx]
        dtheta = delta_theta[idx]

        particle_hits = []
        best_particle = particles[0]

        if (idx % 25 == 0):
            best_grid.update_live_plot(best_particle[0], best_particle[1])

        for p, particle in enumerate(particles):
            theta_mid = particle[2] + 0.5 * dtheta

            particle[0] += d_forward * np.cos(theta_mid) - d_perp * np.sin(theta_mid)
            particle[1] += d_forward * np.sin(theta_mid) + d_perp * np.cos(theta_mid)
            particle[2] += dtheta

            # Add motion noise after the deterministic odometry update.
            particle[0] += np.random.normal(0, movement_noise)
            particle[1] += np.random.normal(0, movement_noise)
            particle[2] += np.random.normal(0, angle_noise)

            ox, oy = mp.lidar_hits_global(
                scan,
                particle[0],
                particle[1],
                particle[2],
                res=5
            )
            ox_grid, oy_grid = best_grid.hits_to_grid_cells(ox, oy)
            miss_x_grid, miss_y_grid = best_grid.miss_to_grid_cells(ox_grid, oy_grid, particle[0], particle[1])
            particle_hits.append((ox, oy, ox_grid, oy_grid))

            if ox_grid.size == 0:
                likelihood = 1e-12
            else:
                # hits should be high score, misses should be low score
                # added if and else statements incase no lidar hits or misses were found for some reason...
                hit_score = np.mean(best_grid.grid[oy_grid, ox_grid]) if ox_grid.size else -5.0
                miss_score = np.mean(best_grid.grid[miss_y_grid, miss_x_grid]) if miss_x_grid.size else 0.0
                score = hit_score - miss_score
                likelihood = np.exp(np.clip(score, -10, 10))

            weights[p] *= likelihood

        weight_sum = np.sum(weights)
        if weight_sum == 0 or not np.isfinite(weight_sum):
            weights = np.ones(num_particles) / num_particles
        else:
            weights /= weight_sum

        # Add only the best particle's scan to the occupancy grid.
        best_idx = np.argmax(weights)
        best_particle = particles[best_idx]
        best_trajectory.append(best_particle.copy())
        best_ox, best_oy, _, _ = particle_hits[best_idx]
        # print(f"best particle's weight = {weights[best_idx]}")

        # apply threshold on when best particle updates map, I want only highly confident particles to contribute
        # threshold starts only after some warm-up period so that the particles can understand their starting pos.
        if idx < 10 or weights[best_idx] >= 0.05:
            best_grid.add_scan_hits(best_ox, best_oy, best_particle[0], best_particle[1])

        # Resample only when too few particles carry most of the weight.
        n_eff = np.sum(weights)**2 / np.sum(weights ** 2)
        if n_eff < 0.5 * num_particles:
            indices = np.random.choice(
                np.arange(num_particles),
                size=num_particles,
                p=weights,
                replace=True,
            )
            particles = particles[indices].copy()
            weights = np.ones(num_particles) / num_particles

    # fig, ax = plt.subplots()
    # best_grid.plot(fig, ax, None, None)
    best_trajectory = np.asarray(best_trajectory)
    best_grid.add_traj_live_plot(f"./outputs/figures/SLAM_map_{room_name}")