import os
import numpy as np
import matplotlib.pyplot as plt


class PointCloud_Generator:
    def __init__(self, args):
        self.args = args

    def generate_point_cloud(self, result):
        """Convert heatmaps to ``[azimuth_deg, ToF_ns, fd_Hz, dB]`` points.

        ``result`` is the list returned by
        ``Azi_ToF.gen_Doppler_projection_spectrum``. Each heatmap is filtered
        independently using the stricter of the configured percentile and
        dynamic-range thresholds. The returned intensity is the original
        MUSIC pseudo-spectrum value in dB.
        """
        if result is None:
            raise ValueError("result cannot be None")
        results = [result] if isinstance(result, dict) else list(result)
        if not results:
            return np.empty((0, 4), dtype=float)

        percentile = float(
            getattr(self.args, "point_cloud_percentile", 95.0)
        )
        dynamic_range_db = float(
            getattr(self.args, "point_cloud_dynamic_range_db", 10.0)
        )
        max_points = int(
            getattr(self.args, "point_cloud_max_points", 60000)
        )
        if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
            raise ValueError("point-cloud percentile must be in [0, 100]")
        if not np.isfinite(dynamic_range_db) or dynamic_range_db < 0.0:
            raise ValueError("point-cloud dynamic range must be non-negative")
        if max_points < 1:
            raise ValueError("point-cloud max_points must be positive")

        clouds = []
        required_keys = {"fd", "theta_grid", "tau_grid", "spectrum_db"}
        for map_idx, item in enumerate(results):
            if not isinstance(item, dict):
                raise TypeError(f"result[{map_idx}] must be a dictionary")
            missing = required_keys.difference(item)
            if missing:
                raise KeyError(
                    f"result[{map_idx}] is missing keys: {sorted(missing)}"
                )

            fd = float(item["fd"])
            theta_grid = np.asarray(item["theta_grid"], dtype=float)
            tau_grid = np.asarray(item["tau_grid"], dtype=float)
            spectrum_db = np.asarray(item["spectrum_db"], dtype=float)

            if not np.isfinite(fd):
                raise ValueError(f"result[{map_idx}]['fd'] must be finite")
            if theta_grid.ndim != 1 or theta_grid.size == 0:
                raise ValueError(
                    f"result[{map_idx}]['theta_grid'] must be non-empty 1-D"
                )
            if tau_grid.ndim != 1 or tau_grid.size == 0:
                raise ValueError(
                    f"result[{map_idx}]['tau_grid'] must be non-empty 1-D"
                )
            expected_shape = (theta_grid.size, tau_grid.size)
            if spectrum_db.shape != expected_shape:
                raise ValueError(
                    f"result[{map_idx}]['spectrum_db'] must have shape "
                    f"{expected_shape}, got {spectrum_db.shape}"
                )
            if not np.all(np.isfinite(theta_grid)) or not np.all(
                np.isfinite(tau_grid)
            ):
                raise ValueError(
                    f"result[{map_idx}] grids must contain only finite values"
                )

            finite = np.isfinite(spectrum_db)
            finite_values = spectrum_db[finite]
            if finite_values.size == 0:
                continue

            threshold_db = max(
                float(np.max(finite_values)) - dynamic_range_db,
                float(np.percentile(finite_values, percentile)),
            )
            selected = finite & (spectrum_db >= threshold_db)
            theta_idx, tau_idx = np.nonzero(selected)
            if theta_idx.size == 0:
                continue

            azimuth = theta_grid[theta_idx]
            tof_ns = tau_grid[tau_idx] * 1e9
            doppler = np.full(theta_idx.size, fd, dtype=float)
            intensity = spectrum_db[theta_idx, tau_idx]
            clouds.append(
                np.column_stack((azimuth, tof_ns, doppler, intensity))
            )

        if not clouds:
            points = np.empty((0, 4), dtype=float)
            self._save_point_cloud(points)
            return points

        points = np.concatenate(clouds, axis=0)
        if points.shape[0] > max_points:
            keep = np.argpartition(points[:, 3], -max_points)[-max_points:]
            points = points[keep]

        self._save_point_cloud(points)
        
        return points

    def _save_point_cloud(self, points):
        """Save an Azimuth-ToF-Doppler scatter colored by Doppler."""
        pics_dir = getattr(self.args, "pics_dir", None)
        if pics_dir is None:
            return

        frame_idx = int(getattr(self.args, "frame_idx", 0))
        point_size = float(getattr(self.args, "point_cloud_point_size", 3.0))
        point_alpha = float(getattr(self.args, "point_cloud_point_alpha", 0.05))
        if not np.isfinite(point_size) or point_size <= 0.0:
            raise ValueError("point_cloud_point_size must be positive")
        if not np.isfinite(point_alpha) or not 0.0 <= point_alpha <= 1.0:
            raise ValueError("point_cloud_point_alpha must be in [0, 1]")

        save_dir = os.path.join(pics_dir, "Point_Cloud")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{frame_idx}_pc.png")

        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        if points.size:
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=points[:, 2],
                cmap="bwr",
                vmin=float(np.min(points[:, 2])),
                vmax=float(np.max(points[:, 2])),
                marker="o",
                s=point_size,
                alpha=point_alpha,
                depthshade=False,
            )
            
            # 將 colorbar 獨立指派給變數 cbar
            cbar = fig.colorbar(
                scatter,
                ax=ax,
                label="Doppler frequency (Hz)",
                shrink=0.6,
                pad=0.1,
            )
            
            # 強制將 colorbar 的透明度設為 1 (完全不透明) 並重新繪製
            cbar.set_alpha(1.0)


        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("ToF (ns)")
        ax.set_zlabel("Doppler frequency (Hz)")
        ax.set_title(f"Raw Azimuth-ToF-Doppler Point Cloud @ frame {frame_idx}")
        ax.view_init(elev=20, azim=-45)
        fig.savefig(save_path, dpi=120)


        #plt.close(fig)
        plt.show()
        print(f"Saved: {save_path}")
