import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
# import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSE_PATH = Path('/Users/YPL/Downloads/20261006-205740_test4_triangulated.npy')

NUM_KEYPOINTS = 20

SKELETON_EDGES = np.array([
	[0, 1], # head-neck
	[1, 4], # neck-collar
	[4, 5], # collar-leftshoulder
	[5, 6], # leftshoulder-leftelbow
	[6, 7], # leftelbow-leftwrist
	# [7, 8], # leftwrist-lefthand
	[4, 2], # collar-torso
	[4, 10], # collar-rightshoulder
	[10, 11], # rightshoulder-rightelbow
	[11, 12], # rightelbow-rightwrist
	# [12, 13], # rightwrist-righthand
	[2, 3], # torso-waist
	[3, 14], # waist-lefthip
	[3, 17], # waist-righthip
	[14, 15], # lefthip-leftknee
	[15, 16], # leftknee-leftankle
	[17, 18], # righthip-rightknee
	[18, 19] # rightknee-rightankle
	])

SKELETON_EDGES_CMU = np.array([
	# 0, Nose
    # 1, Neck
    # 2, RShoulder, 3, RElbow, 4, RWrist
    # 5, LShoulder, 6, LElbow, 7, LWrist
    # 8, RHip, 9, RKnee 10 RAnkle
    # 11 LHip, 12 LKnee, 13 LAnkle
    # 14 REye, 15 LEye, 16 REar, 17 LEar
	[0, 1], # Nose-Neck
	[1, 2], # Neck-RShoulder
	[2, 3], # RShoulder-RElbow
	[3, 4], # RElbow-RWrist
	[1, 5], # Neck-LShoulder
	[5, 6], # LShoulder-LElbow
	[6, 7], # LElbow-LWrist
	[1, 8], # Neck-RHip
	[8, 9], # RHip-RKnee
	[9, 10], # Rknee-RAnkle
	[1, 11], # Neck-LHip
	[11, 12], # LHip-LKnee
	[12, 13], # LKnee-LAnkle
	])

LIMB_COLORS = {
	'Head / torso': '#334155',
	'Right arm': '#E76F51',
	'Left arm': '#3B82F6',
	'Right leg': '#E9A23B',
	'Left leg': '#17A398',
}

# Same order as SKELETON_EDGES_CMU.
EDGE_GROUPS = (
	'Head / torso',
	'Right arm', 'Right arm', 'Right arm',
	'Left arm', 'Left arm', 'Left arm',
	'Head / torso', 'Right leg', 'Right leg',
	'Head / torso', 'Left leg', 'Left leg',
)

def find_skeleton_bounds(args, skeletons):
	# find the min and max of each axis
	if not np.isfinite(skeletons).any():
		raise ValueError("No valid 3D joints to display")
	x_min = np.nanmin(skeletons[:, :, 0])
	x_max = np.nanmax(skeletons[:, :, 0])
	y_min = np.nanmin(skeletons[:, :, 1])
	y_max = np.nanmax(skeletons[:, :, 1])
	z_min = np.nanmin(skeletons[:, :, 2])
	z_max = np.nanmax(skeletons[:, :, 2])

	# return the bounds
	return (x_min, x_max), (y_min, y_max), (z_min, z_max)

# def update_lines(num, data, lines):
# 	# num = 10
# 	for line, dat in zip(lines, data):
# 		# NOTE there is no .set_data() for 3 dim data...
# 		line.set_data(dat[num, [0, 2], :])
# 		line.set_3d_properties(dat[num, 1, :])
# 	return lines

def format_timestamp(seconds):
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(int(minutes), 60)
	if hours:
		return f'{hours:02d}:{minutes:02d}:{seconds:05.2f}'
	return f'{minutes:02d}:{seconds:05.2f}'

def make_title(pose_name, frame_idx, total_frames, fps):
	elapsed = frame_idx / fps
	total_time = (total_frames - 1) / fps
	return (
		f'{pose_name}\n'
		f'Frame {frame_idx + 1:04d} / {total_frames:04d}   |   '
		f'Time {format_timestamp(elapsed)} / {format_timestamp(total_time)}   |   '
		f'{fps:g} FPS'
	)

def update_lines(num, data, lines, title_artist, pose_name, total_frames, fps):
    for line, dat in zip(lines, data):
        # The triangulated pose uses Z for height; the axis is inverted below.
        line.set_data(dat[num, 0, :], dat[num, 1, :])
        line.set_3d_properties(dat[num, 2, :])
    title_artist.set_text(make_title(pose_name, num, total_frames, fps))
    return [*lines, title_artist]

def generate_skeleton_lines(args, skeletons):
	# goal to create a list of lines, each line is a (length, 2, 3)

	lines = []
	for skeleton_edge in SKELETON_EDGES_CMU:
		line = np.concatenate(
			 (np.expand_dims(skeletons[:, skeleton_edge, 0], axis=2), 
			 np.expand_dims(skeletons[:, skeleton_edge, 1], axis=2), 
			 np.expand_dims(skeletons[:, skeleton_edge, 2], axis=2)
			 )
			, 
			axis=2)
		lines.append(np.transpose(line, (0, 2, 1)))
	return lines

def view(args):
	if args.dataset_type is None:
		pose_path = Path(args.pose_path).expanduser()
	elif args.dataset_type == "rt":
		pose_path = Path(args.intermediates_root) / args.exp_name / "triangulated_poses" / f"{args.exp_name}_triangulated.npy"
	elif args.dataset_type == "data":
		date = args.exp_name.split('-')[0]
		pose_path = Path(args.data_root) / date / "pose3D" / f"pose3D_{args.exp_name}.npy"

	if not pose_path.is_file():
		raise FileNotFoundError(f"3D pose file not found: {pose_path}")
	skeletons = np.load(pose_path)
	if skeletons.ndim != 3 or skeletons.shape[1] < 14 or skeletons.shape[2] != 3:
		raise ValueError(
			f"Expected pose shape (frames, joints>=14, xyz=3), got {skeletons.shape}"
		)
	print(f"Loaded 3D pose: {pose_path} | shape={skeletons.shape}")

	length = skeletons.shape[0]
	if length == 0:
		raise ValueError('The 3D pose file contains no frames')
	if args.max_fps <= 0:
		raise ValueError('--max_fps must be greater than 0')

	interval = 1000 / args.max_fps

	# Attaching 3D axis to the figure
	fig = plt.figure(figsize=(9, 7), facecolor='#F7F9FC')
	# ax = p3.Axes3D(fig)
	ax = fig.add_subplot(projection="3d")
	ax.set_facecolor('#F7F9FC')
	title_artist = fig.suptitle(
		make_title(pose_path.name, 0, length, args.max_fps),
		fontsize=12, color='#243047', y=0.965, linespacing=1.35,
	)

	# Setting the axes properties
	if args.limit_type == 'responsive':
		xlim3d, ylim3d, zlim3d = find_skeleton_bounds(args, skeletons)
		# resize to make it look correct in aspect ratio
		x_range = xlim3d[1] - xlim3d[0]
		y_range = ylim3d[1] - ylim3d[0]
		z_range = zlim3d[1] - zlim3d[0]
		max_range = max(x_range, y_range, z_range)
		x_center = (xlim3d[1] + xlim3d[0]) / 2
		y_center = (ylim3d[1] + ylim3d[0]) / 2
		z_center = (zlim3d[1] + zlim3d[0]) / 2
		xlim3d = (x_center - max_range / 2, x_center + max_range / 2)
		ylim3d = (y_center - max_range / 2, y_center + max_range / 2)
		zlim3d = (z_center - max_range / 2, z_center + max_range / 2)
	elif args.limit_type == 'fixed':
		xlim3d = (-1000.0, 1000.0)
		ylim3d = (1000.0, 3000.0)
		zlim3d = (-750.0, 1200.0)

	# ax.set(xlim3d=xlim3d, xlabel='X')
	# ax.set(ylim3d=ylim3d, ylabel='Y')
	# ax.set(zlim3d=zlim3d, zlabel='Z')
	ax.set(xlim3d=xlim3d, xlabel='X')
	ax.set(ylim3d=ylim3d, ylabel='Y')
	ax.set(zlim3d=zlim3d, zlabel='Z')
	ax.set_box_aspect((1, 1, 1))
	for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
		axis.set_major_locator(MaxNLocator(nbins=5))
		axis.line.set_color('#A8B5C6')
		axis.line.set_linewidth(0.7)
		axis._axinfo['grid'].update(color=(0.38, 0.48, 0.60, 0.24), linewidth=0.55, linestyle=':')
		axis.pane.set_facecolor((0.91, 0.95, 0.99, 0.22))
		axis.pane.set_edgecolor((0.55, 0.64, 0.74, 0.30))
		axis.pane.set_linewidth(0.6)
		axis.label.set_color('#475569')
	ax.tick_params(colors='#64748B', labelsize=8, pad=2)
	# reverse z axis to have better view
	ax.invert_yaxis()
	ax.invert_zaxis()

	# edges lines of 3D lines, each contain (length, 2, 3)
	data = generate_skeleton_lines(args, skeletons)

	# lines should be list of (m, 3, 2) where 2 is two endpoint, 3 is xyz and m is times
	# lines = [ax.plot(dat[0, [0, 1, 2], 0], dat[0, [0, 1, 2], 1])[0] for dat in data]
	# lines = [ax.plot(dat[0, 0, :], dat[0, 1, :], dat[0, 2, :])[0] for dat in data]
	lines = [
		ax.plot(
			dat[0, 0, :], dat[0, 1, :], dat[0, 2, :],
			color=LIMB_COLORS[group], linewidth=2.8,
			marker='o', markersize=4.5, markeredgecolor='white', markeredgewidth=0.6,
			solid_capstyle='round',
		)[0]
		for dat, group in zip(data, EDGE_GROUPS)
	]
	fig.legend(
		handles=[
			Line2D([0], [0], color=color, linewidth=2.8, marker='o', markersize=5, label=group)
			for group, color in LIMB_COLORS.items()
		],
		loc='lower center', ncol=5, frameon=False, fontsize=9,
		bbox_to_anchor=(0.5, 0.02),
	)
	fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.12)

	# Creating the Animation object
	line_ani = animation.FuncAnimation(
		fig, update_lines, length,
		fargs=(data, lines, title_artist, pose_path.name, length, args.max_fps),
		interval=interval, blit=False)

	plt.show()

	print()

if __name__ == "__main__":
	# parser
	parser = argparse.ArgumentParser()

	# data config
	parser.add_argument(
		'--pose-path',
		default=str(DEFAULT_POSE_PATH),
		help='Direct path to a triangulated 3D pose .npy file (default mode)',
	)
	parser.add_argument(
		'--dataset-type',
		type=str,
		help='Use a pipeline-resolved path instead of --pose-path',
		choices=["data", "rt"],
		default=None,
	)
	parser.add_argument('--data-root', default=str(PIPELINE_ROOT / 'data'))
	parser.add_argument('--intermediates-root', default=str(PIPELINE_ROOT.parent / 'CSI_files_2026' / 'intermediates'))
	parser.add_argument('--exp-name', default = 'hand')
	parser.add_argument('--max_fps', type = int, default = 30)
	parser.add_argument('--limit_type', default='responsive', choices=['responsive', 'fixed'])

	# parse parser
	args = parser.parse_args()

	# start the viewer
	view(args)
