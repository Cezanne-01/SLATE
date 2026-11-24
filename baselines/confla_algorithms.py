# -*- coding: utf-8 -*-
# CONF-LA algorithm: Kaltenberger, Chen, Thaqi & Kasneci (2026),
# "Sure About That Line? Approaching Confidence-Based, Real-Time Line Assignment
# in Reading Gaze Data", ETRA/PACM CGIT, doi:10.1145/3803540.
# Source: https://gitlab.lrz.de/hctl/conf-la (commit c25442d, retrieved 2026-08-20),
# licensed CC BY 4.0. The functions confidence / _compute_transition_probs /
# velocity_calculation were copied verbatim from code/algorithms.py. The only
# modification: the `import core` dependency is replaced by the shim below, in which
# the physical screen size is set to the tablet used in this study (original: a
# 60 x 34 cm monitor, here 26.7 x 16.7 cm). This constant only enters the
# velocity (cm/s) threshold computation. The three default parameter sets
# (best_params_*) are taken unchanged from the original code/core.py.

import numpy as np


class core:                              # shim replacing the original `import core`
    screen_width = 1920
    screen_height = 1080
    screen_width_cm = 26.7               # tablet used in this study (original: 60)
    screen_height_cm = 16.7              # tablet used in this study (original: 34)


best_params_N = {
    'mu_shift': 1.0, 'sigma_scale': 0.7, 'confidence_threshold': 0.6,
    'window_confidence_threshold': 0.6, 'window_confidence_update_weight': 0.4,
}
best_params_10 = {
    'mu_shift': 1.0, 'sigma_scale': 0.4, 'confidence_threshold': 0.6,
    'window_confidence_threshold': 0.6, 'window_confidence_update_weight': 0.4,
}
best_params_1 = {
    'mu_shift': 1.0, 'sigma_scale': 0.2, 'confidence_threshold': 0.5,
    'window_confidence_threshold': 0.6, 'window_confidence_update_weight': 0.4,
}

def confidence(
	segment_fixations, segment_fixation_times, line_Y, text_width,
	confidence_threshold,
	sigma_scale, mu_shift,
	window_confidence_threshold,
	window_confidence_update_weight,
	return_sweep_threshold=300,
	sudden_x_change_threshold=0.8,
	window_size=5,
	forward_pass_on=None
): 
	saccades = velocity_calculation(segment_fixations, segment_fixation_times)
	n = len(segment_fixations)
	line_height = line_Y[1] - line_Y[0]
	n_lines = len(line_Y)
	fixation_x = segment_fixations[:, 0]
	fixation_y = segment_fixations[:, 1]
	fixation_start = segment_fixation_times[:, 0]
	
	# calculate distance probabilities
	sigma = line_height * sigma_scale
	distances = np.abs(fixation_y[:, np.newaxis] - line_Y[np.newaxis, :] * mu_shift)
	distance_probs = np.exp(-0.5 * (distances / sigma) ** 2)
	distance_probs = distance_probs / distance_probs.sum(axis=1, keepdims=True)

	# calculate context feautres
	dx = np.zeros(n)
	dx[1:] = fixation_x[1:] - fixation_x[:-1]
	sudden_change_masks = dx < -sudden_x_change_threshold * text_width

	saccade_map = {}
	saccade_velocities = np.zeros(n)
	for i in range(1, len(fixation_start)):
		prev_time = fixation_start[i-1]
		curr_time = fixation_start[i]

		for saccade in saccades:
			if prev_time <= saccade[0] <= curr_time:
				saccade_map[i] = saccade
				break
			else:
				saccade_map[i] = None
	for i, saccade in saccade_map.items():
		if saccade:
			saccade_velocities[i] = saccade[-1]

	return_sweep_masks = saccade_velocities > return_sweep_threshold # cm/s threshold
	context = {
		'sudden_x_change':sudden_change_masks,
		'return_sweep_saccade':return_sweep_masks
	}
	
	# forward-backward algorithm
	if forward_pass_on is None:
		forward_pass_on = np.ones(n_lines)
	forward_probs = np.vstack([forward_pass_on, distance_probs.copy()])

	for i in range(len(distance_probs)):
		transition = _compute_transition_probs(context, i, n_lines)

		prior = np.zeros(n_lines)
		for curr_line in range(n_lines):
			for prev_line in range(n_lines):
				prior[curr_line] += (
					transition[prev_line, curr_line] * forward_probs[i, prev_line]
				)
		fp_idx = i + 1
		forward_probs[fp_idx] = distance_probs[i] * prior
		fp_idx_sum = np.sum(forward_probs[fp_idx])

		if fp_idx_sum > 0:
			forward_probs[fp_idx] /= fp_idx_sum

	forward_probs = forward_probs[1:]


	backward_probs = np.ones((n, n_lines))
	for i in range(n-2, -1, -1):
		transition = _compute_transition_probs(context, i+1, n_lines)
		
		for curr_line in range(n_lines):
			prob_sum = 0
			for next_line in range(n_lines):
				prob_sum += (
					transition[curr_line, next_line] * backward_probs[i+1, next_line] * distance_probs[i+1, next_line]
				)
			backward_probs[i, curr_line] = prob_sum
		
		bp_idx_sum = np.sum(backward_probs[i])
		if bp_idx_sum > 0:
			backward_probs[i] /= bp_idx_sum

	final_probs = forward_probs * backward_probs
	final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
	assigned_lines = np.argmax(final_probs, axis=1)
	confidences = np.max(final_probs, axis=1)

	corrected_fixations = []
	for i, fixation in enumerate(segment_fixations):
		corrected_fix = fixation.copy()                          

		if confidences[i] >= confidence_threshold:
			corrected_fix[1] = line_Y[assigned_lines[i]]
		corrected_fixations.append(corrected_fix)

	if len(corrected_fixations) > 1:
		for i in range(n):
			if confidences[i] >= confidence_threshold:
				continue
			
			window_start = max(0, i - window_size -1)
			window_end = min(n, i + window_size + 1)
			
			neighbor_lines = assigned_lines[window_start:window_end]
			line_counts = np.bincount(neighbor_lines, minlength=len(line_Y))
			majority_line = np.argmax(line_counts)

			if line_counts[majority_line] >= len(neighbor_lines) * window_confidence_threshold:
				if assigned_lines[i] == majority_line:
					win_i_confidence = confidences[i] * (1 - window_confidence_update_weight) + (line_counts[majority_line] / len(neighbor_lines)) * window_confidence_update_weight  # update confidence
					if win_i_confidence < confidence_threshold:
						continue
				
				corrected_fixations[i][1] = line_Y[majority_line]
				assigned_lines[i] = majority_line
	
		# adaptive mu_shift
		dis = fixation_y - line_Y[assigned_lines]
		neg_count = np.sum(dis < 0)
		pos_count = np.sum(dis > 0)

		if neg_count > pos_count:
			mu_shift = 0.99
		elif neg_count < pos_count:
			mu_shift = 1.01
		else:
			mu_shift = 1.0

	return (np.array(corrected_fixations, dtype=int), forward_probs[-1], mu_shift)


def _compute_transition_probs(context, i, n_lines):

	transition = np.sum(
		[np.eye(n_lines, k=k) * v for k, v in zip([-1, 0, 1], [0.03, 0.85, 0.1])],
		axis=0
	)
	non_adjacent_eye = (
		np.sum([np.eye(n_lines, k=k) for k in range(2, n_lines)], axis=0)
		+ np.sum([np.eye(n_lines, k=-k) for k in range(2, n_lines)], axis=0)
	)
	transition += np.multiply(
		np.divide((1.0 - np.sum(transition, axis=1)), np.sum(non_adjacent_eye, axis=0))[:, np.newaxis], 
		non_adjacent_eye
	)

	# adjust weight based on return signal
	return_sweep = context['return_sweep_saccade'][i]
	sudden_x_change = context['sudden_x_change'][i]

	return_sweep_context_len = len(context['return_sweep_saccade'])
	sudden_x_change_context_len = len(context['sudden_x_change'])

	if i < return_sweep_context_len and return_sweep:
		for j in range(n_lines - 1):
			transition[j, j] = 0.05
			transition[j, j+1] = 0.90
			if j > 0:
				transition[j, j-1] = 0.03
	elif i < sudden_x_change_context_len and sudden_x_change:
		for j in range(n_lines - 1):
			transition[j, j] = 0.3
			transition[j, j+1] = 0.70
			if j > 0:
				transition[j, j-1] = 0.03

	row_sums = transition.sum(axis=1, keepdims=True)
	transition = transition / row_sums

	return transition

def velocity_calculation(fixations, fixation_times):

	px_per_cm_x = core.screen_width / core.screen_width_cm
	px_per_cm_y = core.screen_height / core.screen_height_cm

	saccades = []
	for i in range(len(fixations) - 1):
		_, start = fixation_times[i]
		end, _ = fixation_times[i+1]
		x1, y1 = fixations[i]
		x2, y2 = fixations[i+1]

		saccade_duration = end - start
		if saccade_duration <= 0:
			continue

		dx = (x2 - x1) / px_per_cm_x
		dy = (y2 - y1) / px_per_cm_y
		distance = np.sqrt(dx**2 + dy**2)

		mean_velocity = (distance / saccade_duration) * 1000
		peak_velocity = mean_velocity * 1.5

		saccades.append([start, end, mean_velocity, peak_velocity])

	return saccades
