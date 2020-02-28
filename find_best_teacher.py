import os


def load_best_model(teacher_name, master_path):
	"""Assumes teacher_name specifies a subdirectory
	of master path. Will deep scan through all subsubdirectories
	looking for best performing model."""
	print('teacher name fed into find_best_teacher is {0}'.format(teacher_name))
	if teacher_name is 'human':
		print('human teacher')
		return {'name': 'human', 'probs': None}
	elif teacher_name is 'control':
		print('control detected but nott implemented')
	else:
		print('NO TEACHER MODEL SEARCH FUNCTION, YET')
