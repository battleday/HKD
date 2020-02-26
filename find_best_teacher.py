import os


def load_best_model(teacher_name, master_path):
	"""Assumes teacher_name specifies a subdirectory
	of master path. Will deep scan through all subsubdirectories
	looking for best performing model."""
	if teacher_name is 'human':
		return {'name': 'human', 'teacherProbs': None}
	else:
		print('NO TEACHER MODEL SEARCH FUNCTION, YET')
