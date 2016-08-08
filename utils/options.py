import os.path as osp
import json
import os
import sys

class Options:

	def __init__(self, path):

		self.def_opts = {\
		'num_bins':8,\
		'num_pascal': 8,\
		'difficult': False,\
		'sampler': False,\
		'share_pose': True,\
		'imagenet': True,\
		'pose_weight': 1.0,\
		'stepsize': 11000,\
		'rotate': False,\
		'gpu':'0',\
		'max_iter':30000,\
		'base_lr':0.00004,\
		'size':300,
		'resume':True,\
		'full3D':False,\
		'sep3D':False\
		}

		self.opts = self.set_opts(path)
		for k, v in self.def_opts.iteritems():
			if k in self.opts:
				continue
			else:
				self.opts[k] = v


	def set_opts(self, json_path):
		if json_path != '':
			return json.load(open(json_path, 'r'))
		else:
			temp = {}
			return temp


	def get_opts(self, opt_id):
		if opt_id not in self.opts:
			if opt_id in self.def_opts:
				return self.def_opts[opt_id]
			else:
				print '%s not found in opt or default opts' % opt_id
				sys.exit()
		else:
			return self.opts[opt_id]


	def get_db_name_stem(self, split):

		out = '%s_bins=%d_diff=%r_imgnet=%r_numPascal=%d_rotate=%r' \
		% (split, self.get_opts('num_bins'), self.get_opts('difficult'),\
		 self.get_opts('imagenet'), self.get_opts('num_pascal'), self.get_opts('rotate'))
		
		if self.get_opts('full3D'):
			if self.get_opts('sep3D'):
				out = '%s_sep3d' % out
			else:
				out = '%s_joint3d' % out 
		return out


	def add_kv(self, key, val):
		self.opts[key] = val

	def write_opt(self, path):
		with open(path, 'w') as out:
			json.dump(self.opts, out)

