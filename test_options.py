#from ssd_pose.utils import options
from utils import options

if __name__ == "__main__":
	test = options.Options('configs/test.json')
	print test.get_db_name_stem('train')
	test.dummy_test()