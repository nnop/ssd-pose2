import numpy as np
import sys
import os.path as osp
import os
import scipy.io as sio
import json
import random
import argparse
import shutil
from random import shuffle



'''
This script preprocesses the annotations for ObjectNet3D
''' 

class MakeAnns:

    def __init__(self, opt):
        self.opt = opt

    def run_main(self):

        base_path = 'data/ObjectNet3D/'

        if not osp.exists(osp.join(base_path, 'all_anns.json')):
            makeAllAnns()
        '''

        data = json.load(open('data/ObjectNet3D/all_anns.json', 'r'))
        if not osp.exists('data/ObjectNet3D/map.txt'):
            createLabelMap(data)

        train_dir = self.opt.get_db_name_stem('train')
        val_dir = self.opt.get_db_name_stem('val')
        tes_dir = self.opt.get_db_name_stem('test')
        
        if not osp.exists(osp.join(base_path, 'cache/')):
            os.mkdir(osp.join(base_path, 'cache/'))
        
        splits = [train_dir, val_dir, tes_dir]
        for split in splits:
            if not osp.exists(osp.join(base_path, 'cache', split)):
                os.mkdir(osp.join(base_path, 'cache', split))

        tr = False
        val = False
        test = False

        print train_dir

        # always filter difficult ?
        if not self.opt.get_opts('difficult'):
            filterDifficult(data)

        if not osp.exists(osp.join(base_path, 'cache', train_dir, 'train.txt')):
            tr = True
            trList = []

        if not osp.exists(osp.join(base_path, 'cache', val_dir, 'val.txt')):
            vaList = []
            val = True
        
        if not osp.exists(osp.join(base_path, 'cache', tes_dir, 'test.txt')):
            teList = []
            test = True

        rot = self.opt.get_opts('rotate')
        bins = self.opt.get_opts('num_bins')
        imnet = self.opt.get_opts('imagenet')
        npascal = self.opt.get_opts('num_pascal')
        
        for idx, ann in data.iteritems():
            annLoc = osp.join(getAnnPath(ann, train_dir, val_dir, tes_dir), idx + '.json')
            output = getImPath(ann) + ' ' + annLoc + '\n'


            binAngles(ann, bins, rot)

            json.dump(ann, open(annLoc, 'w'))

            if ann['split'] == 'train' and tr:
                # append to train list
                if ann['database'] == 'ImageNet':
                    if imnet:
                        trList.append(output)
                else:
                    for _ in xrange(npascal):
                        trList.append(output)
            elif ann['split'] == 'val' and val:
                # append to val list
                vaList.append(output)
            elif ann['split'] == 'test' and test:
                # append to test list
                teList.append(output)
        
        if tr:
            with open(osp.join(base_path, 'cache', train_dir, 'train.txt'), 'w') as outfile:
                shuffle(trList)
                for line in trList:
                    outfile.write(line)
        if val:
            with open(osp.join(base_path, 'cache', val_dir, 'val.txt'), 'w') as outfile:
                shuffle(vaList)
                for line in vaList:
                    outfile.write(line)
        if test:
            with open(osp.join(base_path, 'cache', tes_dir, 'test.txt'), 'w') as outfile:
                shuffle(teList)
                for line in teList:
                    outfile.write(line)
        '''


def filterDifficult(data):
    keysToRemove = []
    count = 0
    for key, ann in data.iteritems():
        newList = []
        for obj in ann['annotation']:
            if not obj['difficult']:
                newList.append(obj)
            else:
                count += 1
        if len(newList) == 0: keysToRemove.append(key)
        ann['annotation'] = newList
    
    for rem in keysToRemove:
        del data[rem]


def createLabelMap(data):
    objClasses = set([obj['category_id'] for ann in data.itervalues() for obj in ann['annotation'] if 'viewpoint' in obj and obj['category_id'] != 'bottle' ])
    labelToCls = dict((idx+1, cls) for idx, cls in enumerate(objClasses))
    f = open('data/ObjectNet3D/map.txt', 'w')
    for label, name in labelToCls.iteritems():
        f.write(name + ' ' + str(label) + ' ' + name + '\n')
    f.close()


def getImPath(ann):
    path = osp.join('data/ObjectNet3D/Images/', ann['filename'])
    return path


def getAnnPath(ann, train_dir, val_dir, tes_dir):
    if ann['split'] == 'train':
        path = osp.join('data/ObjectNet3D/cache', train_dir)
    elif ann['split'] == 'val':
        path = osp.join('data/ObjectNet3D/cache', val_dir)
    elif ann['split'] == 'test':
        path = osp.join('data/ObjectNet3D/cache', tes_dir)

    return path


def binAngles(ann, bins, rotate=False):
    offset = 0
    if rotate:
        offset = 360 / (bins * 2)
    for obj in ann['annotation']:
        if 'viewpoint' in obj:
            azi = obj['viewpoint']['azimuth_coarse']
            
            if 'azimuth' in obj['viewpoint'] and obj['viewpoint']['distance'] != 0.0:
                azi = obj['viewpoint']['azimuth']


            # bin = int(azi / (360/bins))
            bin = int(((azi + offset) % 360) / (360/bins))
            obj['aziLabel'] = bin
            flipAzi = 360 - azi
            obj['aziLabelFlip'] = int(((flipAzi + offset) % 360) / (360/bins))
        else:
            print 'woah where yo angle'



def makeAllAnns():
    base_path = 'data/ObjectNet3D'

    all_anns = {}
    anns = os.listdir(osp.join(base_path, 'Annotations'))
    for idx, ann in enumerate(anns):
        if idx  % 1000 == 0: 
            print 'processing file %d/%d' %(idx, len(anns))

        data = sio.loadmat(osp.join(base_path, 'Annotations', ann), squeeze_me=True, struct_as_record=False)
        cur_ann = get_im_info(data)
        obj_info = get_obj_info(data)
        cur_ann['annotation'] = obj_info
        fi = ann

        if fi[:-4] in all_anns:
            all_anns[fi[:-4]]['annotation'] += cur_ann['annotation']
        else:
            all_anns[fi[:-4]] = cur_ann

    all_anns = splitData(all_anns)
    #convertBbox(all_anns)

    json.dump(all_anns, open(osp.join(base_path, 'all_anns.json'), 'w'))
    print 'json dumped'


def convertBbox(data):
    for ann in data.itervalues():
        for obj in ann['annotation']:

            if obj['bbox'][2] < obj['bbox'][0]:
                temp = obj['bbox'][0]
                obj['bbox'][0] = obj['bbox'][2]
                obj['bbox'][2] = temp 
                print 'a' 
                print ann['filename']

            if obj['bbox'][3] < obj['bbox'][1]:
                temp = obj['bbox'][1]
                obj['bbox'][1] = obj['bbox'][3]
                obj['bbox'][3] = temp
                print b
                print ann['filename']

            if obj['bbox'][0] < 0:
                obj['bbox'][0] = 0

            if obj['bbox'][1] < 0:
                obj['bbox'][1] = 0

            if obj['bbox'][2] > ann['image']['width']:
                obj['bbox'][2] = ann['image']['width']
            if obj['bbox'][3] > ann['image']['height']:
                obj['bbox'][3] = ann['image']['height']

            obj['bbox'][2] = obj['bbox'][2] - obj['bbox'][0]
            obj['bbox'][3] = obj['bbox'][3] - obj['bbox'][1]


def splitData(data):
    f = open('data/ObjectNet3D/Image_sets/train.txt', 'r')
    train = [line.strip('\n') for line in f ]
    f.close()

    f = open('data/ObjectNet3D/Image_sets/val.txt', 'r')
    val = [line.strip('\n') for line in f ]
    f.close()

    f = open('data/ObjectNet3D/Image_sets/test.txt', 'r')
    test = [line.strip('\n') for line in f ]
    f.close()

    print 'training data %d' % len(train)
    print 'val data %d' % len(val)
    print 'test data %d' % len(test)

    for idx, ann in data.iteritems():
        if idx in train:
            ann['split'] = 'train'
        elif idx in val:
            ann['split'] = 'val'
        elif idx in test:
            ann['split'] = 'test'
        else:
            print 'woah'
            print idx
            sys.exit()

    return data


def get_obj_info(data):
    
    list_obj = []
    attrs = ['bbox', 'class', 'viewpoint', 'difficult', 'cad_index', 'subtype',\
            'sub_index', 'truncated', 'occluded', 'difficult', 'is_datatang', 'shapenet']
    
    # get objects in the image
    tempAttr = getattr(data['record'], 'objects')
    objects_obj = np.asarray([tempAttr]).flatten()

    for ind in range(objects_obj.shape[0]):
        the_obj = objects_obj[ind]
        the_obj = vars(the_obj)
        objects = {}
        
        for field in attrs:
            if field == 'shapenet' or field == 'viewpoint':
                field_obj = the_obj[field]
                if isinstance(field_obj, np.asarray([]).__class__):
                    val = []
                else:
                    val = vars(field_obj)
                    del val['_fieldnames']
                    for k in val.keys():
                        if isinstance(val[k], np.asarray([]).__class__):
                            val[k] = []
                objects[field] = val
    
            elif field == 'bbox':
                objects[field] = the_obj[field].tolist()
            # handle T/F as 1/0 in matlab
            elif field == 'difficult' or field == 'is_datatang' or\
                 field == 'iscrowd' or field == 'occluded' or field == 'iscrowd':
                diff_int = the_obj[field]
                if diff_int == 0:
                    objects[field] = False
                else:
                    objects[field] = True
            else:
                field_val = []
                if field in the_obj.keys():
                    field_val = the_obj[field]
                    if isinstance(field_val, np.asarray([]).__class__):
                        field_val = []
                objects[field] = field_val
        if 'viewpoint' in objects:
            list_obj.append(objects)
    
    return list_obj


def get_im_info(data):
    img_annotation = {}
    attrs = ['filename', 'database']
    for attr in attrs:
        img_annotation[attr] = getattr(data['record'], attr)

    # get size attributes about the image
    size_obj = getattr(data['record'], 'size')
    sz = []
    for field in size_obj._fieldnames:
        sz.append(getattr(size_obj, field))
    img_annotation['image'] = {}
    img_annotation['image']['width'] = int(sz[0])
    img_annotation['image']['height'] = int(sz[1])
    img_annotation['image']['depth'] = int(sz[2])
    
    return img_annotation



if __name__ == "__main__":
    a = MakeAnns({})
    a.run_main()

