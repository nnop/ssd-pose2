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
This script preprocesses the annotations for scenes dataset
''' 


DATA_DIR = 'data/scene_one/'

class MakeAnns:

    def __init__(self, opt):
        self.opt = opt

    def run_main(self):

        base_path = DATA_DIR

        labelidx = readLabelToCls()

        if not osp.exists(osp.join(base_path, 'all_anns.json')):
            makeAllAnns(labelidx)

        sceneidx = readSceneToIdx()

        data = json.load(open(osp.join(base_path, 'all_anns.json'), 'r'))
        if not osp.exists(osp.join(base_path, 'map.txt')):
            createLabelMap(labelidx)

        train_dir = self.opt.get_gmu_db_stem('train')
        #val_dir = self.opt.get_gmu_db_stem('val')
        tes_dir = self.opt.get_gmu_db_stem('test')

        if not osp.exists(osp.join(base_path, 'cache/')):
            os.mkdir(osp.join(base_path, 'cache/'))

        #splits = [train_dir, val_dir, tes_dir]
        splits = [train_dir, tes_dir]
        for split in splits:
            if not osp.exists(osp.join(base_path, 'cache', split)):
                os.mkdir(osp.join(base_path, 'cache', split))

        tr = False
        #val = False
        test = False

        print train_dir

        scene_id = self.opt.get_opts('scene')
        data = splitData(data, sceneidx, scene_id)


        if not osp.exists(osp.join(base_path, 'cache', train_dir, 'train.txt')):
            tr = True
            trList = []

        #if not osp.exists(osp.join(base_path, 'cache', val_dir, 'val.txt')):
        #    vaList = []
         #   val = True

        if not osp.exists(osp.join(base_path, 'cache', tes_dir, 'test.txt')):
            teList = []
            test = True


        for idx, ann in data.iteritems():

            if ann['split'] == 'train':
                annpath = osp.join(base_path, 'cache', train_dir)
            elif ann['split'] == 'test':
                 annpath = osp.join(base_path, 'cache', tes_dir)
            #elif ann['split'] == 'val':
            #    annpath = osp.join(base_path, 'cache', val_dir)

            annLoc = osp.join(annpath, idx + '.json')
            output = getImPath(ann) + ' ' + annLoc + '\n'
            if not osp.exists(getImPath(ann)):
                continue

            json.dump(ann, open(annLoc, 'w'))

            if ann['split'] == 'train' and tr:
                trList.append(output)
            elif ann['split'] == 'test' and test:
                teList.append(output)
            #elif ann['split'] == 'val' and val:
            #    vaList.append(output)

        if tr:
            with open(osp.join(base_path, 'cache', train_dir, 'train.txt'), 'w') as outfile:
                shuffle(trList)
                for line in trList:
                    outfile.write(line)
        '''
        if val:
            with open(osp.join(base_path, 'cache', val_dir, 'val.txt'), 'w') as outfile:
                shuffle(vaList)
                for line in vaList:
                    outfile.write(line)
        '''
        if test:
            with open(osp.join(base_path, 'cache', tes_dir, 'test.txt'), 'w') as outfile:
                shuffle(teList)
                for line in teList:
                    outfile.write(line)


def createLabelMap(mapping):
    f = open(osp.join(DATA_DIR, 'map.txt'), 'w')
    for label, name in mapping.iteritems():
        f.write(name + ' ' + str(label) + ' ' + name + '\n')
    f.close()


def getImPath(ann):
    path = osp.join(DATA_DIR, 'images', ann['scene_name'], ann['filename'])
    return path


def getAnnPath(ann, train_dir, tes_dir, sceneidx):
    if sceneidx[ann['scene_name']] != args['test_scene_id']:
        path = osp.join(DATA_DIR, 'cache', train_dir)
    elif sceneidx[ann['scene_name']] == args['test_scene_id']:
        path = osp.join(DATA_DIR, 'cache', tes_dir)

    return path


def makeAllAnns(mapping):

    all_anns = {}
    ann_path = osp.join(DATA_DIR, 'annotations')
    scenes = os.listdir(ann_path)

    for scene_name in scenes:
        files = os.listdir(osp.join(ann_path, scene_name))
        for idx, fi in enumerate(files):
            matfile = sio.loadmat(osp.join(ann_path, scene_name, fi))
            #all_anns[fi[:-4]] = parseMat(matfile)
            temp = parseMat(matfile, mapping)
            temp['scene_name'] = scene_name
            temp['filename'] = fi[:-4] + '.jpg'
            im_id = fi[:-4]
            if im_id in all_anns:
                all_anns[im_id]['annotation'] += temp['annotation']
            else:
                all_anns[im_id] = temp

    print len(all_anns.keys())
    all_anns = removeEmptyAnns(all_anns)
    print len(all_anns.keys())
    makeSceneToIdx(all_anns)

    json.dump(all_anns, open(osp.join(DATA_DIR, 'all_anns.json'), 'w'))


def readLabelToCls():
    cls2lbl = {}
    with open(osp.join(DATA_DIR, 'lbl_cat.txt'), 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            print line
            cls_name, idx = line.split(' ')
            cls2lbl[int(idx)] = str(cls_name)
    return cls2lbl


def readSceneToIdx():
    sceneidx = {}
    with open(osp.join(DATA_DIR, 'sceneidx.txt'), 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            print line
            scene, idx = line.split(' ')
            sceneidx[scene] = int(idx)
    return sceneidx


def makeSceneToIdx(data):
    scenes = []
    for ann in data.itervalues():
        if ann['scene_name'] not in scenes:
            scenes.append(ann['scene_name'])

    with open(osp.join(DATA_DIR, 'sceneidx.txt'), 'w') as f:
        for idx, scene in enumerate(scenes):
            f.write('%s %d\n' % (scene, idx+1))


def removeEmptyAnns(data):
    keysToRemove = []

    for idx, ann in data.iteritems():
        if ann['annotation'] == []:
            keysToRemove.append(idx)

    for idx in keysToRemove:
        del data[idx]
    return data


def checkCounts(data):
    pascalTrData = {}

    for idx, obj in data.iteritems():
        if obj['database'] != 'ImageNet' and obj['split'] == 'test':
            pascalTrData[idx] = obj
    counts = {}
    for im in pascalTrData.itervalues():
        for obj in im['annotation']:
            cat = obj['category_id']
            if cat in counts:
                counts[cat] += 1
            else:
                counts[cat] = 1
    for key in sorted(counts):
        print "%s: %s" % (key, counts[key])


def splitData(data, sc_map, sc_test):

    for idx, ann in data.iteritems():
        if sc_map[ann['scene_name']] == sc_test:
            ann['split'] = 'test'
        else:
            ann['split'] = 'train'
    '''
    keys = data.keys()
    random.shuffle(keys)
    counter = 0
    for k in keys:
        if counter == 500: break
        if data[k]['split'] == 'train':
            data[k]['split'] = 'val'
            counter += 1
    '''
    return data



def parseMat(matfile, mapping):
    img_annotation = {}
    img_annotation['image'] = {}
    img_annotation['image']['width'] = int(600)
    img_annotation['image']['height'] = int(338)

    mat_ann = matfile['boxes']
    list_obj = []

    if mat_ann.size == 0:
        img_annotation['annotation'] = list_obj
        return img_annotation
    else:
        num_objects = mat_ann.shape[0]
        for ind in xrange(num_objects):
            objects = {}
            cat_id = mat_ann[ind, 4]

            # TODO
            objects['category_id'] = mapping[cat_id]
            
            bbox = mat_ann[ind, 0:4]

            scale_x = (600.0/1920.0)
            scale_y = (338.0/1080.0)

            bbox[0] = float(bbox[0]) * scale_x
            bbox[2] = float(bbox[2]) * scale_x

            bbox[1] = float(bbox[1]) * scale_y
            bbox[3] = float(bbox[3]) * scale_y

            # Convert to w,h TODO
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]


            bbox = bbox.tolist()
            objects['bbox'] = bbox
            list_obj.append(objects)

    img_annotation['annotation'] = list_obj
    return img_annotation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_scene_id', type=int, help='int id of the scene to use for testing')
    parser.add_argument('--num_bins', default='8', type=int, help='number of bins to divide angles into')
    parser.add_argument('--rotate', action='store_true', help='bin angles non standard way')

    args = parser.parse_args()
    params = vars(args) # turn into a dict
    main(params)
