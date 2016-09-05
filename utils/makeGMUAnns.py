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


DATA_DIR = 'data/gmu_kitchen/'
ALL_SCENES = ['gmu_scene_00%d' % (idx+1) for idx in range(9)]

class MakeAnns:

    def __init__(self, opt):
        self.opt = opt

    def run_main(self):

        base_path = DATA_DIR

        if not osp.exists(osp.join(base_path, 'all_anns.json')):
            makeAllAnns()

        sceneidx = readSceneToIdx()

        data = json.load(open(osp.join(base_path, 'all_anns.json'), 'r'))
        if not osp.exists(osp.join(base_path, 'map.txt')):
            createLabelMap(data)

        train_dir = self.opt.get_gmu_db_stem('train')
        val_dir = self.opt.get_gmu_db_stem('val')
        tes_dir = self.opt.get_gmu_db_stem('test')

        if not osp.exists(osp.join(base_path, 'cache/')):
            os.mkdir(osp.join(base_path, 'cache/'))

        splits = [train_dir, val_dir, tes_dir]
        #splits = [train_dir, tes_dir]
        for split in splits:
            if not osp.exists(osp.join(base_path, 'cache', split)):
                os.mkdir(osp.join(base_path, 'cache', split))

        tr = False
        val = False
        test = False

        print train_dir

        # split data
        scene_id = self.opt.get_opts('scene')
        data = splitData(data, sceneidx, scene_id)


        if not osp.exists(osp.join(base_path, 'cache', train_dir, 'train.txt')):
            tr = True
            trList = []

        if not osp.exists(osp.join(base_path, 'cache', val_dir, 'val.txt')):
            vaList = []
            val = True

        if not osp.exists(osp.join(base_path, 'cache', tes_dir, 'test.txt')):
            teList = []
            test = True


        for idx, ann in data.iteritems():

            if ann['split'] == 'train':
                annpath = osp.join(base_path, 'cache', train_dir)
            elif ann['split'] == 'test':
                 annpath = osp.join(base_path, 'cache', tes_dir)
            elif ann['split'] == 'val':
                annpath = osp.join(base_path, 'cache', val_dir)

            annLoc = osp.join(annpath, idx + '.json')
            output = getImPath(ann) + ' ' + annLoc + '\n'
            if not osp.exists(getImPath(ann)):
                continue

            json.dump(ann, open(annLoc, 'w'))

            if ann['split'] == 'train' and tr:
                trList.append(output)
            elif ann['split'] == 'test' and test:
                teList.append(output)
            elif ann['split'] == 'val' and val:
                vaList.append(output)

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



def createLabelMap(data):
    objClasses = set([(obj['cls_label'], obj['category_id']) for ann in data.itervalues() for obj in ann['annotation'] ])
    labelToCls = dict((idx, cls_id) for idx, cls_id in objClasses)
    f = open(osp.join(DATA_DIR, 'map.txt'), 'w')
    for label, name in labelToCls.iteritems():
        f.write(name + ' ' + str(label) + ' ' + name + '\n')
    f.close()


def getImPath(ann):
    path = osp.join(DATA_DIR, 'images', ann['scene_name'], 'Images', ann['filename'])
    return path


def getAnnPath(ann, train_dir, tes_dir, sceneidx):
    if sceneidx[ann['scene_name']] != args['test_scene_id']:
        path = osp.join(DATA_DIR, 'cache', train_dir)
    elif sceneidx[ann['scene_name']] == args['test_scene_id']:
        path = osp.join(DATA_DIR, 'cache', tes_dir)

    return path


def makeAllAnns():

    all_anns = {}
    ann_path = osp.join(DATA_DIR, 'scene_annotation/bboxes/')
    scenes = os.listdir(ann_path)

    # TODO fix loop
    for scene_name in ALL_SCENES:
        matfname = osp.join(ann_path, scene_name + '_annotated_bboxes.mat')

        data = sio.loadmat(open(matfname, 'r'))
        data['bboxes'] = data['bboxes'].squeeze()

        all_anns = parseMat(data, scene_name, all_anns)

    print len(all_anns.keys())
    all_anns = removeEmptyAnns(all_anns)
    print len(all_anns.keys())
    makeSceneToIdx(all_anns)

    json.dump(all_anns, open(osp.join(DATA_DIR, 'all_anns.json'), 'w'))


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
    with open(osp.join(DATA_DIR, 'sceneidx.txt'), 'w') as f:
        for idx, scene in enumerate(ALL_SCENES):
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


def splitData(data, sc_map, sc_test):

    for idx, ann in data.iteritems():
        if sc_map[ann['scene_name']] == sc_test:
            ann['split'] = 'test'
        else:
            ann['split'] = 'train'

    keys = data.keys()
    random.shuffle(keys)
    counter = 0
    for k in keys:
        if counter == 500: break
        if data[k]['split'] == 'train':
            data[k]['split'] = 'val'
            counter += 1
    return data


def parseMat(data, scene_name, out):

    count = 0
    for im_id in xrange(len(data['bboxes'])):
        temp = {}

        temp['image'] = {}
        temp['image']['width'] = int(1920)
        temp['image']['height'] = int(1080)

        try:
            objs = data['bboxes'][im_id].squeeze()[:]
        except:
            continue
            count += 1

        if len(objs) ==0:
            count += 1
            continue

        im_name = str(objs['imgName'][0].squeeze())
        temp['filename'] = im_name
        temp['scene_name'] = scene_name

        obj_list = []

        for obj_id in range(len(objs)):
            obj_ann = {}

            cls_id = str(objs['category'][obj_id].squeeze())
            # HARD coded HACK
            #cls_id[3:-2]
            obj_ann['category_id'] = cls_id[3:-2]

            label = int(objs['label'][obj_id].squeeze())
            obj_ann['cls_label'] = label

            y1 = float(objs['top'][obj_id].squeeze())
            x1 = float(objs['left'][obj_id].squeeze())
            y2 = float(objs['bottom'][obj_id].squeeze())
            x2 = float(objs['right'][obj_id].squeeze())
            bbox = [x1, y1, x2-x1, y2-y1]
            obj_ann['bbox'] = bbox

            obj_list.append(obj_ann)

        if obj_list != []:
            temp['annotation'] = obj_list
            key = scene_name + '_' + im_name[:-4]
            out[key] = temp

    print count
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_dir', type=str, help='int id of the scene to use for testing')

    args = parser.parse_args()
    params = vars(args) # turn into a dict

    myAnns = MakeAnns(params['opt_dir'])
    myAnns.run_main()
