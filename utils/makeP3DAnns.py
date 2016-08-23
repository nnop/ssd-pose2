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
This script preprocesses the annotations 
''' 

class MakeAnns:

    def __init__(self, opt):
        self.opt = opt

    def run_main(self):

        base_path = 'data/pascal3D/'

        if not osp.exists(osp.join(base_path, 'all_anns.json')):
            makeAllAnns()

        data = json.load(open('data/pascal3D/all_anns.json', 'r'))
        if not osp.exists('data/pascal3D/map.txt'):
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
    f = open('data/pascal3D/map.txt', 'w')
    for label, name in labelToCls.iteritems():
        f.write(name + ' ' + str(label) + ' ' + name + '\n')
    f.close()


def getImPath(ann):
    if ann['database'] == 'ImageNet':
        path = osp.join('data/pascal3D/images/imagenet/', ann['filename'])
    else:
        path = osp.join('data/pascal3D/images/pascal/', ann['filename'])
    return path

def getAnnPath(ann, train_dir, val_dir, tes_dir):
    if ann['split'] == 'train':
        path = osp.join('data/pascal3D/cache', train_dir)
    elif ann['split'] == 'val':
        path = osp.join('data/pascal3D/cache', val_dir)
    elif ann['split'] == 'test':
        path = osp.join('data/pascal3D/cache', tes_dir)

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

            #if obj['viewpoint']['distance'] == 0:

            # bin = int(azi / (360/bins))
            bin = int(((azi + offset) % 360) / (360/bins))
            #print "Azi: %f  bin: %d" %(azi, bin) 
            obj['aziLabel'] = bin
            flipAzi = 360 - azi
            # obj['aziLabelFlip'] = int(flipAzi / (360/bins))
            obj['aziLabelFlip'] = int(((flipAzi + offset) % 360) / (360/bins))
        else:
            print 'woah where yo angle'



def makeAllAnns():
    base_path = 'data/pascal3D'

    all_anns = {}
    #myFiles = []
    anns = os.listdir(osp.join(base_path, 'Annotations'))
    for ann in anns:
        files = os.listdir(osp.join(base_path, 'Annotations', ann))
        for idx, fi in enumerate(files):
            #if idx % 100 == 0: print 'processing file %d in %s' %(idx, ann)
            matfile = sio.loadmat(osp.join(base_path, 'Annotations', ann, fi), squeeze_me=True, struct_as_record=False)
            #all_anns[fi[:-4]] = parseMat(matfile)
            temp = parseMat(matfile)
            if fi[:-4] in all_anns:
                all_anns[fi[:-4]]['annotation'] += temp['annotation']
            else:
                all_anns[fi[:-4]] = temp

    all_anns = removeBottle(all_anns)

    #checkCounts(all_anns)
    # checkDB(all_anns)
    # checkTrain(myFiles)
    #checkCounts(all_anns)
    all_anns = splitData(all_anns)
    checkCounts(all_anns)
    convertBbox(all_anns)

    # hack hard coded 
    del all_anns['n03790512_11192']
    del all_anns['n02690373_190'] 

    json.dump(all_anns, open('data/pascal3D/all_anns.json', 'w'))
    print 'json dumped'

# def checkDB(data):
#     dbs = []
#     for obj in data.itervalues():
#         dbs.append(obj['database'])
#     set(dbs)
#     print dbs
#     sys.exit()



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

def removeBottle(data):
    keysToRemove = []
    count = 0
    for key, ann in data.iteritems():
        newList = []
        for obj in ann['annotation']: 
            if obj['category_id'] != 'bottle':
                newList.append(obj)
            else:
                count += 1
        if len(newList) == 0: keysToRemove.append(key)
        ann['annotation'] = newList

    for rem in keysToRemove:
        del data[rem]
    return data

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
    f = open('data/pascal3D/PASCAL/VOCdevkit/VOC2012/ImageSets/Main/train.txt', 'r')
    train = [line.strip('\n') for line in f ]
    print 'training data %d' % len(train)
    count = 0
    for tr in train:
        if tr in data:
            count += 1
            data[tr]['split'] = 'train'
        #else:
            #print tr
    print 'actual training %d' % count

    f = open('data/pascal3D/PASCAL/VOCdevkit/VOC2012/ImageSets/Main/val.txt', 'r')
    val = [line.strip('\n') for line in f ]
    i = 0
    print 'testing data %d' % len(val)
    count = 0
    for tr in val:
        if tr in data:
            data[tr]['split'] = 'test'
            count += 1
    print 'actual test %d' % count

    #remKeys = []

    for key, obj in data.iteritems():
        if obj['database'] == 'ImageNet':
            obj['split'] = 'train'
            #remKeys.append(key)

    #for key in remKeys:
    #    del data[key]

    keys = data.keys()
    random.shuffle(keys)
    counter = 0
    for k in keys:
        if counter == 500: break
        if data[k]['split'] == 'train' and data[k]['database'] != 'ImageNet':
            data[k]['split'] = 'val'
            counter += 1
    return data



def parseMat(matfile):
    img_annotation = {}
    attrs = ['filename', 'database']
    for attr in attrs:
        img_annotation[attr] = getattr(matfile['record'], attr)

    # get size attributes about the image
    size_obj = getattr(matfile['record'], 'size')
    sz = []
    for field in size_obj._fieldnames:
        sz.append(getattr(size_obj, field))
    img_annotation['image'] = {}
    img_annotation['image']['width'] = int(sz[0])
    img_annotation['image']['height'] = int(sz[1])

    list_obj = []
    attrs = ['bbox', 'class', 'viewpoint', 'difficult']
    
    # get objects in the image
    tempAttr = getattr(matfile['record'], 'objects')
    objects_obj = np.asarray([tempAttr]).flatten()

    for ind in range(objects_obj.shape[0]):
        the_obj = objects_obj[ind]
        objects = {}
        #print 'broke to a'
        for field in attrs:
            #print 'broke to b'
            if field == 'viewpoint':    
                view_obj = getattr(the_obj, 'viewpoint')
                if isinstance(view_obj, np.asarray([]).__class__):
                    continue
                views = {}
                #print view_obj._fieldnames
                for view_field in view_obj._fieldnames:
                    views[view_field] = getattr(view_obj, view_field)
                objects[field] = views
            elif field == 'bbox':
                objects[field] = getattr(the_obj, field).tolist()
            elif field == 'difficult':
                diff_int = getattr(the_obj, field)
                if diff_int == 0:
                    objects[field] = False
                else:
                    objects[field] = True
            else:
                cat_id = getattr(the_obj, field)
                objects['category_id'] = cat_id
        if 'viewpoint' in objects:
            objects['iscrowd'] = 0
            list_obj.append(objects)

    if list_obj == []:
        print 'whoops'
    img_annotation['annotation'] = list_obj
    return img_annotation

