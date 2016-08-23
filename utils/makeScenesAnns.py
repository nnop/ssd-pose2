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


DATA_DIR = 'data/scenes/'
LABEL_CLS = {1:'chair', 2:'diningtable', 3:'monitor', 4:'sofa'}

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

        train_dir = self.opt.get_scene_db_stem('train')
        #val_dir = self.opt.get_scene_db_stem('val')
        tes_dir = self.opt.get_scene_db_stem('test')

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

        if not osp.exists(osp.join(base_path, 'cache', train_dir, 'train.txt')):
            tr = True
            trList = []

        '''
        if not osp.exists(osp.join(base_path, 'cache', val_dir, 'val.txt')):
            vaList = []
            val = True
        '''

        if not osp.exists(osp.join(base_path, 'cache', tes_dir, 'test.txt')):
            teList = []
            test = True

        bins = self.opt.get_opts('num_bins')
        scene_id = self.opt.get_opts('scene')

        for idx, ann in data.iteritems():

            if sceneidx[ann['scene_name']] != scene_id:
                annpath = osp.join(base_path, 'cache', train_dir)
            elif sceneidx[ann['scene_name']] == scene_id:
                 annpath = osp.join(base_path, 'cache', tes_dir)

            annLoc = osp.join(annpath, idx + '.json')
            output = getImPath(ann) + ' ' + annLoc + '\n'
            binAngles(ann, bins, True)
            json.dump(ann, open(annLoc, 'w'))

            if sceneidx[ann['scene_name']] != scene_id and tr:
                trList.append(output)
            elif sceneidx[ann['scene_name']] == scene_id and test:
                teList.append(output)

        if tr:
            with open(osp.join(base_path, 'cache', train_dir, 'train.txt'), 'w') as outfile:
                shuffle(trList)
                for line in trList:
                    outfile.write(line)
        if test:
            with open(osp.join(base_path, 'cache', tes_dir, 'test.txt'), 'w') as outfile:
                shuffle(teList)
                for line in teList:
                    outfile.write(line)


'''
def main(args):

    if not osp.exists(osp.join(DATA_DIR, 'all_anns.json')):
        makeAllAnns()

    # get list of scene indexes
    
    data = json.load(open(osp.join(DATA_DIR, 'all_anns.json'), 'r'))
    if not osp.exists(osp.join(DATA_DIR, 'map.txt')):
        createLabelMap(data)
    
    # make a train/val/test directory
    #shutil.rmtree('./cache/')
    train_dir = 'train_bins=%d_rotate=%r_excl=%d' % (args['num_bins'], args['rotate'], args['test_scene_id'])
    val_dir = 'val_bins=%d_rotate=%r' % (args['num_bins'], args['rotate'])
    tes_dir = 'test_bins=%d_rotate=%r_scene=%d' % (args['num_bins'], args['rotate'], args['test_scene_id'])
    
    if not osp.exists(osp.join(DATA_DIR, 'cache')):
        os.mkdir(osp.join(DATA_DIR, 'cache'))
    
    splits = [train_dir, val_dir, tes_dir]
    for split in splits:
        if not osp.exists(osp.join(DATA_DIR, 'cache', split)):
            os.mkdir(osp.join(DATA_DIR, 'cache', split))

    tr = False
    val = False
    test = False

    print train_dir

    if not osp.exists(osp.join(DATA_DIR, 'cache', train_dir, 'train.txt')):
        #trWriter = open(osp.join('./cache', train_dir, 'train.txt'), 'w')
        tr = True
        trList = []
        #if not args['difficult']:
        #    filterDifficult(data)

    if not osp.exists(osp.join(DATA_DIR, 'cache', val_dir, 'val.txt')):
        #valWriter = open(osp.join('./cache', val_dir, 'val.txt'), 'w')
        vaList = []
        val = True
    
    if not osp.exists(osp.join(DATA_DIR, 'cache', tes_dir, 'test.txt')):
        #teWriter = open(osp.join('./cache', tes_dir, 'test.txt'), 'w')
        teList = []
        test = True

    #cur_dir = os.getcwd()
    val = False

    print args['test_scene_id']

    for idx, ann in data.iteritems():
        
        if sceneidx[ann['scene_name']] != args['test_scene_id']:
            #print sceneidx[ann['scene_name']]
            annpath = osp.join(DATA_DIR, 'cache', train_dir)
        elif sceneidx[ann['scene_name']] == args['test_scene_id']:
            annpath = osp.join(DATA_DIR, 'cache', tes_dir)
        
        annLoc = osp.join(annpath, idx + '.json')
        output = getImPath(ann) + ' ' + annLoc + '\n'
        binAngles(ann, args['num_bins'], args['rotate'])
        json.dump(ann, open(annLoc, 'w'))

        if sceneidx[ann['scene_name']] != args['test_scene_id'] and tr:
            trList.append(output)
                
        #elif ann['split'] == 'val' and val:
            # use val file writer
         #   vaList.append(output)
            #valWriter.write(output)
        elif sceneidx[ann['scene_name']] == args['test_scene_id'] and test:
            # use test file writer
            teList.append(output)
            #teWriter.write(output)
    
    if tr:
        with open(osp.join(DATA_DIR, 'cache', train_dir, 'train.txt'), 'w') as outfile:
            shuffle(trList)
            for line in trList:
                outfile.write(line)
        #trWriter.close()
    if val:
        with open(osp.join(DATA_DIR, 'cache', val_dir, 'val.txt'), 'w') as outfile:
            shuffle(vaList)
            for line in vaList:
                outfile.write(line)
        #valWriter.close()
    if test:
        with open(osp.join(DATA_DIR, 'cache', tes_dir, 'test.txt'), 'w') as outfile:
            shuffle(teList)
            for line in teList:
                outfile.write(line)
        #teWriter.close()
'''


def createLabelMap(data):
    #objClasses = set([obj['category_id'] for ann in data.itervalues() for obj in ann['annotation'] if 'viewpoint' in obj and obj['category_id'] != 'bottle' ])
    #labelToCls = dict((idx+1, cls) for idx, cls in enumerate(objClasses))
    labelToCls = {1:'chair', 2:'diningtable', 3:'monitor', 4:'sofa'}
    f = open(osp.join(DATA_DIR, 'map.txt'), 'w')
    for label, name in labelToCls.iteritems():
        f.write(name + ' ' + str(label) + ' ' + name + '\n')
    f.close()


def getImPath(ann):
    path = osp.join(DATA_DIR, 'images', ann['scene_name'], ann['filename'])
    #if ann['database'] == 'ImageNet':
    #    path = osp.join('images/imagenet/', ann['filename'])
    #else:
    #    path = osp.join('images/pascal/', ann['filename'])
    return path

def getAnnPath(ann, train_dir, tes_dir, sceneidx):
    if sceneidx[ann['scene_name']] != args['test_scene_id']:
        path = osp.join(DATA_DIR, 'cache', train_dir)
    elif sceneidx[ann['scene_name']] == args['test_scene_id']:
        path = osp.join(DATA_DIR, 'cache', tes_dir)

    return path



def binAngles(ann, bins, rotate=False):
    offset = 0
    if rotate:
        offset = 360 / (bins * 2)
    for obj in ann['annotation']:
        if 'viewpoint' in obj:
            
            azi = obj['viewpoint']['azimuth']

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

    all_anns = {}
    #myFiles = []
    ann_path = osp.join(DATA_DIR, 'Annotations')
    scenes = os.listdir(ann_path)

    for scene_name in scenes:
    #ann = 'bottle_pascal'
    #if True:
        #if 'pascal' in ann:
        #print ann
        files = os.listdir(osp.join(ann_path, scene_name))
        for idx, fi in enumerate(files):
            #if idx % 100 == 0: print 'processing file %d in %s' %(idx, ann)
            matfile = sio.loadmat(osp.join(ann_path, scene_name, fi))
            #all_anns[fi[:-4]] = parseMat(matfile)
            temp = parseMat(matfile)
            temp['scene_name'] = scene_name
            temp['filename'] = fi[:-4] + '.png'
            im_id = temp['scene_name'] + fi[:-4]
            if im_id in all_anns:
                all_anns[im_id]['annotation'] += temp['annotation']
            else:
                all_anns[im_id] = temp

                #myFiles.append(fi[:-4])
    #print "Total: " + str(len(myFiles))

    #all_anns = removeBottle(all_anns)

    #checkCounts(all_anns)
    # checkDB(all_anns)
    # checkTrain(myFiles)
    #checkCounts(all_anns)
    #all_anns = splitData(all_anns)
    #checkCounts(all_anns)
    #convertBbox(all_anns)

    # hack hard coded 
    #del all_anns['n03790512_11192']
    #del all_anns['n02690373_190'] 
    print len(all_anns.keys())
    all_anns = removeEmptyAnns(all_anns)
    print len(all_anns.keys())
    makeSceneToIdx(all_anns)

    #for i in xrange(len(all_anns.keys())):
    json.dump(all_anns, open(osp.join(DATA_DIR, 'all_anns.json'), 'w'))
    #    print i
    #    print 'json dumped'

# def checkDB(data):
#     dbs = []
#     for obj in data.itervalues():
#         dbs.append(obj['database'])
#     set(dbs)
#     print dbs
#     sys.exit()


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
    f = open('./PASCAL/VOCdevkit/VOC2012/ImageSets/Main/train.txt', 'r')
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

    f = open('./PASCAL/VOCdevkit/VOC2012/ImageSets/Main/val.txt', 'r')
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
    img_annotation['image'] = {}
    img_annotation['image']['width'] = int(600)
    img_annotation['image']['height'] = int(338)

    mat_ann = matfile['objects']
    list_obj = []

    if mat_ann.size == 0:
        img_annotation['annotation'] = list_obj
        return img_annotation
    else:
        num_objects = mat_ann.shape[0]
        for ind in xrange(num_objects):
            objects = {}
            cat_id = mat_ann[ind, 0]
            objects['category_id'] = LABEL_CLS[cat_id]
            pose = mat_ann[ind, 5]
            objects['viewpoint'] = {}
            objects['viewpoint']['azimuth'] = float(pose)
            
            bbox = mat_ann[ind, 1:5]
            bbox[0] /= 1920.0
            bbox[0] *= 600.0
            bbox[2] /= 1920.0
            bbox[2] *= 600.0

            bbox[1] /= 1080.0
            bbox[1] *= 338.0
            bbox[3] /= 1080.0
            bbox[3] *= 338.0

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
