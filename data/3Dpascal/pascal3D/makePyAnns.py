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


def main(args):

    if not osp.exists('all_anns.json'):
        makeAllAnns()

    # something with args
    data = json.load(open('all_anns.json', 'r'))
    if not osp.exists('map.txt'):
        createLabelMap(data)
    
    # make a train/val/test directory
    #shutil.rmtree('./cache/')
    train_dir = 'train_bins=%d_diff=%r_imgnet=%r_numPascal=%d' % (args['num_bins'], args['difficult'], args['imagenet'], args['num_pascal'])
    val_dir = 'val_bins=%d_diff=%r' % (args['num_bins'], args['difficult'])
    tes_dir = 'test_bins=%d_diff=%r' % (args['num_bins'], args['difficult'])
    
    if not osp.exists('./cache/'):
        os.mkdir('./cache/')
    
    splits = [train_dir, val_dir, tes_dir]
    for split in splits:
        if not osp.exists(osp.join('./cache', split)):
            os.mkdir(osp.join('./cache', split))

    tr = False
    val = False
    test = False

    print train_dir

    # always filter difficult ?
    if not args['difficult']:
        filterDifficult(data)

    if not osp.exists(osp.join('./cache', train_dir, 'train.txt')):
        #trWriter = open(osp.join('./cache', train_dir, 'train.txt'), 'w')
        tr = True
        trList = []
        #if not args['difficult']:
        #    filterDifficult(data)

    if not osp.exists(osp.join('./cache', val_dir, 'val.txt')):
        #valWriter = open(osp.join('./cache', val_dir, 'val.txt'), 'w')
        vaList = []
        val = True
    
    if not osp.exists(osp.join('./cache', tes_dir, 'test.txt')):
        #teWriter = open(osp.join('./cache', tes_dir, 'test.txt'), 'w')
        teList = []
        test = True

    #cur_dir = os.getcwd()

    for idx, ann in data.iteritems():
        annLoc = osp.join(getAnnPath(ann, train_dir, val_dir, tes_dir), idx + '.json')
        output = getImPath(ann) + ' ' + annLoc + '\n'
        binAngles(ann, args['num_bins'])
        json.dump(ann, open(annLoc, 'w'))

        if ann['split'] == 'train' and tr:
            # use train file writer
            if ann['database'] == 'ImageNet':
                #print 'stop it '
                if args['imagenet']:
                    trList.append(output)
                    #trWriter.write(output)
                #yo = 'do nothing '
            else:
                for _ in xrange(args['num_pascal']):
                    trList.append(output)
                    #trWriter.write(output)
            
        elif ann['split'] == 'val' and val:
            # use val file writer
            vaList.append(output)
            #valWriter.write(output)
        elif ann['split'] == 'test' and test:
            # use test file writer
            teList.append(output)
            #teWriter.write(output)
    
    if tr:
        with open(osp.join('./cache', train_dir, 'train.txt'), 'w') as outfile:
            shuffle(trList)
            for line in trList:
                outfile.write(line)
        #trWriter.close()
    if val:
        with open(osp.join('./cache', val_dir, 'val.txt'), 'w') as outfile:
            shuffle(vaList)
            for line in vaList:
                outfile.write(line)
        #valWriter.close()
    if test:
        with open(osp.join('./cache', tes_dir, 'test.txt'), 'w') as outfile:
            shuffle(teList)
            for line in teList:
                outfile.write(line)
        #teWriter.close()


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
    f = open('map.txt', 'w')
    for label, name in labelToCls.iteritems():
        f.write(name + ' ' + str(label) + ' ' + name + '\n')
    f.close()


def getImPath(ann):
    if ann['database'] == 'ImageNet':
        path = osp.join('images/imagenet/', ann['filename'])
    else:
        path = osp.join('images/pascal/', ann['filename'])
    return path

def getAnnPath(ann, train_dir, val_dir, tes_dir):
    if ann['split'] == 'train':
        path = osp.join('cache', train_dir)
    elif ann['split'] == 'val':
        path = osp.join('cache', val_dir)
    elif ann['split'] == 'test':
        path = osp.join('cache', tes_dir)

    return path



def binAngles(ann, bins):
    for obj in ann['annotation']:
        if 'viewpoint' in obj:
            azi = obj['viewpoint']['azimuth_coarse']
            
            if 'azimuth' in obj['viewpoint']:
                azi = obj['viewpoint']['azimuth']

            bin = int(azi / (360/bins))
            obj['aziLabel'] = bin
            flipAzi = 360 - azi
            obj['aziLabelFlip'] = int(flipAzi / (360/bins))
        else:
            print 'woah where yo angle'



def makeAllAnns():
    #matfile = sio.loadmat('./Annotations/chair_pascal/2008_007827.mat', squeeze_me=True, struct_as_record=False)
    #parseMat(matfile, 'fuck you')

    all_anns = {}
    #myFiles = []
    anns = os.listdir('./Annotations/')
    for ann in anns:
    #ann = 'bottle_pascal'
    #if True:
        #if 'pascal' in ann:
        #print ann
        files = os.listdir(osp.join('./Annotations', ann))
        for idx, fi in enumerate(files):
            #if idx % 100 == 0: print 'processing file %d in %s' %(idx, ann)
            matfile = sio.loadmat(osp.join('./Annotations', ann, fi), squeeze_me=True, struct_as_record=False)
            #all_anns[fi[:-4]] = parseMat(matfile)
            temp = parseMat(matfile, osp.join('./Annotations', ann, fi))
            if fi[:-4] in all_anns:
                all_anns[fi[:-4]]['annotation'] += temp['annotation']
            else:
                all_anns[fi[:-4]] = temp

                #myFiles.append(fi[:-4])
    #print "Total: " + str(len(myFiles))

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

    json.dump(all_anns, open('all_anns.json', 'w'))
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



def parseMat(matfile, finame):
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
                    #print finame
                    # if view_obj.shape[0] == 0:
                        #print 'skipping'
                        # temp = getattr(objects_obj[2], 'viewpoint')
                        # print "temp is: " + str(temp)
                        # views = {}
                        # for view_field in temp._fieldnames:
                        #     views[view_field] = getattr(temp, view_field)
                        # print views

                        # if isinstance(temp, np.asarray([]).__class__):
                        #     if temp.shape[0] == 0:
                        #         print "fail"
                        # sys.exit()
                        # continue
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
                #if cat_id == 'bottle': 
                #    print cat_id
                #    print 'breaking'
                #    break
                objects['category_id'] = cat_id
        if 'viewpoint' in objects:
            objects['iscrowd'] = 0
            list_obj.append(objects)
    #print list_obj
    #print len(list_obj)
    #sys.exit()
    if list_obj == []:
        print finame
        print 'whoops'
    img_annotation['annotation'] = list_obj
    return img_annotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_bins', default='8', type=int, help='number of bins to divide angles into')
    parser.add_argument('--num_pascal', default='1', type=int, help='number of times to include pascal ims')
    parser.add_argument('--difficult', action='store_false', help='filter difficult images from the dataset')
    parser.add_argument('--imagenet', action='store_false', help='exclude imagenet images')

    args = parser.parse_args()
    params = vars(args) # turn into a dict
    main(params)
