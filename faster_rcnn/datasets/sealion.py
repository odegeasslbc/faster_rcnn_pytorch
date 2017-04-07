# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


#from .sealion import sealion
import os
from .imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class sealion(imdb):

    def __init__(self, image_set, devkit_path):
        imdb.__init__(self, image_set)

        self._image_set = image_set


        self._devkit_path = os.path.join(devkit_path)
        print 'devkit path: ' + self._devkit_path

	self._data_path = os.path.join(self._devkit_path, 'images')
        print 'image data path: ' + self._data_path

        self._classes = ('__background__', # always index 0
                         'adult_male', 'subadult_male', 'adult_female', 'juvenile', 'pups')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'

        self._image_index = self._load_image_set_index()

	#print self._image_index

        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

	#print self._devkit_path
        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
#'/home/bingchen/kaggle/test.txt'

        image_set_file = os.path.join(self._devkit_path, 'bbox.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip().split()[0] for x in f.readlines()]
        return image_index

#    def _get_default_path(self):
#        """
#        Return the default path where PASCAL VOC is expected to be installed.
#        """
#        return os.path.join(datasets.ROOT_DIR, 'data', 'VOCdevkit' + self._year)


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self._load_annotation()
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

	if self._image_set !='SealionTest':
		gt_roidb = self.gt_roidb()
		#ss_roidb = self._load_selective_search_roidb(gt_roidb)
		#roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
		roidb = gt_roidb	
	else:
		roidb = self._load_selective_search_roidb(None)
	with open(cache_file, 'wb') as fid:
		cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
	print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_annotation(self):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
	gt_roidb = []
	annotationfile = os.path.join(self._devkit_path, 'bbox.txt')
	f = open(annotationfile)
	split_line = f.readline().strip().split()
	num = 1

	while(split_line):
        	num_objs = int(split_line[1])
       		boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        	gt_classes = np.zeros((num_objs), dtype=np.int32)
        	overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        	for i in range(num_objs):
            		
			cls = int (split_line[2 + i * 5])
			
			x1 = float( split_line[3 + i * 5])
            		y1 = float (split_line[4 + i * 5])
            		x2 = float (split_line[5 + i * 5])
            		y2 = float (split_line[6 + i * 5])
			
            		boxes[i,:] = [x1, y1, x2, y2]
            		gt_classes[i] = cls
            		overlaps[i, cls] = 1.0

        	overlaps = scipy.sparse.csr_matrix(overlaps)
        	gt_roidb.append({'boxes' : boxes, 'gt_classes': gt_classes, 'gt_overlaps' : overlaps, 'flipped' : False})
        	split_line = f.readline().strip().split()

    	f.close()
    	return gt_roidb

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'sealion',
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        #cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.sealion('SeaLionTrain', '/home/bingchen/kaggle/sea-lion/Train')
    res = d.roidb
    from IPython import embed; embed()
