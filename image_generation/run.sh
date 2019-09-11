#!/bin/sh

NUM_OBJ=${1:-5}
NUM_IMG=${2:-1}
NUM_SAMPLE=${3:-50}

/data/private/work/blender/blender --background --python myrenderer.py -- --num_images ${NUM_IMG} --render_num_samples ${NUM_SAMPLE} --min_objects ${NUM_OBJ} --max_objects ${NUM_OBJ} --use_gpu 1
