#!/bin/sh

NUM_SEQ=${1:-1}
NUM_OBJ=${2:-5}
NUM_SAMPLE=${3:-50}

/data/private/work/blender/blender --background --python myrenderer.py -- --sequence_length ${NUM_SEQ} --render_num_samples ${NUM_SAMPLE} --min_objects ${NUM_OBJ} --max_objects ${NUM_OBJ} --use_gpu 1
