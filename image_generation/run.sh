#!/bin/sh

NUM_SCENES=${1:-1}
NUM_SEQ=${2:-1}
NUM_OBJ=${3:-5}
NUM_SAMPLE=${4:-512}

/data/private/work/blender/blender --background --python myrenderer.py -- --sequence_length ${NUM_SEQ} --render_num_samples ${NUM_SAMPLE} --min_objects ${NUM_OBJ} --max_objects ${NUM_OBJ} --use_gpu 1 --num_scenes ${NUM_SCENES} --export_blend true
