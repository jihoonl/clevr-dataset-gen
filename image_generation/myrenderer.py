# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function

import argparse
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime as dt
from pathlib import Path
from copy import deepcopy
"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
        import utils2
    except ImportError as e:
        print("\nERROR")
        print(
            "Running render_images.py from Blender and cannot import utils.py.")
        print(
            "You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print(
            "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth"
        )
        print(
            "\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument(
    '--base_scene_blendfile',
    default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
    "ground plane, lights, and camera.")
parser.add_argument(
    '--properties_json',
    default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
    "The \"colors\" field maps from CLEVR color names to RGB values; " +
    "The \"sizes\" field maps from CLEVR size names to scalars used to " +
    "rescale object models; the \"materials\" and \"shapes\" fields map " +
    "from CLEVR material and shape names to .blend files in the " +
    "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument(
    '--shape_dir',
    default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument(
    '--material_dir',
    default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument(
    '--shape_color_combos_json',
    default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
    "allowed color names for that shape. This allows rendering images " +
    "for CLEVR-CoGenT.")

# Settings for images
parser.add_argument(
    '--num_images',
    default=1,
    type=int,
    help="The maximum number of images to place in each scene")

# Settings for objects
parser.add_argument(
    '--min_objects',
    default=3,
    type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument(
    '--max_objects',
    default=3,
    type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument(
    '--min_dist',
    default=0.25,
    type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument(
    '--margin',
    default=0.4,
    type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
    "objects will be at least this distance apart. This makes resolving " +
    "spatial relationships slightly less ambiguous.")
parser.add_argument(
    '--min_pixels_per_object',
    default=20,
    type=int,
    help="All objects will have at least this many visible pixels in the " +
    "final rendered images; this ensures that no objects are fully " +
    "occluded by other objects.")
parser.add_argument(
    '--max_retries',
    default=50,
    type=int,
    help="The number of times to try placing an object before giving up and " +
    "re-placing all objects in the scene.")

# Output settings
parser.add_argument(
    '--start_idx',
    default=0,
    type=int,
    help="The index at which to start for numbering rendered images. Setting " +
    "this to non-zero values allows you to distribute rendering across " +
    "multiple machines and recombine the results later.")
parser.add_argument(
    '--num_scenes', default=1, type=int, help="The number of scenes to render")
parser.add_argument(
    '--filename_prefix',
    default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")

parser.add_argument(
    '--output_dir',
    default='../output/',
    help="The directory where outputs will be stored. It will be " +
    "created if it does not exist.")

parser.add_argument('--export_blend', default=True, type=bool)
parser.add_argument(
    '--version',
    default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument(
    '--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument(
    '--date',
    default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
    "defaults to today's date")

# Rendering options
parser.add_argument(
    '--use_gpu',
    default=0,
    type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
    "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
    "to work.")
parser.add_argument(
    '--width',
    default=240,
    type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument(
    '--height',
    default=240,
    type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument(
    '--key_light_jitter',
    default=1.0,
    type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument(
    '--fill_light_jitter',
    default=1.0,
    type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument(
    '--back_light_jitter',
    default=1.0,
    type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument(
    '--camera_jitter',
    default=0.5,
    type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument(
    '--render_num_samples',
    default=512,
    type=int,
    help="The number of samples to use when rendering. Larger values will " +
    "result in nicer images but will cause rendering to take longer.")
parser.add_argument(
    '--render_min_bounces',
    default=8,
    type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument(
    '--render_max_bounces',
    default=8,
    type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument(
    '--render_tile_size',
    default=256,
    type=int,
    help="The tile size to use for rendering. This should not affect the " +
    "quality of the rendered image but may affect the speed; CPU-based " +
    "rendering may achieve better performance using smaller tile sizes " +
    "while larger tile sizes may be optimal for GPU-based rendering.")


def main(args):
    prefix = '%s' % (args.filename_prefix)

    properties = utils2.load_property_file(args)
    for i in range(args.num_scenes):
        scene_root = Path('{}_{}'.format(prefix, str(i)))
        scene_root.mkdir(exist_ok=True)

        num_objects = random.randint(args.min_objects, args.max_objects)
        num_images = args.num_images
        render_scene(
            args,
            num_objects=num_objects,
            num_images=num_images,
            scene_root=scene_root,
            properties=properties)

    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    """
    all_scenes = []
    for scene_path in all_scene_paths:
        with open(str(scene_path), 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': args.split,
            'license': args.license,
        },
        'scenes': all_scenes
    }
    with open(args.output_scene_file, 'w') as f:
        json.dump(output, f)
    """


def rand(L):
    return 2.0 * L * (random.random() - 0.5)


def render_scene(args,
                 num_objects=1,
                 num_images=1,
                 scene_root='.',
                 properties=None):
    scene_path = 'scene.json'
    blender_path = 'scene.blend'

    # Prepare blender world
    render_args = prepare_world(args)

    # Add random jitter to lamp positions
    # add_lamp_jitter(args)
    directions = prepare_plane()

    success = False
    objects = []
    blender_objects = []
    while not success:
        success, objects, blender_objects = add_objects(args, num_objects,
                                                        directions, properties)
    scene_struct = {'objects': deepcopy(objects), 'directions': directions}
    for o in scene_struct['objects']:
        o['location'] = tuple(o['location'])
        o['bbox'] = [b.to_tuple() for b in o['bbox']]
    with open(str(scene_root / scene_path), 'w') as f:
        json.dump(scene_struct, f)

    if args.export_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(scene_root / blender_path))

    for i in range(num_images):
        imgname = '{}.png'.format(i)
        meta = '{}.json'.format(i)
        objs_export = deepcopy(objects)
        camera = bpy.data.objects['Camera']
        # for i in range(3):
        #    camera.location[i] += rand(args.camera_jitter)

        # Record data about the object in the scene data structure
        for o in objs_export:
            pixel_coords = utils.get_camera_coords(camera, o['location'])
            pixel_bbox = utils.get_pixel_bbox(camera, o['bbox'])
            o['location'] = tuple(o['location'])
            o['pixel_coords'] = pixel_coords
            o['pixel_bbox'] = pixel_bbox
            o['bbox'] = [b.to_tuple() for b in o['bbox']]
        filepath = str(scene_root / '{}'.format(imgname))
        render_single(render_args, filepath)

        for t in ['full', 'cube', 'sphere', 'cylinder']:
            print(t)
            to_omit = []
            for b, o in zip(blender_objects, objs_export):
                if o['shape'] == t:
                    to_omit.append(b)
            filepath = str(scene_root / '{}_{}'.format(t, imgname))

            mat = bpy.data.materials.new(name="MaterialName")
            mat.diffuse_color = (0, 0, 0)
            mat.diffuse_intensity = 0.0
            mat.ambient = 0.0

            for ob in bpy.data.objects:
                try:
                    ob.cycles_visibility.shadow = 0
                    ob.data.materials[0] = mat
                except:
                    print("No shadows: ", ob.name)
            bpy.data.worlds['World'].cycles.sample_as_light = False
            bpy.context.scene.cycles.blur_glossy = 10
            bpy.context.scene.cycles.transparent_min_bounces = 0
            bpy.context.scene.cycles.transparent_max_bounces = 0
            bpy.context.scene.update_tag()

            render_single(render_args, filepath, to_omit)

        with open(str(scene_root / meta), 'w') as f:
            json.dump(objs_export, f, indent=2)


def render_single(render_args, filepath, to_omit=[]):
    for b, c in to_omit:
        utils.delete_object(b)

    render_args.filepath = filepath
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)
    for b, c in to_omit:
        utils.add_object(*c)


def add_objects(args, num_objects, directions, properties):

    positions = []
    objects = []
    blender_objects = []

    for i in range(num_objects):
        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for (obj, c) in blender_objects:
                    utils.delete_object(obj)
                return False, objects

            data = add_single_object(positions, directions, *properties)
            if not data:
                continue

            # Actually add the object to the scene
            config = (args.shape_dir, data['obj'][0], data['size'][0],
                      data['coord'], data['rotation'], data['mat'][0],
                      data['color'][0])
            utils.add_object(*config)
            obj = bpy.context.object
            obj_boundbox, obj_cylinder = world_coordinate_bbox(
                obj, data['obj'][1])
            blender_objects.append((obj, config))
            positions.append((*data['coord'], data['size'][0]))

            # Add object meta info
            objects.append(
                dict(
                    shape=data['obj'][1],
                    size=data['size'][1],
                    material=data['mat'][1],
                    location=obj.location,
                    rotation=data['rotation'],
                    bbox=obj_boundbox,
                    cylinder=obj_cylinder,
                    color=data['color'][1]))
            break
    return True, objects, blender_objects


def world_coordinate_bbox(obj, obj_type, local=False):
    """
    from https://blender.stackexchange.com/questions/32283/what-are-all-values-in-bound-box
    """
    local_coords = obj.bound_box[:]
    location = obj.location
    print(location)
    om = obj.matrix_world

    worldify = lambda p: om * Vector(p[:])
    world_coords = [worldify(p) for p in local_coords]
    print(world_coords)
    vs = [Vector(p[:]) for p in local_coords]

    bbox_z = (vs[1] - vs[0]).z
    bbox_x = (vs[4] - vs[0]).x
    bbox_y = (vs[3] - vs[0]).y

    if obj_type in ['cylinder', 'sphere']:
        # Circle type
        radius = bbox_x / 2
        height = bbox_z
    else:
        # Cube type
        radius = math.sqrt(bbox_x**2 + bbox_y**2)
        height = bbox_z

    return world_coords, (radius, height)


def add_single_object(positions, directions, color_name_to_rgba,
                      material_mapping, object_mapping, size_mapping,
                      shape_color_combos):
    # Choose a random size
    size_name, scale = random.choice(size_mapping)

    x = random.uniform(-3, 3)
    y = random.uniform(-3, 3)
    # Check to make sure the new object is further than min_dist from all
    # other objects, and further than margin along the four cardinal directions
    dists_good = True
    margins_good = True
    for (xx, yy, ss) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - scale - ss < args.min_dist:
            dists_good = False
            break
        for direction_name in ['left', 'right', 'front', 'behind']:
            direction_vec = directions[direction_name]
            assert direction_vec[2] == 0
            margin = dx * direction_vec[0] + dy * direction_vec[1]
            if 0 < margin < args.margin:
                print(margin, args.margin, direction_name)
                print('BROKEN MARGIN!')
                margins_good = False
                break
        if not margins_good:
            break

    if not dists_good or not margins_good:
        return None

    # Choose random color and shape
    if shape_color_combos is None:
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
        obj_name_out, color_choices = random.choice(shape_color_combos)
        color_name = random.choice(color_choices)
        obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
        scale /= math.sqrt(2)
    theta = 360.0 * random.random()

    # Choose random orientation for the object.

    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)

    # index 0 for blender, index 1 for object meta info
    return dict(
        obj=(obj_name, obj_name_out),
        size=(scale, size_name),
        mat=(mat_name, mat_name_out),
        coord=(x, y),
        rotation=theta,
        color=(rgba, color_name))


def add_lamp_jitter(args):
    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(
                args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(
                args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(
                args.fill_light_jitter)


def prepare_plane():
    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    # Add random jitter to camera position
    # if args.camera_jitter > 0:
    #    for i in range(3):
    #       bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)
    d = {}
    d['behind'] = tuple(plane_behind)
    d['front'] = tuple(-plane_behind)
    d['left'] = tuple(plane_left)
    d['right'] = tuple(-plane_left)
    d['above'] = tuple(plane_up)
    d['below'] = tuple(-plane_up)
    return d


def prepare_world(args):
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)
    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = ''  # This will be overwritten later
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    render_args.layers[0].use_pass_vector = True

    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons[
                'cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'
    return render_args


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')
