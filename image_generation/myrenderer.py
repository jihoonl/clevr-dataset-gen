# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Modified by Jihoon Lee, KakaoBrain

from __future__ import print_function

import argparse
import json
import math
import operator
import random
import os
import sys
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
import pickle
import tempfile

import numpy as np
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
parser.add_argument('--sequence_length',
                    default=1,
                    type=int,
                    help="The maximum number of images to place in each scene")
parser.add_argument('--topview_z',
                    default=15.0,
                    type=float,
                    help="The height of topview camera")
# Settings for objects
parser.add_argument('--min_objects',
                    default=3,
                    type=int,
                    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects',
                    default=3,
                    type=int,
                    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist',
                    default=0.1,
                    type=float,
                    help="The minimum allowed distance between object centers")
parser.add_argument(
    '--margin',
    default=0.1,
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
parser.add_argument('--num_scenes',
                    default=1,
                    type=int,
                    help="The number of scenes to render")

parser.add_argument(
    '--output_dir',
    default='output',
    help="The directory where outputs will be stored. It will be " +
    "created if it does not exist.")
parser.add_argument('--num_scenes_per_dir',
                    default=1000,
                    type=int,
                    help='Number of scenes to keep in the same directory')

parser.add_argument('--export_blend', default=False, type=bool)
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
parser.add_argument('--width',
                    default=128,
                    type=int,
                    help="The width (in pixels) for the rendered images")
parser.add_argument('--height',
                    default=128,
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
parser.add_argument('--render_min_bounces',
                    default=8,
                    type=int,
                    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces',
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    properties = utils2.load_property_file(args)
    dir_idx = 0
    scene_root = output_dir / str(dir_idx)
    scene_root.mkdir(exist_ok=True)
    for i in range(args.num_scenes):
        scene_idx = i % args.num_scenes_per_dir
        if scene_idx == 0:
            scene_root = output_dir / str(dir_idx)
            scene_root.mkdir(exist_ok=True)
            dir_idx += 1
        num_objects = random.randint(args.min_objects, args.max_objects)
        sequence_length = args.sequence_length
        render_scene(args,
                     num_objects=num_objects,
                     sequence_length=sequence_length,
                     scene_root=scene_root,
                     scene_idx=scene_idx,
                     properties=properties)


def rand(L):
    return 2.0 * L * (random.random() - 0.5)


def render_scene(args,
                 num_objects=1,
                 sequence_length=1,
                 scene_root='.',
                 scene_idx=0,
                 properties=None):
    scene_path = scene_root / '{}.pkl'.format(scene_idx)

    # Prepare blender world

    ################
    # Generate Scene
    ################
    render_args = prepare_world(args)

    # Add random jitter to lamp positions
    add_lamp_jitter(args)
    directions = prepare_plane()

    success = False
    objects = []
    blender_objects = []
    while not success:
        success, objects, blender_objects = add_objects(args, num_objects,
                                                        directions, properties)
    world = {
        'objects': deepcopy(objects),
    }
    num_objs = len(objects)
    colormap = np.linspace(0, 1, num_objs + 2)[1:-1]  # To skip 0, and 255
    colormap_export = np.vectorize(convert_to_srgb)(
        colormap)  # subtracting background color 64
    colormap_export = (colormap_export * 255).round().astype(np.uint8) - 64

    # Add index and assign colormask value
    for i, o in enumerate(world['objects']):
        o['index'] = i
        o['mask_color_render'] = colormap[i]
        o['mask_color'] = colormap_export[i]

    if args.export_blend:
        blender_path = scene_root / '{}.blend'.format(scene_idx)
        bpy.ops.wm.save_as_mainfile(filepath=str(blender_path))

    ########################
    # Generate top camera views
    ########################
    camera = bpy.data.objects['Camera']
    orig_cam = dict(x=camera.location.x,
                    y=camera.location.y,
                    z=camera.location.z)

    camera.location.x = 0.0
    camera.location.y = 0.0
    camera.location.z = args.topview_z
    objs_export = deepcopy(world['objects'])
    topview = render_one_view(scene_root, 'topview', render_args, objs_export,
                              blender_objects)
    camera.location.x = orig_cam['x']
    camera.location.y = orig_cam['y']
    camera.location.z = orig_cam['z']

    ########################
    # Generate rotating camera views
    # adopted from https://github.com/loganbruns/clevr-dataset-gen
    ########################
    r = np.linalg.norm([
        bpy.data.objects['Camera'].location.x,
        bpy.data.objects['Camera'].location.y
    ])

    delta_radians = 2 * np.pi / sequence_length
    theta = 0.
    # Record data about the object in the scene data structure
    views = []
    for img_idx in range(sequence_length):
        objs_export = deepcopy(world['objects'])
        camera = bpy.data.objects['Camera']
        camera.location.x = r * np.cos(theta)
        camera.location.y = r * np.sin(theta)
        camera.rotation_euler.z += delta_radians
        for i in range(3):
            camera.location[i] += rand(args.camera_jitter)

        view = render_one_view(scene_root, img_idx, render_args, objs_export,
                               blender_objects)
        theta += delta_radians
        views.append(view)

    colormap_export = np.vectorize(convert_to_srgb)(
        colormap)  # subtracting background color 64
    colormap_export = (colormap_export * 255).round().astype(np.uint8) - 64

    for i, o in enumerate(world['objects']):
        o['location'] = tuple(o['location'])
        o['bbox'] = [b.to_tuple() for b in o['bbox']]
        del o['mask_color_render']
        o['mask_color']

    scene = {'world': world, 'topview': topview, 'views': views}

    # Replace pickle later
    with open(str(scene_path), 'wb') as f:
        pickle.dump(scene, f, pickle.HIGHEST_PROTOCOL)


def render_one_view(scene_root, img_idx, render_args, objs_export,
                    blender_objects):
    bpy.context.scene.update_tag()
    img = render_single(render_args)

    for o in objs_export:
        pixel_coords = utils.get_camera_coords(bpy.data.objects['Camera'],
                                               o['location'])
        o['location'] = tuple(o['location'])
        o['pixel_coords'] = pixel_coords
        del o['bbox']

    mask = render_masks(scene_root, img_idx, render_args, objs_export,
                        blender_objects)
    for m, o in zip(mask, objs_export):
        del o['mask_color_render']

    export = {}
    export['image'] = img
    export['mask'] = mask
    export['objects'] = objs_export
    export['camera'] = {}
    export['camera']['location'] = [
        bpy.data.objects['Camera'].location.x,
        bpy.data.objects['Camera'].location.y,
        bpy.data.objects['Camera'].location.z
    ]

    cam = bpy.data.objects['Camera']
    cam_up = cam.matrix_world.to_quaternion() * Vector((0.0, 1.0, 0.0))
    cam_direction = cam.matrix_world.to_quaternion() * Vector((0.0, 0.0, -1.0))
    cam_up.normalized()
    cam_direction = cam_direction.normalized()
    export['camera']['lookat'] = [
        cam_direction.x, cam_direction.y, cam_direction.z
    ]
    export['camera']['up'] = [cam_up.x, cam_up.y, cam_up.z]
    return export


def convert_to_srgb(val):
    # convert image pixel values from 8bit to 32bit properly
    if val <= 0.0031308:
        return val * 12.92
    else:
        return 1.055 * (val**(1.0 / 2.4)) - 0.055


def render_masks(scene_root, img_idx, render_args, objs_export,
                 blender_objects):
    # Keep the original configuration
    # And prepare shadeless rendering to extract mask easier
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)

    # Assign colors for each objects
    old_materials = []
    for i, ((obj, c), o) in enumerate(zip(blender_objects, objs_export)):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'shadeless_%d' % i
        mat.diffuse_color = (o['mask_color_render'], o['mask_color_render'],
                             o['mask_color_render'])
        mat.use_shadeless = True
        obj.data.materials[0] = mat
        bpy.context.scene.update_tag()

    # Render
    # filepath = str(scene_root / '{}_{}.png'.format(img_idx, 'mask'))
    mask_img = render_single(
        render_args) - 64  # Subtracting background color 64
    mask_img = mask_img[:, :, 0]  # making as single channel image

    # Revert all changes back to original
    for mat, (obj, c) in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat
    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing
    bpy.context.scene.update_tag()

    return mask_img


def render_single(render_args):
    if not hasattr(render_single, 'temppath'):
        f, temppath = tempfile.mkstemp(suffix='.png')
        render_single.temppath = temppath
    temppath = render_single.temppath

    render_args.filepath = temppath
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)
    img = bpy.data.images.load(temppath)
    np_img = np.array(list(img.pixels)).reshape(img.size[0], img.size[1], 4)
    os.remove(temppath)
    return (np_img[::-1, :, :3] * 255).astype(np.uint8)


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
                dict(shape=data['obj'][1],
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
    om = obj.matrix_world

    worldify = lambda p: om * Vector(p[:])
    world_coords = [worldify(p) for p in local_coords]
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
    return dict(obj=(obj_name, obj_name_out),
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
