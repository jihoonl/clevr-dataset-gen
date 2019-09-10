# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import os
import random
import sys

import numpy as np

import bpy
import bpy_extras
"""
Some utility functions for interacting with Blender
"""


def extract_args(input_argv=None):
    """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv


def parse_args(parser, argv=None):
    return parser.parse_args(extract_args(argv))


# I wonder if there's a better way to do this?
def delete_object(obj):
    """ Delete a specified blender object """
    for o in bpy.data.objects:
        o.select = False
    obj.select = True
    bpy.ops.object.delete()


def get_pixel_bbox(cam, bboxes):
    projected = np.zeros((len(bboxes), 3))

    for i, b in enumerate(bboxes):
        projected[i, :] = get_camera_coords(cam, b)

    return projected.tolist()

def get_camera_coords(cam, pos):
    """
  For a specified point, get both the 3D coordinates and 2D pixel-space
  coordinates of the point from the perspective of the camera.

  Inputs:
  - cam: Camera object
  - pos: Vector giving 3D world-space position

  Returns a tuple of:
  - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
    in the range [-1, 1]
  """
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale = scene.render.resolution_percentage / 100.0
    w = int(scale * scene.render.resolution_x)
    h = int(scale * scene.render.resolution_y)
    px = int(round(x * w))
    py = int(round(h - y * h))
    return (px, py, z)


def set_layer(obj, layer_idx):
    """ Move an object to a particular layer """
    # Set the target layer to True first because an object must always be on
    # at least one layer.
    obj.layers[layer_idx] = True
    for i in range(len(obj.layers)):
        obj.layers[i] = (i == layer_idx)


def add_object(object_dir, name, scale, loc, theta, mat, color):
    """
  Load an object from a file. We assume that in the directory object_dir, there
  is a file named "$name.blend" which contains a single object named "$name"
  that has unit size and is centered at the origin.

  - scale: scalar giving the size that the object should be in the scene
  - loc: tuple (x, y) giving the coordinates on the ground plane where the
    object should be placed.
  """
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
    bpy.ops.wm.append(filename=filename)

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    x, y = loc
    bpy.context.scene.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.transform.translate(value=(x, y, scale))

    add_material(mat, Color=color)


def load_materials(material_dir):
    """
  Load materials from a directory. We assume that the directory contains .blend
  files with one material each. The file X.blend has a single NodeTree item named
  X; this NodeTree item must have a "Color" input that accepts an RGBA value.
  """
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'): continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.join(material_dir, fn, 'NodeTree', name)
        bpy.ops.wm.append(filename=filepath)


def add_material(name, **properties):
    """
  Create a new material and assign it to the active object. "name" should be the
  name of a material that has been previously loaded using load_materials.
  """
    # Figure out how many materials are already in the scene
    mat_count = len(bpy.data.materials)

    # Create a new material; it is not attached to anything and
    # it will be called "Material"
    bpy.ops.material.new()

    # Get a reference to the material we just created and rename it;
    # then the next time we make a new material it will still be called
    # "Material" and we will still be able to look it up by name
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

    # Attach the new material to the active object
    # Make sure it doesn't already have materials
    obj = bpy.context.active_object
    assert len(obj.data.materials) == 0
    obj.data.materials.append(mat)

    # Find the output node of the new material
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break

    # Add a new GroupNode to the node tree of the active material,
    # and copy the node tree from the preloaded node group to the
    # new group node. This copying seems to happen by-value, so
    # we can create multiple materials of the same type without them
    # clobbering each other
    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    group_node.node_tree = bpy.data.node_groups[name]

    # Find and set the "Color" input of the new group node
    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    # Wire the output of the new group node to the input of
    # the MaterialOutput node
    mat.node_tree.links.new(
        group_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )


    def __str__(self):
        return "<Box, x=%i, y=%i, width=%i, height=%i>" % \
               (self.x, self.y, self.width, self.height)

    def to_tuple(self):
        if self.width == 0 or self.height == 0:
            return (0, 0, 0, 0)
        return (self.x, self.y, self.width, self.height)


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    from https://blender.stackexchange.com/questions/7198/save-the-2d-bounding-box-of-an-object-in-rendered-image-to-a-text-file

    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    me = me_ob.to_mesh(scene, True, 'PREVIEW')
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            #if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    bpy.data.meshes.remove(me)

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

