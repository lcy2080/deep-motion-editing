import os
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
from models.IK import fix_foot_contact
from os.path import join as pjoin

import platform

# downsampling and remove redundant joints
def copy_ref_file(src, dst):
    file = BVH_file(src)
    writer = BVH_writer(file.edges, file.names)
    writer.write_raw(file.to_tensor(quater=True)[..., ::2], 'quaternion', dst)


def get_height(file):
    file = BVH_file(file)
    return file.get_height()


def example(src_name, dest_name, bvh_name, test_type, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_file = './datasets/Mixamo/{}/{}'.format(src_name, bvh_name)
    ref_file = './datasets/Mixamo/{}/{}'.format(dest_name, bvh_name)

    if platform.system() == "Windows":
        copy_ref_file(input_file, pjoin(output_path, 'input.bvh').replace('\\', '/'))
        copy_ref_file(ref_file, pjoin(output_path, 'gt.bvh').replace('\\', '/'))
    else:
        copy_ref_file(input_file, pjoin(output_path, 'input.bvh'))
        copy_ref_file(ref_file, pjoin(output_path, 'gt.bvh'))
    height = get_height(input_file)

    bvh_name = bvh_name.replace(' ', '_')
    input_file = './datasets/Mixamo/{}/{}'.format(src_name, bvh_name)
    ref_file = './datasets/Mixamo/{}/{}'.format(dest_name, bvh_name)

    cmd = 'python eval_single_pair.py --input_bvh={} --target_bvh={} --output_filename={} --test_type={}'.format(
        input_file, ref_file, pjoin(output_path, 'result.bvh').replace('\\', '/') if platform.system() == "Windows" else pjoin(output_path, 'result.bvh'), test_type
    )
    os.system(cmd)

    result_path = pjoin(output_path, 'result.bvh').replace('\\', '/') if platform.system() == "Windows" else pjoin(output_path, 'result.bvh')
    input_path = pjoin(output_path, 'input.bvh').replace('\\', '/') if platform.system() == "Windows" else pjoin(output_path, 'input.bvh')

    fix_foot_contact(result_path,
                     input_path,
                     result_path,
                     height)


if __name__ == '__main__':
    example('Aj', 'BigVegas', 'Dancing Running Man.bvh', 'intra', './examples/intra_structure')
    example('BigVegas', 'Mousey_m', 'Dual Weapon Combo.bvh', 'cross', './examples/cross_structure')
    print('Finished!')
