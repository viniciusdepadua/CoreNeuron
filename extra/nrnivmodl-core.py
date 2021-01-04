#!/bin/python

import argparse
import glob
import os
import shutil
import subprocess

def get_root():
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(parent_dir, '..'))

class ModFiles():
    def __init__(self, d, work_dir):
        self.dir = d
        self.work_dir = work_dir
        self.basic_dir = os.path.join(get_root(), 'share', 'modfile')

    def get_files(self):
        files = glob.glob(os.path.join(self.basic_dir, '*.mod'))
        files.extend(glob.glob(os.path.join(self.dir, '*.mod')))
        return files

    """Artifical cell file cannot be processed by ispc so let's them be processed by GCC."""
    def get_ispc_files(self):
        cpp_files = list()
        ispc_files = list()

        for f in self.get_files():
            with open(f, "r") as h:
                if "ARTIFICIAL_CELL" in h.read():
                    cpp_files.append(f)
                else:
                    ispc_files.append(f)
        return (cpp_files, ispc_files)

    def get_ispc_files_for_rules(self, ispc_files):
        l = list()
        for f in ispc_files:
            filename = os.path.basename(os.path.splitext(f)[0])
            l.append({'mod_file': f, 'ispc_file': filename+".ispc", 'obj_file': filename+".obj", 'cpp_file': filename+".cpp", 'o_file': filename+".o"})
        return l

    def get_cpp_files_for_rules(self, cpp_files):
        l = list()
        for f in cpp_files:
            filename = os.path.basename(os.path.splitext(f)[0])
            l.append({'mod_file': f, 'cpp_file': filename+".cpp", 'o_file': filename+".o"})
        return l

rules = {
    'ispc': r'''
$(MOD_TO_CPP_DIR)/{ispc_file}: {mod_file}
	$(info Generating for {mod_file})
	$(MOD2CPP_ENV_VAR) $(MOD2CPP_BINARY_PATH) $< -o $(MOD_TO_CPP_DIR) $(MOD2CPP_BINARY_FLAG)

$(MOD_OBJS_DIR)/{obj_file}: $(MOD_TO_CPP_DIR)/{ispc_file}
	$(ISPC_COMPILE_CMD) $< -o $@

$(MOD_TO_CPP_DIR)/{cpp_file}: $(MOD_TO_CPP_DIR)/{ispc_file}

$(MOD_OBJS_DIR)/{o_file}: $(MOD_TO_CPP_DIR)/{cpp_file}
	$(CXX_COMPILE_CMD) -c $< -o $@

''',
    'cpp': r'''
$(MOD_TO_CPP_DIR)/{cpp_file}: {mod_file}
	$(info Generating for {mod_file})
	$(MOD2CPP_ENV_VAR) $(MOD2CPP_BINARY_PATH) $< -o $(MOD_TO_CPP_DIR) $(NMODL_FLAGS_C)

$(MOD_OBJS_DIR)/{o_file}: $(MOD_TO_CPP_DIR)/{cpp_file} $(KINDERIV_H_PATH)
	$(CXX_COMPILE_CMD) -c $< -o $@

''',
    'general': r'''
MOD_FILES = {mod_files}
PRODUCED_OBJS_FROM_ISPC = {obj_ispc_files}
PRODUCED_OBJS_FROM_CPP = {obj_cpp_files}

'''
}

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--binary')
    parser.add_argument('--nmodl', action='store_true')

    host_group = parser.add_mutually_exclusive_group()
    host_group.add_argument('--cpp', action='store_const', dest='host_backend', const='cpp')
    host_group.add_argument('--ispc', action='store_const', dest='host_backend', const='ispc')
    host_group.add_argument('--omp', action='store_const', dest='host_backend', const='omp')
    parser.set_defaults(host_backend='cpp')

    parser.add_argument('--gpu', choices=['cuda', 'OpenAcc'], const='OpenAcc', nargs='?')

    parser.add_argument('--work-dir', default=os.path.abspath("./output"))

    parser.add_argument('--build-type', choices=['STATIC', 'SHARED'])

    parser.add_argument('--suffix')

    parser.add_argument('-j', '--jobs', type=int, default=4)
    parser.add_argument('--output-dir')
    parser.add_argument('-v', '--verbose', action='store_true')

    parser.add_argument('mod_dir')

    return parser.parse_args()

"""Generate drop-in"""
class MakefileGenerator():
    def __init__(self, arguments, files):
        self.arguments = arguments
        self.files = files

    def generate(self):
        s = str()
        s += self.generateCompilers()
        s += self.generateRules()

        return s

    def generateHost(self):
        if self.arguments.nmodl:
            if self.arguments.host_backend == 'cpp':
                return 'host --c'
            elif self.arguments.host_backend == 'ispc':
                return 'host --ispc'
            elif self.arguments.host_backend == 'omp':
                return 'host --omp'
        return ''

    def generateGPU(self):
        if self.arguments.gpu:
            if self.arguments.nmodl:
                if self.arguments.gpu == 'OpenAcc':
                    return 'acc --oacc'
                elif self.arguments.gpu == 'cuda':
                    return 'acc --cuda'
                else:
                    raise "Error"
        return ''

    def generatePasses(self):
        if self.arguments.nmodl:
            return 'passes --inline'
        return ''

    def generateCompilers(self):
        s = str()
        if self.arguments.binary:
            s += 'MOD2CPP_BINARY_PATH = {}'.format(self.arguments.binary)
        else:
            s += 'MOD2CPP_BINARY_PATH = {}'.format('$(NMODL_COMPILER)' if self.arguments.nmodl else '$(MOD2C_COMPILER)')
        s += '\n'
        s += 'MOD2CPP_BINARY_FLAG = ' + self.generateHost() + " " + self.generateGPU() + " " + self.generatePasses()
        s += '\n'

        return s

    def generateRules(self):
        s = str()
        mod_files = [os.path.basename(x) for x in self.files.get_files()]
        if self.arguments.host_backend == 'ispc':
            cpp_files, ispc_files = self.files.get_ispc_files()
            cpp_files = self.files.get_cpp_files_for_rules(cpp_files)
            ispc_files = self.files.get_ispc_files_for_rules(ispc_files)
            obj_ispc_files = ['$(MOD_OBJS_DIR)/' + x['obj_file'] for x in ispc_files]
            obj_cpp_files = ['$(MOD_OBJS_DIR)/' + x['o_file'] for x in cpp_files]
            obj_cpp_files.extend(['$(MOD_OBJS_DIR)/' + x['o_file'] for x in ispc_files])
            s += rules['general'].format(obj_ispc_files=' '.join(obj_ispc_files), obj_cpp_files=' '.join(obj_cpp_files), mod_files=' '.join(mod_files))
            for f in ispc_files:
                s += rules['ispc'].format(**f, work_dir=arguments.work_dir)
        else:
            files = self.files.get_files()
            cpp_files = self.files.get_cpp_files_for_rules(files)
            obj_cpp_files = ['$(MOD_OBJS_DIR)/' + x['o_file'] for x in cpp_files]
            s += rules['general'].format(obj_ispc_files='', obj_cpp_files=' '.join(obj_cpp_files), mod_files=' '.join(mod_files))

        for f in cpp_files:
            s += rules['cpp'].format(**f, work_dir=arguments.work_dir)

        return s

if __name__ == '__main__':
    arguments = parse_args()
    if not os.path.exists(arguments.work_dir):
        os.makedirs(arguments.work_dir)
    files = ModFiles(arguments.mod_dir, arguments.work_dir)
    print("Output dir = 'make -f {}'".format(arguments.work_dir))
    G = MakefileGenerator(arguments, files)
    Makefile = G.generate()
    with open(os.path.join(arguments.work_dir, "GeneratedMakefile.make"), "w") as h:
        h.write(Makefile)
    shutil.copy(os.path.join(get_root(), 'share', 'coreneuron', 'nrnivmodl_core_makefile'), arguments.work_dir)

    make_args = [ 'make' ]
    make_args.append('-f{}'.format(os.path.join(arguments.work_dir, 'nrnivmodl_core_makefile')))
    make_args.append('-j{}'.format(arguments.jobs))
    make_args.append('ROOT={}'.format(get_root()))

    if arguments.verbose:
        make_args.append('VERBOSE=1')

    if arguments.binary:
        make_args.append('MOD2CPP_BINARY={}'.format(arguments.binary))

    if arguments.build_type:
        make_args.append('BUILD_TYPE={}'.format(arguments.build_type))

    if arguments.suffix:
        make_args.append('MECHLIB_SUFFIX={}'.format(arguments.suffix))

    if arguments.output_dir:
        make_args.append('DESTDIR={}'.format(arguments.output_dir))
        make_args.append('install') 
    else:
        make_args.append('all')

    make_args.append('WORK_DIR={}'.format(arguments.work_dir))

    print('Launching "{}"'.format(' '.join(make_args)))
    subprocess.call(make_args)
