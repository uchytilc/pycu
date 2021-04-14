from pycu.nvrtc import create_program, compile_program, destroy_program, get_ptx, get_ptx_size, add_name_expression, get_lowered_name
from pycu.utils import open_file

from .nvrtc_headers import nvrtc_headers

import os
import sys
from ctypes import c_char_p
import weakref

class NVRTCPtr:
	def __init__(self, handle):
		self.handle = handle

	def compile_program(self, options = {}):
		'''
		https://docs.nvidia.com/cuda/nvrtc/index.html#group__options

		Compilation targets
			--gpu-architecture=<arch> (-arch)
				Specify the name of the class of GPU architectures for which the input must be compiled.
					Valid <arch>s:
						compute_35
						compute_37
						compute_50
						compute_52 (Default)
						compute_53
						compute_60
						compute_61
						compute_62
						compute_70
						compute_72
						compute_75
						compute_80
		Separate compilation / whole-program compilation
			--device-c (-dc) 
				Generate relocatable code that can be linked with other relocatable device code. It is equivalent to --relocatable-device-code=true.
			--device-w (-dw)
				Generate non-relocatable code. It is equivalent to --relocatable-device-code=false.
			--relocatable-device-code={true|false} (-rdc)
				Enable (disable) the generation of relocatable device code.
				Default: false
			--extensible-whole-program (-ewp)
				Do extensible whole program compilation of device code.
				Default: false

		Debugging support
			--device-debug (-G)
				Generate debug information.
			--generate-line-info (-lineinfo)
				Generate line-number information.

		Code generation
			--maxrregcount=<N> (-maxrregcount)
				Specify the maximum amount of registers that GPU functions can use. Until a function-specific
				limit, a higher value will generally increase the performance of individual GPU threads that
				execute this function. However, because thread registers are allocated from a global register
				pool on each GPU, a higher value of this option will also reduce the maximum thread block size,
				thereby reducing the amount of thread parallelism. Hence, a good maxrregcount value is the
				result of a trade-off. If this option is not specified, then no maximum is assumed. Value less
				than the minimum registers required by ABI will be bumped up by the compiler to ABI minimum limit.
			--ftz={true|false} (-ftz)
				When performing single-precision floating-point operations, flush denormal values to zero or preserve denormal values. --use_fast_math implies --ftz=true.
				Default: false
			--prec-sqrt={true|false} (-prec-sqrt)
				For single-precision floating-point square root, use IEEE round-to-nearest mode or use a faster approximation. --use_fast_math implies --prec-sqrt=false.
				Default: true
			--prec-div={true|false} (-prec-div)
				For single-precision floating-point division and reciprocals, use IEEE round-to-nearest mode or use a faster approximation. --use_fast_math implies --prec-div=false.
				Default: true
			--fmad={true|false} (-fmad)
				Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA). --use_fast_math implies --fmad=true.
				Default: true
			--use_fast_math (-use_fast_math)
				Make use of fast math operations. --use_fast_math implies --ftz=true--prec-div=false--prec-sqrt=false--fmad=true.
			--extra-device-vectorization (-extra-device-vectorization)
				Enables more aggressive device code vectorization in the NVVM optimizer.

		Preprocessing
			--define-macro=<def> (-D)
			<def> can be either <name> or <name=definitions>.
				<name>
					Predefine <name> as a macro with definition 1.
				<name>=<definition>
					The contents of <definition> are tokenized and preprocessed as if they appeared during translation phase three in a #define directive. In particular, the definition will be truncated by embedded new line characters.
			--undefine-macro=<def> (-U)
				Cancel any previous definition of <def>.
			--include-path=<dir> (-I)
				Add the directory <dir> to the list of directories to be searched for headers. These paths are searched after the list of headers given to nvrtcCreateProgram.

			--pre-include=<header> (-include)
				Preinclude <header> during preprocessing.

		Language Dialect
			--std={c++03|c++11|c++14|c++17} (-std={c++11|c++14|c++17})
				Set language dialect to C++03, C++11, C++14 or C++17
			--builtin-move-forward={true|false} (-builtin-move-forward)
				Provide builtin definitions of std::move and std::forward, when C++11 language dialect is selected.
				Default: true

			--builtin-initializer-list={true|false} (-builtin-initializer-list)
				Provide builtin definitions of std::initializer_list class and member functions when C++11 language dialect is selected.
				Default: true

		Misc.
			--disable-warnings (-w)
				Inhibit all warning messages.
			--restrict (-restrict)
				Programmer assertion that all kernel pointer parameters are restrict pointers.
			--device-as-default-execution-space (-default-device)
				Treat entities with no execution space annotation as __device__ entities.
		'''

		arch = options.get("gpu-architecture", options.get("arch", 52))
		dc = options.get("device-c", options.get("dc", False))
		dw = options.get("device-w", options.get("dw", not dc))
		rdc = options.get("relocatable-device-code", options.get("rdc", dw))
		ftz = options.get("ftz", False)
		prec_sqrt = options.get("prec-sqrt", True)
		prec_div = options.get("prec-div", True)
		fmad = options.get("fmad", True)
		use_fast_math = options.get("use_fast_math", False)

		opts = [f"--gpu-architecture=compute_{arch}",
				f"--relocatable-device-code={rdc}".lower(),
				f"--ftz={ftz}".lower(),
				f"--prec-sqrt={prec_sqrt}".lower(),
				f"--prec-div={prec_div}".lower(),
				f"--fmad={fmad}".lower()]

		#jitify default
		if options.get("device-as-default-execution-space", options.get('default-device', True)): #False
			opts.append("-default-device")

		#TO DO
			#make options checking less stupid (more efficient)
		if options:
			if options.get("use_fast_math", False):
				opts.append("--use_fast_math")

			ewp = options.get("extensible-whole-program", options.get("-ewp", False))
			if ewp:
				opts.append("--extensible-whole-program")

			if options.get("device-debug", options.get('G', False)):
				opts.append("-G")

			if options.get("generate-line-info", options.get('lineinfo', False)):
				opts.append("-lineinfo")

			N = options.get("maxrregcount", 0)
			if N:
				opts.append(f"--maxrregcount={N}")

			if options.get("extra-device-vectorization", False):
				opts.append("--extra-device-vectorization")

			D = options.get("define-macro", options.get('D', ""))
			if D:
				opts.append(f"--define-macro={D}")

			U = options.get("undefine-macro", options.get('U', ""))
			if U:
				opts.append(f"--undefine-macro={U}")

			I = options.get("include-path", options.get('I', ""))
			if I:
				opts.append(f"--include-path={I}")

			header = options.get("pre-include", options.get('include', ""))
			if header:
				opts.append(f"--pre-include={header}")

			std = options.get("std", "")
			if std:
				opts.append(f"--std={std}")

			builtin_move_forward = options.get("builtin-move-forward", True)
			if builtin_move_forward:
				# if std == 'c++11':
				opts.append(f"--builtin-move-forward={builtin_move_forward}".lower())
				#else:
					#raise warning

			builtin_initializer_list = options.get("builtin-initializer-list", True)
			if builtin_initializer_list:
				# if std == 'c++11':
				opts.append(f"--builtin-initializer-list={builtin_initializer_list}".lower())
				#else:
					#raise warning

			if options.get("disable-warnings", options.get('w', False)):
				opts.append("--disable-warnings")

			if options.get("restrict", False):
				opts.append("--restrict")

		options = (c_char_p * len(opts))(*[c_char_p(opt.encode('utf8')) for opt in opts])
		compile_program(self.handle, options)
		return self.get_ptx()

	def get_ptx(self):
		return get_ptx(self.handle)

	def get_ptx_size(self):
		return get_ptx_size(self.handle)

	def add_name_expression(self, name_expression):
		add_name_expression(self.handle, name_expression.encode('utf8'))

	def get_lowered_name(self, name_expression):
		return get_lowered_name(self.handle, name_expression.encode('utf8'))

class NVRTCCache:
	def __init__(self):
		#absolute path to file
		#file source
		self.source = {}

		#absolute path to file
		#absolute path to header files contained within file
		self.headers = {}

		#absolute path to file
		#include names within file
		self.includes = {}

	def clear(self):
		self.source.clear()
		self.headers.clear()
		self.includes.clear()

def _prepare_headers(headers):
	#note: header names are the names exaclty how they appear in the c++ files
	header_src   = (c_char_p * len(headers))(*[c_char_p(header.encode('utf8')) for header in headers.values()])
	header_names = (c_char_p * len(headers))(*[c_char_p(name.encode('utf8')) for name in headers.keys()])

	return header_src, header_names

class NVRTC(NVRTCPtr):
	cache = NVRTCCache()

	def __init__(self, source, name = "", cache = True, I = [], load_headers = True):
		self.do_cache = cache

		# include_dir = "/usr/local/cuda/include/"
		I = [os.getcwd(),
			 os.path.dirname(os.path.realpath(__file__))] + I
			 #os.path.dirname(os.path.realpath(sys.argv[0]))

		headers = {}
		if load_headers:
			headers.update(self.load_headers(source, I = I))
		header_src, header_names = _prepare_headers(headers)

		source = source.encode('utf8')
		name = name.encode('utf8')

		# self.handle = handle = create_program(source, header_src, header_names, name)
		handle = create_program(source, header_src, header_names, name)
		weakref.finalize(self, destroy_program, handle)

		super().__init__(handle)

	def load_headers(self, source, I = []):
		includes = get_includes(source)

		queue = []
		for include in includes:
			hpath = find_header(include, I)
			hdir = hpath.rsplit(os.path.sep, 1)[0]

			queue.append((include, hpath))

			loaded = set()
			cache = self.cache
			if not self.do_cache:
				#create temporary cache to load headers but do not save
				cache = NVRTCCache()
			_load_headers(hpath, cache, loaded, [hdir] + I)

		#generate header name -> source mapping for NVRTC
		headers = {}
		while queue:
			include, path = queue.pop(0)
			headers[include] = cache.source[path]
			for include, path in zip(cache.includes[path], cache.headers[path]):
				if path not in headers:
					queue.append((include, path))

		return headers

def _load_headers(path, nvrtc_cache, loaded, I = []):
	#prevents infinite recursion with circular header dependencies
	if path in loaded:
		return

	if path not in nvrtc_cache.source:
		source = load_header(path)
		includes = get_includes(source)

		loaded.add(path)

		nvrtc_cache.source[path] = source
		nvrtc_cache.headers[path] = []
		nvrtc_cache.includes[path] = []

		for include in includes:
			hpath = find_header(include, I)
			hdir  = hpath.rsplit(os.path.sep, 1)[0]

			nvrtc_cache.headers[path].append(hpath)
			nvrtc_cache.includes[path].append(include) 

			#TO DO
				#Do I want to maintain all directories as I move down dir chain?
			_load_headers(hpath, nvrtc_cache, loaded, [hdir] + I[1:])

def load_header(path):
	#abs path to header file (if just a name check if it is a jitsafe header)
	if path in nvrtc_headers.keys():
		source = nvrtc_headers[path]
	else:
		source = open_file(path)
	return source

def find_header(include, paths):
	if include in nvrtc_headers.keys():
		return include
	for path in paths:
		hpath = os.path.join(path, include)
		if os.path.exists(hpath):
			# subpath, *include = include.rsplit(os.path.sep, 1)
			# if include:
			# 	path = os.path.join(path, subpath)
			return hpath
	nl = '\n'
	raise FileNotFoundError(f'{include} could not be found at any of the known paths {nl.join([path for path in paths])}')

def get_includes(source):
	#parse code loaded in for includes

	#TO DO:
		#Need to watch out for includes within compiler system defs (like _WIN32)
			#if ifdef != system.os skip all lines until next ifdef or endif

	lines = source.split('\n')
	includes = []
	for line in lines:
		if "#include" in line:
			header = get_include(line)
			if header:
				includes.append(header)

	return includes

def get_include(line):
	#extracts include name from within the line and checks if it is within a comment
	line, comment = clean_line(line)

	#check if include statement used "" or <> and get name of header
	header = line.split('"')
	if len(header) - 1:
		header = header[1]
	else:
		header = line.split('<', 1)[-1].split('>')[0]
	return header

def clean_line(line):
	cmt, n = first_substring(line, ["//", "/*"])

	if not cmt:
		return line, []

	line, comment = line.split(cmt, 1)
	comment = cmt + comment
	comments = [comment]

	#check if block comment ends in the same line it began (ex. code /*comment*/ more code)
	if cmt == "/*" and '*/' in comment:
		comment, _line = comment.rsplit('*/', 1)
		comments = [comment + '*/']
		_line, _comment = clean_line(_line)
		line += _line
		if _comment:
			comments.extend(_comment)

	return line, comments

def first_substring(string, substrings):
	#finds first occurrence between all substrings
	sub = ''
	pos = len(string)
	for substring in substrings:
		n = string.find(substring)
		if n > -1 and n < pos:
			pos = n
			sub = substring
	return sub, pos
