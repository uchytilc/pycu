from pycu.nvrtc import create_program, compile_program, destroy_program, get_ptx, get_ptx_size, add_name_expression, get_lowered_name, get_libcudadevrt
from pycu.utils import open_file

from .nvrtc_headers import nvrtc_headers

from enum import Enum, auto
import os
import sys
from ctypes import c_char_p
import weakref

def single_or_multi_arg(option, args):
	if isinstance(args, tuple):
		args = list(args)
	if not isinstance(args, list):
		args = [args]
	for n, arg in enumerate(args):
		if not isinstance(arg, str):
			arg = str(arg)
		if arg == 'True' or arg == 'False':
			args[n] = arg.lower()

	return [option.format(arg = arg) for arg in args]

_remap_nvrtc_options = {"arch":"gpu-architecture",
						"dc":"device-c",
						"dw":"device-w",
						"rdc":"relocatable-device-code",
						"ewp":"extensible-whole-program",
						"g":"device-debug",
						"lineinfo":"generate-line-info",
						"xptxas":"ptxas-options", #Xptxas
						"dlto":"dlink-time-opt",
						"d":"define-macro",
						"u":"undefine-macro",
						"i":"include-path",
						"include":"pre-include",
						"w":"disable-warnings",
						"default-device":"device-as-default-execution-space",
						"opt-info":"optimization-info",
						"dq":"version-ident", #dQ
						"err-no":"display-error-number"}

#--{option}
_options = {"device-c",
			"device-w",
			"extensible-whole-program",
			"device-debug",
			"generate-line-info",
			"use_fast_math",
			"extra-device-vectorization",
			"dlink-time-opt",
			"disable-warnings",
			"restrict",
			"device-as-default-execution-space",
			"display-error-number"}

#--{option}={args}
_options_arg = {"gpu-architecture",
				"relocatable-device-code",
				"ptxsa-options",
				"maxrregcount",
				"ftz",
				"prec-sqrt",
				"prec-div",
				"fmad",
				"modify-stack-limit",
				"define-macro",
				"undefine-macro",
				"include-path",
				"pre-include",
				"std",
				"builtin-move-forward",
				"builtin-initializer-list",
				"optimization-info",
				"version-ident",
				"diag-error",
				"diag-suppress",
				"diag-warn"}

def parse_nvrtc_options(options):
	opts = []

	for option, args in options.items():
		option = option.replace("-"," ").strip().replace(" ","-").lower()
		option = _remap_nvrtc_options.get(option, option)

		if option in _options:
			opts.append(f"--{option}")
		elif option in _options_arg:
			#The input for std and gpu-architecture needs to be formatted correctly if given as integers
			if option == 'std':
				if isinstance(args, int):
					args = str(args)
					args = args.zfill(2 - len(args))
				if args[:3] != "c++":
					args = f"c++{args}"
			elif option == 'gpu-architecture':
				if isinstance(args, int):
					args = f'compute_{args}'
			opts.extend(single_or_multi_arg(f"--{option}={{arg}}", args))

	return opts

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

		options = parse_nvrtc_options(options)
		options = (c_char_p * len(options))(*[c_char_p(option.encode('utf8')) for option in options])
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
		header_src, header_names = prepare_headers(headers)

		source = source.encode('utf8')
		name = name.encode('utf8')

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
				#create temporary cache to load headers
				cache = NVRTCCache()
			load_headers(hpath, cache, loaded, [hdir] + I)

		#generate header name -> source mapping for NVRTC
		headers = {}
		while queue:
			include, path = queue.pop(0)
			# print(include, path)
			headers[include] = cache.source[path]
			for include, path in zip(cache.includes[path], cache.headers[path]):
				if path not in headers:
					queue.append((include, path))

		# print(headers.keys())
		return headers

def prepare_headers(headers):
	#note: header names are the names exaclty how they appear in the c++ files
	header_src   = (c_char_p * len(headers))(*[c_char_p(header.encode('utf8')) for header in headers.values()])
	header_names = (c_char_p * len(headers))(*[c_char_p(name.encode('utf8')) for name in headers.keys()])

	return header_src, header_names

def load_headers(path, nvrtc_cache, loaded, I = []):
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
			try:
				hpath = find_header(include, I)
			except FileNotFoundError:
				raise FileNotFoundError(path)

			hdir  = hpath.rsplit(os.path.sep, 1)[0]

			nvrtc_cache.headers[path].append(hpath)
			nvrtc_cache.includes[path].append(include) 
			#TO DO
				#Do I want to maintain all directories as I move down dir chain?
			load_headers(hpath, nvrtc_cache, loaded, [hdir] + I[1:])

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
		hpath = os.path.join(path, replace_include_separators(include))
		if os.path.exists(hpath):
			# subpath, *include = include.rsplit(os.path.sep, 1)
			# if include:
			# 	path = os.path.join(path, subpath)
			return hpath
	nl = '\n'
	raise FileNotFoundError(f'{include} could not be found at any of the known paths\n{nl.join([str(path) for path in paths])}')

def replace_include_separators(header):
	#replace `/` used wtihinin include with the os specific seperators so the file can be found
	header = header.rsplit('/')
	for _ in range(len(header)):
		header.append(*(header.pop().rsplit('\\')))
	header = f"{os.path.sep}".join(header)

	return header

def get_includes(source):
	#TODO
		#defines should be kept for all headers included in a source file
	parser = Parser(source)
	parser.defines["__CUDACC__"] = True

	return parser.get_includes()

class TokenType(Enum):
	unknown = auto()
	backslash = auto()
	forwardslash = auto()
	period = auto()
	comma = auto()
	asterisk = auto()
	hashtag = auto()
	exclamation = auto()
	semicolon = auto()
	colon = auto()
	left_parenthesis = auto()
	right_parenthesis = auto()
	left_brace = auto()
	right_brace = auto()
	left_bracket = auto()
	right_bracket = auto()
	lt = auto()
	gt = auto()
	equal = auto()
	le = auto()
	ge = auto()
	string = auto()
	char = auto()
	newline = auto()
	text = auto()
	numeric = auto()
	end = auto()

class Token:
	def __init__(self, text, type):
		self.text = text
		self.type = type

	@property
	def length(self):
		return len(self.text)

	def __repr__(self):
		return self.text

	def __str__(self):
		return self.text

	def __eq__(self, other):
		return self.text == other

	def __hash__(self):
		return hash(self.text)

	# def __contains__(self, other):
	# 	return other in self.

class Tokenizer:
	def __init__(self, source):
		self.source = source
		self.length = len(source)
		self.pos = 0

	@staticmethod
	def is_EOL(char):
		return (char == '\n') or (char == '\r')

	@staticmethod
	def is_whitespace(char):
		return (char == ' ') or (char == '\t') or Tokenizer.is_EOL(char)

	@staticmethod
	def is_alpha(char):
		return ('a' <= char and char <= 'z') or ('A' <= char and char <= 'Z')

	@staticmethod
	def is_number(char):
		return '0' <= char and char <= '9'

	@staticmethod
	def is_numeric(char):
		return Tokenizer.is_number(char) or char == '.' or char == 'e' or char == '+' or char == '-'

	@staticmethod
	def is_text(char):
		return Tokenizer.is_alpha(char) or Tokenizer.is_number(char) or char == '_'

	# def probe_ahead(self, count, skip_newline = True):
		# #returns the next n tokens ahead of current token
		# tokens = []
		# pos = self.pos
		# for n in range(count):
		# 	tokens.append(self.get_token(skip_newline = skip_newline))
		# self.pos = pos

		# return tokens

	def get_token(self, skip_newline = True):
		#skip whitespace
		while True:
			if self.pos == self.length:
				return Token('', TokenType.end)
			elif self.is_whitespace(self.source[self.pos]):
				self.pos += 1
				if self.is_EOL(self.source[self.pos - 1]) and not skip_newline:
					return Token(self.source[self.pos - 1], TokenType.newline)
			#cpp style comment
			elif self.pos < self.length - 1 and self.source[self.pos] == '/' and self.source[self.pos + 1] == '/':
				#advance past the start comment
				self.pos += 2
				#parse until end of line
				while self.pos < self.length and not self.is_EOL(self.source[self.pos]):
					self.pos += 1
			#c style comment
			elif self.pos < self.length - 1 and self.source[self.pos] == '/' and self.source[self.pos + 1] == '*':
				#advance past the start comment
				self.pos += 2
				while self.pos < self.length - 1 and self.source[self.pos] == '*' and self.source[self.pos + 1] == '/':
					self.pos += 1
				#advance past the end comment
				if self.source[self.pos] == '*':
					self.pos += 2
			else:
				break

		start = self.pos
		self.pos += 1
		if start == self.length:
			return Token('', TokenType.end)
		elif self.source[start] == '\\':
			return Token(self.source[start:self.pos], TokenType.backslash)
		elif self.source[start] == '/':
			return Token(self.source[start:self.pos], TokenType.forwardslash)
		elif self.source[start] == '.':
			return Token(self.source[start:self.pos], TokenType.period)
		elif self.source[start] == ',':
			return Token(self.source[start:self.pos], TokenType.comma)
		elif self.source[start] == '*':
			return Token(self.source[start:self.pos], TokenType.asterisk)
		elif self.source[start] == '#':
			return Token(self.source[start:self.pos], TokenType.hashtag)
		elif self.source[start] == '!':
			return Token(self.source[start:self.pos], TokenType.hashtag)
		elif self.source[start] == ';':
			return Token(self.source[start:self.pos], TokenType.semicolon)
		elif self.source[start] == ':':
			return Token(self.source[start:self.pos], TokenType.colon)
		elif self.source[start] == '(':
			return Token(self.source[start:self.pos], TokenType.left_parenthesis)
		elif self.source[start] == ')':
			return Token(self.source[start:self.pos], TokenType.right_parenthesis)
		elif self.source[start] == '{':
			return Token(self.source[start:self.pos], TokenType.left_brace)
		elif self.source[start] == '}':
			return Token(self.source[start:self.pos], TokenType.right_brace)
		elif self.source[start] == '[':
			return Token(self.source[start:self.pos], TokenType.left_bracket)
		elif self.source[start] == ']':
			return Token(self.source[start:self.pos], TokenType.right_bracket)
		elif self.source[start] == '<':
			if self.source[self.pos + 1] == '=':
				self.pos += 1
				return Token(self.source[start:self.pos], TokenType.le)
			return Token(self.source[start:self.pos], TokenType.lt)
		elif self.source[start] == '>':
			if self.source[self.pos + 1] == '=':
				self.pos += 1
				return Token(self.source[start:self.pos], TokenType.ge)
			return Token(self.source[start:self.pos], TokenType.gt)
		elif self.source[start] == '=':
			if self.source[self.pos + 1] == '=':
				self.pos += 1
				return Token(self.source[start:self.pos], TokenType.eq)
			return Token(self.source[start:self.pos], TokenType.equal)
		# elif self.source[start] == "'":
			# start += 1
			# self.pos += 2
			# return Token(self.source[start:self.pos - 1], TokenType.char)
		elif self.source[start] == '"':
			start += 1
			while self.pos < self.length and self.source[self.pos] != '"':
				#avoid erroneous string ending of the string contains an escape character
				if self.pos < self.length and self.source[self.pos] == "\\":
					self.pos += 1
				self.pos += 1
			if self.pos == self.length:
				return Token('', TokenType.end)
			self.pos += 1
			return Token(self.source[start:self.pos - 1], TokenType.string)
		else:
			if self.is_alpha(self.source[start]) or self.source[start] == '_':
				while self.pos < self.length and self.is_text(self.source[self.pos]):
					self.pos += 1
				if self.pos == self.length:
					return Token('', TokenType.end)
				return Token(self.source[start:self.pos], TokenType.text)
			#note: this only recognizes negative numbers of the - sign is next to the number
			elif self.is_number(self.source[start]) or (self.source[start] == '.' and self.is_number(self.source[start + 1])) or (self.source[start] == '-' and self.is_number(self.source[start + 1])):
				while self.pos < self.length and self.is_numeric(self.source[self.pos]):
					self.pos += 1
				if self.pos == self.length:
					return Token('', TokenType.end)
				return Token(self.source[start:self.pos], TokenType.numeric)
			else:
				return Token('', TokenType.unknown)

#states
inactive = 0
active = 1
complete = 2

class Parser:
	#A rudimentary parser which attempts to find all valid #include statements within a c source file

	def __init__(self, source):
		self.tokenizer = Tokenizer(source)
		self.includes = []
		self.defines = {}
		self.state = [True]

	def get_includes(self):
		parse_source(self.tokenizer, self.includes, self.defines, self.state)
		# print(self.includes)
		return self.includes

def if_conditional(tokenizer, defines):
	token = tokenizer.get_token()

	#TODO
		#need to check for line wrapping character if conditional spans more than one line

	tokens = []
	while token.type != TokenType.newline:
		tokens.append(token)
		token = tokenizer.get_token(skip_newline = False)

	if len(tokens) == 1:
		pass
		#check if true, false, 0, 1
	else:
		pass
		#TODO
			#handle more complex define statements
		#if defined(...)

	return False

def ifdef_conditional(tokenizer, defines):
	return tokenizer.get_token() in defines

def ifndef_conditional(tokenizer, defines):
	return not ifdef_conditional(tokenizer, defines)

def undefine(tokenizer, defines):
	token = tokenizer.get_token()
	defines.remove(tokenizer.get_token().text)

def define(tokenizer, defines):
	token = tokenizer.get_token()

	#TODO
		#need to check for line wrapping character if define spans more than one line

	tokens = []
	while token.type != TokenType.newline:
		tokens.append(token)
		token = tokenizer.get_token(skip_newline = False)

	if len(tokens) == 1:
		defines[tokens[0].text] = None
	elif len(tokens) == 2:
		defines[tokens[0].text] = tokens[1].text
	else:
		pass
		#TODO
			#handle more complex define statements

def add_include(tokenizer, includes):
	token = tokenizer.get_token()
	if token.type == TokenType.lt:
		token = tokenizer.get_token()
		include = []
		while token != ">":
			include.append(token.text)
			token = tokenizer.get_token()
		includes.append(''.join(include))
	elif token.type == TokenType.string:
		includes.append(token.text)
	# else:
		# raise ValueError("include not understood")

def preprocessor_macro(tokenizer, includes, defines, state):
	#next token should be the preprocessor operation
	token = tokenizer.get_token()

	if token in {'if', 'ifdef', 'ifndef'}:
		#a given conditional parses from the entrence (if, ifdef, ifndef) until endif. If another if, ifdef, ifndef
		#is reached before the previous endif, a recursive call is made that itself advances until it finds an endif.
		#This will progress the parent function past the internal endif so it will not exit.

		#asses the truthness of the start of the conditional if it is within another active conditional (or the top most conditional)
		if state[-1] == active:
			if token == 'if':
				state.append(if_conditional(tokenizer, defines))
			elif token == 'ifdef':
				state.append(ifdef_conditional(tokenizer, defines))
			elif token == 'ifndef':
				state.append(ifndef_conditional(tokenizer, defines))
		else:
			state.append(Complete)
		parse_source(tokenizer, includes, defines, state)
	elif token in {'elif', 'else'} and state[-1] != complete:
		#a previous conditional was true and the conditional is complete
		if state[-1] == active:
			state[-1] = complete
		else:
			#asses the truthness of the start of the conditional if a previous conditional was not satisfied
			if token == 'elif':
				state[-1] = if_conditional(tokenizer, defines)
			elif token == 'else':
				state[-1] = active
	elif token == 'endif':
		state.pop()
	elif token == "include" and state[-1] == active:
		add_include(tokenizer, includes)
	elif token == "define" and state[-1] == active:
		define(tokenizer, defines)
	elif token == "undef" and state[-1] == active:
		undefine(tokenizer, defines)
	# else:
		# pass

def parse_source(tokenizer, includes, defines, state):
	while True:
		token = tokenizer.get_token()
		# print(f"{token}			{token.type}")

		#reached end of source
		if token.type == TokenType.end:
			break
		#entered a preprocessor statement
		elif token.type == TokenType.hashtag:
			preprocessor_macro(tokenizer, includes, defines, state)
		# else:
			# pass


# def get_includes(source):
# 	#parse code loaded in for includes

# 	#TO DO:
# 		#Need to watch out for includes within compiler system defs (like _WIN32)
# 			#if ifdef != system.os skip all lines until next ifdef or endif

# 	lines = source.split('\n')
# 	includes = []
# 	for line in lines:
# 		if "#include" in line:
# 			header = get_include(line)
# 			if header:
# 				includes.append(header)

# 	return includes

# def get_include(line):
# 	#extracts include name from within the line and checks if it is within a comment
# 	line, comment = clean_line(line)

# 	#check if include statement used "" or <> and get name of header
# 	header = line.split('"')
# 	if len(header) - 1:
# 		header = header[1]
# 	else:
# 		header = line.split('<', 1)[-1].split('>')[0]
# 	return header

# def clean_line(line):
# 	cmt, n = first_substring(line, ["//", "/*"])

# 	if not cmt:
# 		return line, []

# 	line, comment = line.split(cmt, 1)
# 	comment = cmt + comment
# 	comments = [comment]

# 	#check if block comment ends in the same line it began (ex. code /*comment*/ more code)
# 	if cmt == "/*" and '*/' in comment:
# 		comment, _line = comment.rsplit('*/', 1)
# 		comments = [comment + '*/']
# 		_line, _comment = clean_line(_line)
# 		line += _line
# 		if _comment:
# 			comments.extend(_comment)

# 	return line, comments

# def first_substring(string, substrings):
# 	#finds first occurrence between all substrings
# 	sub = ''
# 	pos = len(string)
# 	for substring in substrings:
# 		n = string.find(substring)
# 		if n > -1 and n < pos:
# 			pos = n
# 			sub = substring
# 	return sub, pos

