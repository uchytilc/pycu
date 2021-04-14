import hashlib

def open_file(path, mode = 'r'):
	with open(path, mode) as source:
		return source.read()

def encode(data, typ = "utf8"):
	return data.encode(typ)

def generate_hash(string):
	if not isinstance(string, bytes):
		string = string.encode()

	m = hashlib.md5() #sha256
	m.update(string)
	return m.hexdigest() #digest()
