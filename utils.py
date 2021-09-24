import hashlib

def open_file(path, mode = 'r'):
	with open(path, mode) as source:
		return source.read()

def encode(string, typ = "utf8"):
	return string.encode(typ)

def generate_hash(data):
	if not isinstance(data, (str, bytes)):
		data = str(data)

	if not isinstance(data, bytes):
		data = data.encode()

	m = hashlib.md5() #sha256
	m.update(data)
	return m.hexdigest() #digest()
