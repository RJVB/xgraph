# a class that adds file-like functionality to a string, allowing notably to print to string instead of file:
class StrSink(str):
	def __init__(self, string=''):
		if( string == None ):
			string = ''
		self.string = string
		self.size = len(string)
#
	def __str__(self):
		return self.string
#
	def __repr__(self):
		return self.string
#
	def __len__(self):
		return self.size
	def __add__(self,text):
		self.string += text
		return self.string
#
	def flush(self):
		return None
#
	def read(self, size=-1):
		if( size> self.size ):
			return self.string
		elif( size >= 0 ):
			return self.string[0:size]
		else:
			return self.string
#
	def write(self, string):
		self.string += string
		self.size += len(string)
#
	def writelines(self, strings):
		for string in strings:
			self.write( self, string )
