# def file(name, mode='r',buffering=None):
# 	import io
# 	if type(buffering) == int:
# 		ioObj = io.open(name,  mode,  buffering)
# 	else:
# 		ioObj = io.open(name,  mode)
# 	return ioObj

import io,sys
class file():
	def __init__(self, name, mode='r', buffering=None):
		if type(buffering) == int:
			self.ioObj = io.open(name,  mode,  buffering)
		else:
			self.ioObj = io.open(name,  mode)
		for m in dir(self.ioObj):
			if not m in ['__class__', '__dict__', '__doc__', '__name__']:
				setattr( self, m, getattr(self.ioObj, m) )
		setattr( self, '__doc__', 'Python2 file-like proxy layer over\n' + self.ioObj.__doc__ )

	def __repr__(self):
		return "<python2 file object " + self.ioObj.__repr__() + ">"
