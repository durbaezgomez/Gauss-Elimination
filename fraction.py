#	Own class, serves the purpose of representing fractions

from decimal import *

class Fraction:
	def __init__(self, a):
		self.nom = a 		#	licznik
		self.denom = 1 		#	mianiownik
		self.reduce()
		self.decimal = Decimal(self.nom)/Decimal(self.denom)
		return

	def __str__(self):
		return str(self.nom) + "/" + str(self.denom)

	def new_fraction(self):
		return Fraction(0,1)

	def new_fraction_one(self,x):
		return Fraction(x,x)

	@staticmethod
	def gcd(a, b):
		while (b != 0):
			temp = b
			b = a % b
			a = temp
		return a

	def __add__(self, other):
		ret = self
		ret.nom *= other.denom
		ret.nom += self.denom * other.nom
		ret.denom *= other.denom
		ret.reduce()
		return ret

	def __sub__(self, other):
		ret = self
		ret.nom *= other.denom
		ret.nom -= self.denom * other.nom
		ret.denom *= other.denom
		ret.reduce()
		return ret

	def __mul__(self, other):
		ret = self
		ret.nom *= other.nom
		ret.denom *= other.denom

		ret.reduce()
		return ret

	def __truediv__(self, other):
		ret = self
		ret.nom *= other.denom
		ret.denom *= other.nom

		ret.reduce()
		return ret

	def __iadd__(self, other):
		self.nom *= other.denom
		self.nom += self.denom * other.nom
		self.denom *= other.denom

		self.reduce()
		return self

	def __isub__(self, other):
		self.nom = self.nom * other.denom - self.denom * other.nom
		self.denom *= other.denom

		self.reduce()
		return self

	def __imul__(self, other):
		self.nom *= other.nom
		self.denom *= other.denom

		self.reduce()
		return self

	def __idiv__(self, other):
		self.nom *= other.denom
		self.denom *= other.nom

		self.reduce()
		return self

	def __lt__(self, other):
		if self.nom * other.denom < other.nom * self.denom :
			return True
		return False

	def abs(self):
		if self.nom >= 0:
			return self
		else:
			return (-1) * self

	def __gt__(self, other):
		if self.nom * other.denom > other.nom * self.denom :
			return True
		return False

	def reduce(self):
		if self.denom == 0 or self.nom == 0 :
			return

		divisor = self.gcd(self.nom, self.denom)
		if divisor == 0 :
			divisor = 1

		self.denom /= divisor
		self.nom /= divisor
		minusone = -1

		if self.denom < 0 :
			if self.nom >= 0 :
				self.nom *= minusone
				self.denom *= minusone
			else :
				self.nom *= minusone
				self.denom *= minusone