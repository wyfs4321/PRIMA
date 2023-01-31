# Maximum entropy distribution.
import numpy as np
import utils
import itertools
import math
from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, div


class Synthesizer:
	def value_attribute(single_list):
		k = int(len(single_list) / 2) + 1
		# k=len(single_list)
		candidate = {}
		value = []
		for key in single_list.keys():
			p = utils.pfs_to_p(single_list.get(key))
			candidate[key] = utils.info_entropy(p)
		candidate = sorted(candidate.items(), key=lambda kv: (kv[1], kv[0]))
		for i in range(k):
			value.append(candidate[i][0])
		# print(value)
		return value

	def min_value_attribute(single_list):
		k = 0
		# k=len(single_list)
		candidate = {}
		value = []
		for key in single_list.keys():
			p = utils.pfs_to_p(single_list.get(key))
			candidate[key] = utils.info_entropy(p)
		candidate = sorted(candidate.items(), key=lambda kv: (kv[1], kv[0]))
		for i in range(k):
			value.append(candidate[i][0])
		# print(value)
		return value

	def max_value_attribute(single_list):
		# k=int(len(single_list)/2)+1
		k = len(single_list)
		candidate = {}
		value = []
		for key in single_list.keys():
			p = utils.pfs_to_p(single_list.get(key))
			candidate[key] = utils.info_entropy(p)
		candidate = sorted(candidate.items(), key=lambda kv: (kv[1], kv[0]))
		for i in range(k):
			value.append(candidate[i][0])
		# print(value)
		return value

	def get_matrixA(single_list, domain):
		A1 = []
		for i in range(len(domain)):
			for j in range(domain[i]):
				a = np.zeros(domain, dtype=float)
				s = utils.cartesian()
				for k in domain:
					s.add_data(range(0, k))
				for k in itertools.product(*s._data_list):
					if k[i] == j:
						a[k] = 1
				A1.append(a.flatten())
		A1 = np.asarray(A1)
		A1 = matrix(A1)
		b1 = []
		for list in single_list:
			p = utils.pfs_to_p(list)
			p = list
			b1 = b1 + p
		# b.append(1.0)
		b1 = np.asarray(b1)
		b1 = matrix(b1)
		print(A1.size,b1.size)
		return A1, b1

	def get_matrixB(double_dict, domain):
		A2 = []
		b2 = []
		for record in double_dict:
			k = record
			v = double_dict.get(record)
			# print('v is:',v)
			s = utils.cartesian()
			for i in v.shape:
				s.add_data(range(0, i))
			for index1 in itertools.product(*s._data_list):
				a = np.zeros(domain, dtype=float)
				ss = utils.cartesian()
				for i in domain:
					ss.add_data(range(0, i))
				for index2 in itertools.product(*ss._data_list):
					if (index2[k[0]] == index1[0] and index2[k[1]] == index1[1]):
						a[index2] = 1
				A2.append(a.flatten())
			p = utils.pfs_to_p2(v)
			p = v
			# print('p is ',v)
			for i in p.flatten():
				b2.append(i)
		A2 = np.asarray(A2)
		A2 = matrix(A2)
		b2 = np.asarray(b2)
		b2 = matrix(b2)
		# print('A2 is',A2)
		# print('b2 is', b2,b2.size)
		return A2, b2

# single_list是一维数组
	def Maximum_entropy(single_list, double_dict, raw_domain, g):
		domain = []
		for i in range(len(raw_domain)):
			domain.append(math.ceil(raw_domain[i] / g))
		# print(single_list,double_dict)
		solvers.options['show_progress'] = False
		n = 1
		for d in domain:
			n = n * d
		print(single_list)
		if single_list != None and single_list != []:
			A1, b1 = Synthesizer.get_matrixA(single_list, domain)
		else:
			A1, b1 = matrix([]), matrix([])
		if double_dict != None and double_dict != {}:
			print('double full ')
			A2, b2 = Synthesizer.get_matrixB(double_dict, domain)
		else:
			print('double empty ')
			A2, b2 = matrix([]), matrix([])
		A3 = matrix(1, (1, n), 'd')
		b3 = matrix([1.0])
		# print('A2 is:',A2)
		# print(A1.size,A2.size,A3.size)
		# print(b1.size, b2.size, b3.size)
		if single_list == None or single_list == []:
			print('single_list ')
			A = matrix([A2, A3])
			b = matrix([b2, b3])
		elif double_dict == None or double_dict == {}:
			A = matrix([A1, A3])
			b = matrix([b1, b3])
		else:
			A = matrix([A1, A2, A3])
			b = matrix([b1, b2, b3])

		def F(x=None, z=None):
			# print(x)
			if x is None: return 0, matrix(1.0 / n, (n, 1))
			# implicit constraint that x should be non-negative
			if min(x) <= 0: return None
			f = x.T * log(x)
			grad = 1.0 + log(x)
			if z is None: return f, grad.T
			H = spdiag(z[0] * x ** -1)
			return f, grad.T, H

		# print(A.size, b.size)
		# print(max(A),min(A),max(b),min(b))
		# print(A,b,A3,b3)
		sol = solvers.cp(F, G=A, h=b, A=A3, b=b3)
		p = sol['x']
		# print(sum(p))
		p_array = np.array(p).reshape(domain)
		#print('p_array:',p_array)

		result = np.zeros(raw_domain, dtype=float)
		s = utils.cartesian()
		for i in raw_domain:
			s.add_data(range(0, i))
		for index in itertools.product(*s._data_list):
			i = tuple(np.asarray(index) // g)
			result[index] = p_array[i] / (g**len(domain))
		return result

	def get_matrixA_HDG(single_list, domain):
		A1 = []
		for i in range(len(domain)):
			for j in range(domain[i]):
				a = np.zeros(domain, dtype=float)
				s = utils.cartesian()
				for k in domain:
					s.add_data(range(0, k))
				for k in itertools.product(*s._data_list):
					if k[i] == j:
						a[k] = 1
				A1.append(a.flatten())
		A1 = np.asarray(A1)
		A1 = matrix(A1)
		b1 = []
		for lst in single_list:
			lst=list(lst)
			b1 = b1 + lst
		# b.append(1.0)
		b1 = np.asarray(b1)
		b1 = matrix(b1)
		return A1, b1

	def get_matrixB_HDG(double_dict, domain):
		A2 = []
		b2 = []
		for record in double_dict:
			k = record
			v = double_dict.get(record)
			# print('v is:',v)
			s = utils.cartesian()
			for i in v.shape:
				s.add_data(range(0, i))
			for index1 in itertools.product(*s._data_list):
				a = np.zeros(domain, dtype=float)
				ss = utils.cartesian()
				for i in domain:
					ss.add_data(range(0, i))
				for index2 in itertools.product(*ss._data_list):
					if (index2[k[0]] == index1[0] and index2[k[1]] == index1[1]):
						a[index2] = 1
				A2.append(a.flatten())
			# print('p is ',v)
			for i in v.flatten():
				b2.append(i)
		A2 = np.asarray(A2)
		A2 = matrix(A2)
		b2 = np.asarray(b2)
		b2 = matrix(b2)
		# print('A2 is',A2)
		# print('b2 is', b2,b2.size)
		return A2, b2

	def Maximum_entropy_HDG(single_list, double_dict, raw_domain,g):
		domain = []
		for i in range(len(raw_domain)):
			domain.append(math.ceil(raw_domain[i] / g))
		# print(single_list,double_dict)
		solvers.options['show_progress'] = False
		n = 1
		for d in domain:
			n = n * d
		if single_list != None and single_list != []:
			A1, b1 = Synthesizer.get_matrixA_HDG(single_list, domain)
		else:
			A1, b1 = matrix([]), matrix([])
		if double_dict != None and double_dict != {}:
			print('double full ')
			A2, b2 = Synthesizer.get_matrixB_HDG(double_dict, domain)
		else:
			print('double empty ')
			A2, b2 = matrix([]), matrix([])
		A3 = matrix(1, (1, n), 'd')
		b3 = matrix([1.0])
		# print('A2 is:',A2)
		# print(A1.size,A2.size,A3.size)
		# print(b1.size, b2.size, b3.size)
		if single_list == None or single_list == []:
			print('single_list ')
			A = matrix([A2, A3])
			b = matrix([b2, b3])
		elif double_dict == None or double_dict == {}:
			A = matrix([A1, A3])
			b = matrix([b1, b3])
		else:
			A = matrix([A1, A2, A3])
			b = matrix([b1, b2, b3])
		# print(A.size,b.size)

		def F(x=None, z=None):
			# print(x)
			if x is None: return 0, matrix(1.0 / n, (n, 1))
			# implicit constraint that x should be non-negative
			if min(x) <= 0: return None
			f = x.T * log(x)
			grad = 1.0 + log(x)
			if z is None: return f, grad.T
			H = spdiag(z[0] * x ** -1)
			return f, grad.T, H
		sol = solvers.cp(F, G=A, h=b, A=A3, b=b3)
		p = sol['x']
		# print(sum(p))
		p_array = np.array(p).reshape(domain)
		# print('p_array:',p_array)

		result = np.zeros(raw_domain, dtype=float)
		s = utils.cartesian()
		for i in raw_domain:
			s.add_data(range(0, i))
		for index in itertools.product(*s._data_list):
			i = tuple(np.asarray(index) // g)
			result[index] = p_array[i] / (g**len(domain))
		return result



if __name__ == '__main__':
	l = [[2, 3], [2, 3]]
	dict = {}
	domain = [3, 3]
	# dict[(0,1)]=np.asarray([[1,2,3],[2,4,6],[3,6,9]])

	print(Synthesizer.Maximum_entropy(l, dict, domain, 2))
	# Synthesizer.value_attribute(dict)

# sol = solvers.cp(F, G, h, A=A, b=b)
# p = sol['x']
