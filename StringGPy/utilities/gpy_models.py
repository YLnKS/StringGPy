from GPy.models import GPHeteroscedasticRegression

class StringGPHeteroscedasticRegression(GPHeteroscedasticRegression):
	def __init__(self, X, Y, kernel=None):
		# Deduce noise variance memberships from the partition of the domain
		# 	which itself is to be inferred
		Y_metadata = kernel.Y_metadata(X)
		super(StringGPHeteroscedasticRegression, self).__init__(X, Y, kernel=kernel, Y_metadata=Y_metadata)


	'''
	Update noise variance mapping on paramters update if needed.
	'''
	def parameters_changed(self):
		if self.kern.learn_b_times:
			# Update Y_metadata to reflect a change in boundary time positions.
			self.Y_metadata = self.kern.Y_metadata(self.X)

		# Call the super-method
		super(StringGPHeteroscedasticRegression, self).parameters_changed()

	'''
	Update noise variance mapping on prediction if needed.
	'''
	def predict(self, Xnew, full_cov=False):
		Y_metadata = self.kern.Y_metadata(Xnew)
		return super(StringGPHeteroscedasticRegression, self).predict(Xnew, full_cov=full_cov,\
			Y_metadata=Y_metadata, kern=self.kern)
