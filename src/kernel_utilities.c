#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cStringGPy_ARRAY_API

#include "kernel_utilities.h"

/* ==== Evaluate the polynomial kernel for given inputs. ==== */
double poly_kernel(double x, double y, double *theta){
	return theta[0]*theta[0]*pow(x*y + theta[1], theta[2]);
}

/* ==== Evaluate the derivative of the polynomial kernel with respect to x. ==== */
double dpoly_kernel_dx(double x, double y, double *theta){
	return theta[0]*theta[0]*theta[2]*y*pow(x*y + theta[1], theta[2]-1);
}

/* ==== Evaluate the derivative of the polynomial kernel with respect to y. ==== */
double dpoly_kernel_dy(double x, double y, double *theta){
	return theta[0]*theta[0]*theta[2]*x*pow(x*y + theta[1], theta[2]-1);
}

/* ==== Evaluate the cross derivative of the polynomial kernel with respect to x and y. ==== */
double dpoly_kernel_dxdy(double x, double y, double *theta){
	return theta[0]*theta[0]*theta[2]*(pow(x*y + theta[1], theta[2]-1) +
		(theta[2]-1)*x*y*pow(x*y + theta[1], theta[2]-2));
}

/* ==== Cosine kernel with unit variance ==== */
 double cos_kernel(double x, double y, double freq){
    return cos(2.0*M_PI*(x-y)*freq);
}

/* ==== Cosine kernel with unit variance ==== */
 double dcos_kernel_dx(double x, double y, double freq){
    return -2.0*M_PI*sin(2.0*M_PI*(x-y)*freq);
}

/* ==== Cosine kernel with unit variance ==== */
 double dcos_kernel_dy(double x, double y, double freq){
    return 2.0*M_PI*sin(2.0*M_PI*(x-y)*freq);
}

/* ==== Cosine kernel with unit variance ==== */
 double dcos_kernel_dxdy(double x, double y, double freq){
    return 4.0*M_PI*M_PI*cos(2.0*M_PI*(x-y)*freq);
}

/* ==== Evaluate an SE kernel for given inputs. ==== */
double se_kernel(double x, double y, double *theta){
	return theta[0]*theta[0]*exp(-0.5*(x-y)*(x-y)/(theta[1]*theta[1]));
}

/* ==== Evaluate the derivative of the SE kernel with respect to x. ==== */
double dse_kernel_dx(double x, double y, double *theta){
	return -(theta[0]/theta[1])*(theta[0]/theta[1])*(x-y)*exp(-0.5*(x-y)*(x-y)/(theta[1]*theta[1]));
}

/* ==== Evaluate the derivative of the SE kernel with respect to y. ==== */
double dse_kernel_dy(double x, double y, double *theta){
	return dse_kernel_dx(y, x, theta);
}

/* ==== Evaluate the cross derivative of the SE kernel with respect to x and y. ==== */
double dse_kernel_dxdy(double x, double y, double *theta){
	return (theta[0]/theta[1])*(theta[0]/theta[1])*(1.0-((x-y)/theta[1])*((x-y)/theta[1]))*exp(-0.5*(x-y)*(x-y)/(theta[1]*theta[1]));
}

/* ==== Evaluate the RQ kernel for given inputs. ==== */
double rq_kernel(double x, double y, double *theta){
	return theta[0]*theta[0]*pow(1.0+(x-y)*(x-y)/(2.0*theta[2]*theta[1]*theta[1]), -theta[2]);
}

/* ==== Evaluate the derivative of the RQ kernel with respect to x. ==== */
double drq_kernel_dx(double x, double y, double *theta){
	return -pow(theta[0]/theta[1], 2)*(x-y)*pow(1.0+(x-y)*(x-y)/(2.0*theta[2]*theta[1]*theta[1]), -theta[2]-1.0);
}

/* ==== Evaluate the derivative of the RQ kernel with respect to y. ==== */
double drq_kernel_dy(double x, double y, double *theta){
	return drq_kernel_dx(y, x, theta);
}

/* ==== Evaluate the cross derivative of the RQ kernel with respect to x and y. ==== */
double drq_kernel_dxdy(double x, double y, double *theta){
	return pow(theta[0]/theta[1], 2)*pow(1.0+(x-y)*(x-y)/(2.0*theta[2]*theta[1]*theta[1]), -theta[2]-2.0)*(1.0+ pow((x-y)/theta[1], 2)*(-1.0-0.5/theta[2]));
}

/* ==== Evaluate the MA 1/2 kernel for given inputs. ==== */
double ma12_kernel(double x, double y, double *theta){
	return theta[0]*theta[0]*exp(-fabs(x-y)/theta[1]);
}

/* ==== Evaluate the MA 3/2 kernel for given inputs. ==== */
double ma32_kernel(double x, double y, double *theta){
	return theta[0]*theta[0]*(1+(sqrt(3.0)/theta[1])*fabs(x-y))*exp(-(sqrt(3.0)/theta[1])*fabs(x-y));
}

/* ==== Evaluate the derivative of the MA 3/2 kernel with respect to x. ==== */
double dma32_kernel_dx(double x, double y, double *theta){
	return -3.0*pow(theta[0]/theta[1], 2)*(x-y)*exp(-(sqrt(3.0)/theta[1])*fabs(x-y));
}

/* ==== Evaluate the derivative of the MA 3/2 kernel with respect to y. ==== */
double dma32_kernel_dy(double x, double y, double *theta){
	return dma32_kernel_dx(y, x, theta);
}

/* ==== Evaluate the cross derivative of the MA 3/2 kernel with respect to x and y. ==== */
double dma32_kernel_dxdy(double x, double y, double *theta){
	return 3.0*pow(theta[0]/theta[1], 2)*(1.0-(sqrt(3.0)/theta[1])*fabs(x-y))*exp(-(sqrt(3.0)/theta[1])*fabs(x-y));
}

/* ==== Evaluate the MA 5/2 kernel for given inputs. ==== */
double ma52_kernel(double x, double y, double *theta){
	return theta[0]*theta[0]*(1.0 + (sqrt(5.0)/theta[1])*fabs(x-y) + (5.0/(3.0*theta[1]*theta[1]))*(x-y)*(x-y))*exp(-(sqrt(5.0)/theta[1])*fabs(x-y));
}

/* ==== Evaluate the derivative of the MA 5/2 kernel with respect to x. ==== */
double dma52_kernel_dx(double x, double y, double *theta){
	return theta[0]*theta[0]*(x-y)*(-5.0/(3.0*theta[1]*theta[1])-(5.0*sqrt(5.0))/(3.0*theta[1]*theta[1]*theta[1])*fabs(x-y))*exp(-(sqrt(5.0)/theta[1])*fabs(x-y));
}

/* ==== Evaluate the derivative of the MA 3/2 kernel with respect to y. ==== */
double dma52_kernel_dy(double x, double y, double *theta){
	return dma52_kernel_dx(y, x, theta);
}

/* ==== Evaluate the cross derivative of the MA 5/2 kernel with respect to x and y. ==== */
double dma52_kernel_dxdy(double x, double y, double *theta){
	return theta[0]*theta[0]*(5.0/(3.0*theta[1]*theta[1]) + 5.0/(3.0*theta[1]*theta[1])*(sqrt(5.0)/theta[1])*fabs(x-y) -25.0/(3.0*pow(theta[1], 4))*(x-y)*(x-y))*exp(-(sqrt(5.0)/theta[1])*fabs(x-y)); 
}

/* ==== Evaluate the periodic kernel (McKay) for given inputs. ==== */
double period_kernel(double x, double y, double *theta){
	return pow(theta[0], 2.0)*exp(-2.0*pow(sin(M_PI*(x-y)/theta[2])/theta[1], 2.0));
}

/* ==== Evaluate the derivative of the periodic kernel (McKay) with respect to x. ==== */
double dperiod_kernel_dx(double x, double y, double *theta){
	return pow(theta[0]/theta[1], 2.0)*(-2.0*M_PI/theta[2])*sin(2.0*M_PI*(x-y)/theta[2])*exp(-2.0*pow(sin(M_PI*(x-y)/theta[2])/theta[1], 2.0));
}

/* ==== Evaluate the derivative of the periodic kernel (McKay) with respect to y. ==== */
double dperiod_kernel_dy(double x, double y, double *theta){
	return dperiod_kernel_dx(y, x, theta);
}

/* ==== Evaluate the cross derivative of the periodic kernel (McKay) with respect to x and y. ==== */
double dperiod_kernel_dxdy(double x, double y, double *theta){
	return -pow((2.0*M_PI*theta[0])/(theta[1]*theta[2]), 2.0)*(-cos(2.0*M_PI*(x-y)/theta[2])+pow(sin(2.0*M_PI*(x-y)/theta[2])/theta[1], 2.0))*exp(-2.0*pow(sin(M_PI*(x-y)/theta[2])/theta[1], 2.0));
}

/* ==== Evaluate the locally periodic kernel (McKay) for given inputs. ==== */
double loc_period_kernel(double x, double y, double *theta){
	return se_kernel(x, y, theta)*exp(-2.0*pow(sin(M_PI*(x-y)/theta[2])/theta[1], 2.0));
}

/* ==== Evaluate the derivative of the locally periodic kernel (McKay) with respect to x. ==== */
double dloc_period_kernel_dx(double x, double y, double *theta){
	return pow(1.0/theta[1], 2.0)*((-2.0*M_PI/theta[2])*sin(2.0*M_PI*(x-y)/theta[2])-(x-y))*loc_period_kernel(x, y, theta);
}

/* ==== Evaluate the derivative of the locally periodic kernel (McKay) with respect to y. ==== */
double dloc_period_kernel_dy(double x, double y, double *theta){
	return dloc_period_kernel_dx(y, x, theta);
}

/* ==== Evaluate the cross derivative of the locally periodic kernel (McKay) with respect to x and y. ==== */
double dloc_period_kernel_dxdy(double x, double y, double *theta){
	// Chain rule
	return pow(1.0/theta[1], 2.0)*((-2.0*M_PI/theta[2])*sin(2.0*M_PI*(x-y)/theta[2])-(x-y))*dloc_period_kernel_dy(x, y, theta)
		+ pow(1.0/theta[1], 2.0)*(pow(2.0*M_PI/theta[2], 2.0)*cos(2.0*M_PI*(x-y)/theta[2])+1.0)*loc_period_kernel(x, y, theta);
}

/* ==== Evaluate a Sparse Spectrum kernel for a given input distance ==== */
double ss_kernel(double x, double y, double *theta, unsigned int n){
	int i;
    double res=0.0;
   
    for(i=0; i<n; i++)
        res += theta[2*i]*theta[2*i]*cos_kernel(x, y, theta[1+2*i]);
    return res;
}

/* ==== Evaluate the derivative of the Sparse Spectrum kernel with respect to x. ==== */
double dss_kernel_dx(double x, double y, double *theta, unsigned int n){
	int i;
    double res=0.0;
   
    for(i=0; i<n; i++)
        res += theta[2*i]*theta[2*i]*dcos_kernel_dx(x, y, theta[1+2*i]);
    return res;
}

/* ==== Evaluate the derivative of the Sparse Spectrum kernel with respect to y. ==== */
double dss_kernel_dy(double x, double y, double *theta, unsigned int n){
	return dss_kernel_dx(y, x, theta, n);
}

/* ==== Evaluate the cross derivative of the Sparse Spectrum kernel with respect to x and y. ==== */
double dss_kernel_dxdy(double x, double y, double *theta, unsigned int n){
	int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += theta[2*i]*theta[2*i]*dcos_kernel_dxdy(x, y, theta[1+2*i]);
    return res;
}

/* ==== Evaluate a Spectral SE kernel for a given input distance ==== */
 double sse_kernel(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += se_kernel(x, y, &theta[3*i])*cos_kernel(x, y, theta[2+3*i]);
    return res;
}

/* ==== Derivative of the Spectral SE kernel with respect to x ==== */
 double dsse_kernel_dx(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += (dse_kernel_dx(x, y, &theta[3*i])*cos_kernel(x, y, theta[2+3*i])
                + se_kernel(x, y, &theta[3*i])*dcos_kernel_dx(x, y, theta[2+3*i]));
    return res;
}

/* ==== Derivative of the Spectral SE kernel with respect to x and y ==== */
 double dsse_kernel_dxdy(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += (dse_kernel_dxdy(x, y, &theta[3*i])*cos_kernel(x, y, theta[2+3*i])
                + dse_kernel_dx(x, y, &theta[3*i])*dcos_kernel_dy(x, y, theta[2+3*i])
                + dse_kernel_dy(x, y, &theta[3*i])*dcos_kernel_dx(x, y, theta[2+3*i])
                + se_kernel(x, y, &theta[3*i])*dcos_kernel_dxdy(x, y, theta[2+3*i]));
    return res;
}

/* ==== spectral Matern 1/2 ==== */
double sma12_kernel(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++){
        res += ma12_kernel(x, y, &theta[3*i])*cos_kernel(x, y, theta[2+3*i]);
    }
    return res;
}


/* ==== Spectral Matern 3/2 ==== */
 double sma32_kernel(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += ma32_kernel(x, y, &theta[3*i])*cos_kernel(x, y, theta[2+3*i]);
    return res;
}

/* ==== Derivative of the Spectral MA 3/2 kernel with respect to x ==== */
 double dsma32_kernel_dx(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += (dma32_kernel_dx(x, y, &theta[3*i])*cos_kernel(x, y, theta[2+3*i])
                + ma32_kernel(x, y, &theta[3*i])*dcos_kernel_dx(x, y, theta[2+3*i]));
    return res;
}

/* ==== Derivative of the Spectral MA 3/2 kernel with respect to x and y ==== */
 double dsma32_kernel_dxdy(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += (dma32_kernel_dxdy(x, y, &theta[3*i])*cos_kernel(x, y, theta[2+3*i])
                + dma32_kernel_dx(x, y, &theta[3*i])*dcos_kernel_dy(x, y, theta[2+3*i])
                + dma32_kernel_dy(x, y, &theta[3*i])*dcos_kernel_dx(x, y, theta[2+3*i])
                + ma32_kernel(x, y, &theta[3*i])*dcos_kernel_dxdy(x, y, theta[2+3*i]));
    return res;
}

/* ==== Spectral Matern 5/2 ==== */
 double sma52_kernel(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += ma52_kernel(x, y, &theta[3*i])*cos_kernel(x, y, theta[2+3*i]);
    return res;
}

/* ==== Derivative of the Spectral MA 5/2 kernel with respect to x ==== */
 double dsma52_kernel_dx(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += (dma52_kernel_dx(x, y, &theta[3*i])*cos_kernel(x, y, theta[2+3*i])
                + ma52_kernel(x, y, &theta[3*i])*dcos_kernel_dx(x, y, theta[2+3*i]));
    return res;
}

/* ==== Derivative of the Spectral MA 5/2 kernel with respect to x and y ==== */
 double dsma52_kernel_dxdy(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += (dma52_kernel_dxdy(x, y, &theta[3*i])*cos_kernel(x, y, theta[2+3*i])
                + dma52_kernel_dx(x, y, &theta[3*i])*dcos_kernel_dy(x, y, theta[2+3*i])
                + dma52_kernel_dy(x, y, &theta[3*i])*dcos_kernel_dx(x, y, theta[2+3*i])
                + ma52_kernel(x, y, &theta[3*i])*dcos_kernel_dxdy(x, y, theta[2+3*i]));
    return res;
}

/* ==== Spectral RQ ==== */
 double srq_kernel(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += rq_kernel(x, y, &theta[4*i])*cos_kernel(x, y, theta[3+4*i]);
    return res;
}

/* ==== Derivative of the Spectral RQ kernel with respect to x ==== */
 double dsrq_kernel_dx(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += (drq_kernel_dx(x, y, &theta[4*i])*cos_kernel(x, y, theta[3+4*i])
                + rq_kernel(x, y, &theta[4*i])*dcos_kernel_dx(x, y, theta[3+4*i]));
    return res;
}

/* ==== Derivative of the Spectral RQ kernel with respect to x and y ==== */
 double dsrq_kernel_dxdy(double x, double y, double *theta, unsigned int n){
    int i;
    double res=0.0;
    for(i=0; i<n; i++)
        res += (drq_kernel_dxdy(x, y, &theta[4*i])*cos_kernel(x, y, theta[3+4*i])
                + drq_kernel_dx(x, y, &theta[4*i])*dcos_kernel_dy(x, y, theta[3+4*i])
                + drq_kernel_dy(x, y, &theta[4*i])*dcos_kernel_dx(x, y, theta[3+4*i])
                + rq_kernel(x, y, &theta[4*i])*dcos_kernel_dxdy(x, y, theta[3+4*i]));
    return res;
}


/* ==== Evaluate a kernel for given inputs. ==== */
double kernel(double x, double y, double *theta, unsigned int n, char *type){
	if(strcmp(type, "poly") == 0)
		return poly_kernel(x, y, theta);

	if(strcmp(type, "se") == 0)
		return se_kernel(x, y, theta);
		
	if(strcmp(type, "rq") == 0)
		return rq_kernel(x, y, theta);

	if(strcmp(type, "ma12") == 0)
		return ma12_kernel(x, y, theta);
	
	if(strcmp(type, "ma32") == 0)
		return ma32_kernel(x, y, theta);
	
	if(strcmp(type, "ma52") == 0)
		return ma52_kernel(x, y, theta);
	
	if(strcmp(type, "period") == 0)
		return period_kernel(x, y, theta);
	
	if(strcmp(type, "loc_period") == 0)
		return loc_period_kernel(x, y, theta);
	
	if(strcmp(type, "sse") == 0)
		return sse_kernel(x, y, theta, n);

	if(strcmp(type, "sm") == 0)
		return sse_kernel(x, y, theta, n);

	if(strcmp(type, "ss") == 0)
		return ss_kernel(x, y, theta, n);

	if(strcmp(type, "sma12") == 0)
		return sma12_kernel(x, y, theta, n);

	if(strcmp(type, "sma32") == 0)
		return sma32_kernel(x, y, theta, n);

	if(strcmp(type, "sma52") == 0)
		return sma52_kernel(x, y, theta, n);

	if(strcmp(type, "srq") == 0)
		return srq_kernel(x, y, theta, n);
	
	return se_kernel(x, y, theta);

}

/* ==== Evaluate the derivative of a kernel with respect to x. ==== */
double dkernel_dx(double x, double y, double *theta, unsigned int n, char *type){
	if(strcmp(type, "poly") == 0)
		return dpoly_kernel_dx(x, y, theta);

	if(strcmp(type, "se") == 0)
		return dse_kernel_dx(x, y, theta);
		
	if(strcmp(type, "rq") == 0)
		return drq_kernel_dx(x, y, theta);
	
	if(strcmp(type, "ma32") == 0)
		return dma32_kernel_dx(x, y, theta);
	
	if(strcmp(type, "ma52") == 0)
		return dma52_kernel_dx(x, y, theta);

	if(strcmp(type, "period") == 0)
		return dperiod_kernel_dx(x, y, theta);
	
	if(strcmp(type, "loc_period") == 0)
		return dloc_period_kernel_dx(x, y, theta);
	
	if(strcmp(type, "sse") == 0)
		return dsse_kernel_dx(x, y, theta, n);

	if(strcmp(type, "sm") == 0)
		return dsse_kernel_dx(x, y, theta, n);

	if(strcmp(type, "sma32") == 0)
		return dsma32_kernel_dx(x, y, theta, n);

	if(strcmp(type, "sma52") == 0)
		return dsma52_kernel_dx(x, y, theta, n);

	if(strcmp(type, "srq") == 0)
		return dsrq_kernel_dx(x, y, theta, n);
	
	if(strcmp(type, "ss") == 0)
		return dss_kernel_dx(x, y, theta, n);
	
	return dse_kernel_dx(x, y, theta);
}

/* ==== Evaluate the derivative of a kernel with respect to y. ==== */
double dkernel_dy(double x, double y, double *theta, unsigned int n, char *type){
	return dkernel_dx(y, x, theta, n, type);
}

/* ==== Evaluate the cross derivative of a kernel with respect to x and y. ==== */
double dkernel_dxdy(double x, double y, double *theta, unsigned int n, char *type){
	if(strcmp(type, "poly") == 0)
		return dpoly_kernel_dxdy(x, y, theta);

	if(strcmp(type, "se") == 0)
		return dse_kernel_dxdy(x, y, theta);
		
	if(strcmp(type, "rq") == 0)
		return drq_kernel_dxdy(x, y, theta);
	
	if(strcmp(type, "ma32") == 0)
		return dma32_kernel_dxdy(x, y, theta);
	
	if(strcmp(type, "ma52") == 0)
		return dma52_kernel_dxdy(x, y, theta);
	
	if(strcmp(type, "period") == 0)
		return dperiod_kernel_dxdy(x, y, theta);
	
	if(strcmp(type, "loc_period") == 0)
		return dloc_period_kernel_dxdy(x, y, theta);
	
	if(strcmp(type, "sse") == 0)
		return dsse_kernel_dxdy(x, y, theta, n);

	if(strcmp(type, "sm") == 0)
		return dsse_kernel_dxdy(x, y, theta, n);

	if(strcmp(type, "sma32") == 0)
		return dsma32_kernel_dxdy(x, y, theta, n);

	if(strcmp(type, "sma52") == 0)
		return dsma52_kernel_dxdy(x, y, theta, n);

	if(strcmp(type, "srq") == 0)
		return dsrq_kernel_dxdy(x, y, theta, n);

	if(strcmp(type, "ss") == 0)
		return dss_kernel_dxdy(x, y, theta, n);
	
	return dse_kernel_dxdy(x, y, theta);	
}
/* === Determine the number of spectral components from a PyArrayObject === */
int n_spectral_comp(PyArrayObject* theta, char*  type){
	int n_mixt;
	if(strcmp(type, "srq") == 0){
		n_mixt=PyArray_DIMS(theta)[0]/4;
	}else if(strcmp(type, "ss") == 0){
		n_mixt=PyArray_DIMS(theta)[0]/2;
	}else{
		n_mixt=PyArray_DIMS(theta)[0]/3;
	}
	return n_mixt;
}


/* ==== Compute the cross-covariance matrix of a GP at X1 and X2 under a given kernel. ==== */
void cov(double **res, double *X1, unsigned int n_X1, double *X2, unsigned int n_X2, double *theta, unsigned int n_theta, char *type){
    unsigned int i,j;
    for (i=0; i<n_X1; i++){
        for(j=0; j<n_X2; j++){
            res[i][j]=kernel(X1[i], X2[j], theta, n_theta, type);
        }
    }
    return;
}


/* ==== Compute the cross-covariance matrix of a GP at X1 and X2 under a given kernel. ==== */
/* Unlike the previous function, the inputs X1 and X2 here are assumed to be multi-dimensional. The number n_col of columns of X1 and X2
	corresponds to the number of features. Each dimension has its own set of hyper-parameters. A product of kernels is used. theta should be n_theta x n_col.
*/
void cov_multi_prod(double **res, double **X1, unsigned int n_X1, double **X2, unsigned int n_X2, double **theta, unsigned int n_theta, unsigned int n_col, char *type){
    unsigned int i,j,k;
	double **theta_T;
	theta_T = new_zeros(n_col, n_theta);
	transpose_copy(theta_T, theta, n_theta, n_col);
	
    for (i=0; i<n_X1; i++){
        for(j=0; j<n_X2; j++){
			res[i][j] = 1.0;
			for(k=0; k<n_col; k++){
				res[i][j]*=kernel(X1[i][k], X2[j][k], theta_T[k], n_theta, type);
			}
        }
    }
	free_mem_matrix(theta_T, n_col);
    return;
}

/* ==== Compute the cross-covariance matrix of a GP at X1 and X2 under a given kernel. ==== */
/* Unlike the previous function, the inputs X1 and X2 here are assumed to be multi-dimensional. The number n_col of columns of X1 and X2
	corresponds to the number of features. Each dimension has its own set of hyper-parameters. A sum of kernels is used. theta should be n_theta x n_col.
*/
void cov_multi_sum(double **res, double **X1, unsigned int n_X1, double **X2, unsigned int n_X2, double **theta, unsigned int n_theta, unsigned int n_col, char *type){
    unsigned int i,j,k;
	double **theta_T;
	theta_T = new_zeros(n_col, n_theta);
	transpose_copy(theta_T, theta, n_theta, n_col);
	
    for (i=0; i<n_X1; i++){
        for(j=0; j<n_X2; j++){
			res[i][j] = 0.0;
			for(k=0; k<n_col; k++){
				res[i][j]+=kernel(X1[i][k], X2[j][k], theta_T[k], n_theta, type);
			}
        }
    }
	free_mem_matrix(theta_T, n_col);
    return;
}


/* ==== Compute the cross-covariance matrix of a derivative GP at X1 and X2 under a given kernel. ==== */
void deriv_cov(double **res, double *X1, unsigned int n_X1, double *X2, unsigned int n_X2, double *theta, unsigned int n_theta, char *type){
    unsigned int i,j;
    for (i=0; i<n_X1; i++){
        for(j=0; j<n_X2; j++){
            res[2*i][2*j]=kernel(X1[i], X2[j], theta, n_theta, type);
            res[1+2*i][2*j]=dkernel_dx(X1[i], X2[j], theta, n_theta, type);
            res[2*i][1+2*j]=dkernel_dy(X1[i], X2[j], theta, n_theta, type);
            res[1+2*i][1+2*j]=dkernel_dxdy(X1[i], X2[j], theta, n_theta, type);
        }
    }
    return; 
}

/* ==== Compute the cross-covariance matrix of a derivative GP at X1 and X2 under a given kernel. ==== */
/* Unlike the previous function, the inputs X1 and X2 here are assumed to be multi-dimensional. The number n_col of columns of X1 and X2
	corresponds to the number of features. Each dimension has its own set of hyper-parameters. A product of kernels is used. theta should be n_theta x n_col.
*/
void deriv_cov_multi_prod(double **res, double **X1, unsigned int n_X1, double **X2, unsigned int n_X2, double **theta, unsigned int n_theta, unsigned int n_col, char *type){
    unsigned int i,j,k;
	double **theta_T;
	theta_T = new_zeros(n_col, n_theta);
	transpose_copy(theta_T, theta, n_theta, n_col);
	
    for (i=0; i<n_X1; i++){
        for(j=0; j<n_X2; j++){
			res[i][j] = 1.0;
			for(k=0; k<n_col; k++){
				res[2*i][2*j]*=kernel(X1[i][k], X2[j][k], theta_T[k], n_theta, type);
				res[1+2*i][2*j]*=dkernel_dx(X1[i][k], X2[j][k], theta_T[k], n_theta, type);
				res[2*i][1+2*j]*=dkernel_dy(X1[i][k], X2[j][k], theta_T[k], n_theta, type);
				res[1+2*i][1+2*j]*=dkernel_dxdy(X1[i][k], X2[j][k], theta_T[k], n_theta, type);
			}
        }
    }
	free_mem_matrix(theta_T, n_col);
    return; 
}


/* ==== Compute the cross-covariance matrix of a derivative GP at X1 and X2 under a given kernel. ==== */
/* Unlike the previous function, the inputs X1 and X2 here are assumed to be multi-dimensional. The number n_col of columns of X1 and X2
	corresponds to the number of features. Each dimension has its own set of hyper-parameters. A sum of kernels is used. theta should be n_theta x n_col.
*/
void deriv_cov_multi_sum(double **res, double **X1, unsigned int n_X1, double **X2, unsigned int n_X2, double **theta, unsigned int n_theta, unsigned int n_col, char *type){
    unsigned int i,j,k;
	double **theta_T;
	theta_T = new_zeros(n_col, n_theta);
	transpose_copy(theta_T, theta, n_theta, n_col);
	
    for (i=0; i<n_X1; i++){
        for(j=0; j<n_X2; j++){
			res[i][j] = 0.0;
			for(k=0; k<n_col; k++){
				res[2*i][2*j]+=kernel(X1[i][k], X2[j][k], theta_T[k], n_theta, type);
				res[1+2*i][2*j]+=dkernel_dx(X1[i][k], X2[j][k], theta_T[k], n_theta, type);
				res[2*i][1+2*j]+=dkernel_dy(X1[i][k], X2[j][k], theta_T[k], n_theta, type);
				res[1+2*i][1+2*j]+=dkernel_dxdy(X1[i][k], X2[j][k], theta_T[k], n_theta, type);
			}
        }
    }
	free_mem_matrix(theta_T, n_col);
    return; 
}


/* ==== Compute the cross-covariance matrix of a GP at X1 and X2 under a Spectral Mixture kernel. ==== */
void sm_cov(double **res, double *X1, unsigned int n_X1, double *X2, unsigned int n_X2, double *theta, unsigned int n_theta){
	cov(res, X1, n_X1, X2, n_X2, theta, n_theta, "sse");	
}

/* ==== Compute the cross-covariance matrix of a derivative GP at X1 and X2 under a Spectral Mixture kernel. ==== */
void sm_deriv_cov(double **res, double *X1, unsigned int n_X1, double *X2, unsigned int n_X2, double *theta, unsigned int n_theta){
	deriv_cov(res, X1, n_X1, X2, n_X2, theta, n_theta, "sse");
}



/* ==== Compute the covariance matrix of a string derivative GP at boundary times, under a given kernel. ==== */
/* 
	res: pointer to the matrix to be updated in place. The matrix should be of size 2*n_b_times x 2*n_b_times.
	b_times: array of boundary times, should be sorted (the function will sort it otherwise, and the matrix indices will correspond to sorted boundary times).
	n_b_times: number of boundary times. Also 1 + the number of strings.
	thetas: matrix of string hyper-parameters.
	n_mixts: array of numbers of elements in the spectral mixture for each local expert kernel.
	type: kernel type (se, sm, rq, ma32, ma52)
*/
void string_boundaries_deriv_cov(double **res, double *b_times, unsigned int n_b_times, double **thetas, unsigned int n_mixts, char *type){
	unsigned int i,j,p,q;
	double **cov_a_p_1=NULL, **cov_a_p=NULL, **cov_a_p_a_p_1=NULL, **Mb=NULL, **cond_p_1_cov_p=NULL,
		**global_cov_p_1=NULL, **tmp=NULL, **tmp2=NULL;
	double a_p_1=0.0, a_p=0.0;
	
	// Sort boundary times in place
	qsort(b_times, n_b_times, sizeof(double), cmp_double);
	
	// Initialise all the 2x2 matrices needed.
	cov_a_p=new_zeros(2, 2);
	cov_a_p_1=new_zeros(2, 2);
	cov_a_p_a_p_1=new_zeros(2, 2);
	Mb=new_zeros(2, 2);
	cond_p_1_cov_p=new_zeros(2, 2);
	global_cov_p_1=new_zeros(2, 2);
	tmp=new_zeros(2, 2);
	tmp2=new_zeros(2, 2);
		
	for(p=1; p<n_b_times; p++){
		// First string: update the entries corresponding to the first two 
		// 	boundary times.
		if(p==1){
			for(i=0; i<4; i++){
				for(j=0; j<4; j++){

					if(((i%2)==0) & ((j%2)==0)){
						res[i][j]= kernel(b_times[i/2], b_times[j/2], (double *) thetas[0], n_mixts, type);
					}
					if(((i%2)==1) & ((j%2)==0)){
						res[i][j]=dkernel_dx(b_times[(i-1)/2], b_times[j/2], (double *) thetas[0], n_mixts, type);
					}	
					if(((i%2)==0) & ((j%2)==1)){
						res[i][j]=dkernel_dy(b_times[i/2], b_times[(j-1)/2], (double *) thetas[0], n_mixts, type);
					}
					if(((i%2)==1) & ((j%2)==1)){
						res[i][j]=dkernel_dxdy(b_times[(i-1)/2], b_times[(j-1)/2], (double *) thetas[0], n_mixts, type);
					} 

				}
			}			
		}
		// Other strings: update the entries sequentially.
		else{
			a_p_1=b_times[p-1];
			a_p=b_times[p];
			
			// Covariance matrices under the unconditional cov structure.
			cov_a_p_1[0][0]=kernel(a_p_1, a_p_1, thetas[p-1], n_mixts, type);
			cov_a_p_1[0][1]=dkernel_dy(a_p_1, a_p_1, thetas[p-1], n_mixts, type);
			cov_a_p_1[1][0]=dkernel_dx(a_p_1, a_p_1, thetas[p-1],  n_mixts, type);
			cov_a_p_1[1][1]=dkernel_dxdy(a_p_1, a_p_1, thetas[p-1], n_mixts, type);
			
			cov_a_p[0][0]=kernel(a_p, a_p, thetas[p-1], n_mixts, type);
			cov_a_p[0][1]=dkernel_dy(a_p, a_p, thetas[p-1], n_mixts, type);
			cov_a_p[1][0]=dkernel_dx(a_p, a_p, thetas[p-1], n_mixts, type);
			cov_a_p[1][1]=dkernel_dxdy(a_p, a_p, thetas[p-1], n_mixts, type);
			
			cov_a_p_a_p_1[0][0]=kernel(a_p, a_p_1, thetas[p-1], n_mixts, type);
			cov_a_p_a_p_1[0][1]=dkernel_dy(a_p, a_p_1, thetas[p-1], n_mixts, type);
			cov_a_p_a_p_1[1][0]=dkernel_dx(a_p, a_p_1,  thetas[p-1], n_mixts, type);
			cov_a_p_a_p_1[1][1]=dkernel_dxdy(a_p, a_p_1, thetas[p-1], n_mixts, type);
			
			//{}^b_k M coefficient
			invert2(tmp, cov_a_p_1); // tmp = np.linalg.inv(cov_a_p_1)
			matrix_prod(Mb, cov_a_p_a_p_1, 2, 2, tmp, 2); // Mb = np.dot(cov_a_p_a_p_1, np.linalg.inv(cov_a_p_1))
			
			// Covariance at p conditional on values at p-1
			transpose_copy(tmp2, cov_a_p_a_p_1, 2, 2); // tmp2=cov_a_p_a_p_1.T
			matrix_prod(tmp, Mb, 2, 2, tmp2, 2); //tmp=np.dot(Mb, cov_a_p_a_p_1.T)
			matrix_sub(cond_p_1_cov_p, cov_a_p, tmp, 2, 2); // cond_p_1_cov_p = cov_a_p - np.dot(Mb, cov_a_p_a_p_1.T)
			
			// Global covariance matrix of the boundary conditions at the previous time
			global_cov_p_1[0][0]=res[2*p-2][2*p-2];
			global_cov_p_1[0][1]=res[2*p-2][2*p-1];
			global_cov_p_1[1][0]=res[2*p-1][2*p-2];
			global_cov_p_1[1][1]=res[2*p-1][2*p-1];
			
			for(q=0; q<p+1; q++){
				if(q==p){
					transpose_copy(tmp, Mb, 2, 2); // tmp=Mb.T
					matrix_prod(tmp2, global_cov_p_1, 2, 2, tmp, 2); // tmp2=np.dot(global_cov_p_1, Mb.T)
					matrix_prod(tmp, Mb, 2, 2, tmp2, 2); // tmp=np.dot(Mb, np.dot(global_cov_p_1, Mb.T))
					matrix_add(tmp2, cond_p_1_cov_p, tmp, 2, 2); // tmp2=cond_p_1_cov_p + np.dot(Mb, np.dot(global_cov_p_1, Mb.T))
				}else{
					tmp[0][0]=res[2*p-2][2*q];
					tmp[0][1]=res[2*p-2][2*q+1];
					tmp[1][0]=res[2*p-1][2*q];
					tmp[1][1]=res[2*p-1][2*q+1]; // tmp=cov[np.ix_(select_p_1, select_q)] (=global_cov_p_1_q)
					matrix_prod(tmp2, Mb, 2, 2, tmp, 2); // tmp2=np.dot(Mb, global_cov_p_1_q)
				}
				
				// Update the auto or cross covariance matrix.
				res[2*p][2*q]=tmp2[0][0];
				res[2*p][2*q+1]=tmp2[0][1];
				res[2*p+1][2*q]=tmp2[1][0];
				res[2*p+1][2*q+1]=tmp2[1][1];

				res[2*q][2*p]=tmp2[0][0];
				res[2*q][2*p+1]=tmp2[1][0];
				res[2*q+1][2*p]=tmp2[0][1];
				res[2*q+1][2*p+1]=tmp2[1][1];
			}
		} 
	} 
	
	// Free all the 2x2 matrices needed.
	free_mem_matrix(cov_a_p, 2);
	free_mem_matrix(cov_a_p_1, 2);
	free_mem_matrix(cov_a_p_a_p_1, 2);
	free_mem_matrix(Mb, 2);
	free_mem_matrix(cond_p_1_cov_p, 2);
	free_mem_matrix(global_cov_p_1, 2);
	free_mem_matrix(tmp, 2);
	free_mem_matrix(tmp2, 2);
	return;
}	


/* ==== Compute the covariance matrix of a string derivative GP at boundary times, under a spectral mixture kernel. ==== */
/* 
	res: pointer to the matrix to be updated in place. The matrix should be of size 2*n_b_times x 2*n_b_times.
	b_times: array of boundary times, should be sorted (the function will sort it otherwise, and the matrix indices will correspond to sorted boundary times).
	n_b_times: number of boundary times. Also 1 + the number of strings.
	thetas: matrix of string hyper-parameters.
	n_mixts: array of numbers of elements in the spectral mixture for each local expert kernel.
*/
void string_sm_boundaries_deriv_cov(double **res, double *b_times, unsigned int n_b_times, double **thetas, unsigned int n_mixts){
	string_boundaries_deriv_cov(res, b_times, n_b_times, thetas, n_mixts, "sse");
}	


/* ==== Compute the covariance matrix of a string derivative GP at string times, under a given kernel. ==== */
/* 
	res: pointer to the matrix to be updated in place. The matrix should be of size 2*n_s_times x 2*n_s_times.
	n_s_times: array of string times.
	b_times: array of boundary times, should be sorted (the function will sort it otherwise, and the matrix indices will correspond to sorted boundary times).
	n_b_times: number of boundary times. Also 1 + the number of strings.
	thetas: matrix of string hyper-parameters.
	n_mixts: Number of elements in the spectral mixture for each local expert kernel.
	type: type of kernel (se, rq, ma32, ma52, sm)
*/
void string_deriv_cov(double **res, double *s_times, unsigned int n_s_times, double *b_times, unsigned int n_b_times, double **thetas, unsigned int n_mixts, char *type){
	// 0. Group together the points that belong to the same string
	// 	0.1 Sort string and boundary times.
	qsort(s_times, n_s_times, sizeof(double), cmp_double);
	qsort(b_times, n_b_times, sizeof(double), cmp_double);

	// Make sure that boundary times are unique
	int i;
	for(i=1; i<n_b_times; i++)
	{
		assert(b_times[i] != b_times[i-1]);
	}
	
	// 0.2 Do basic initialisation
	int j,counter,n_strings;
	n_strings=n_b_times-1;
	double **grouped_s_times, **bound_cov;
	int *n_elts_per_string, **grouped_s_idx;
	
	grouped_s_times=malloc(n_strings*sizeof(double *)); // Array of times belonging to each string. Should have DBL_MAX as single element if there is no time on the string.
	grouped_s_idx=malloc(n_strings*sizeof(int *)); // Array of indices of times belonging to each string. Should have INT_MAX as single element if there is no time on the string.
	n_elts_per_string=malloc(n_strings*sizeof(int)); // Number of elements in each string. Should have 0 as single element if there is no element on the string.
	for(i=0; i<n_strings; i++)
	{
		n_elts_per_string[i] = 0;
	}

	j=0;
	// First determine the number of elements per string.
	for(i=0; i<n_s_times; i++){
		while((j+1<n_strings) && (s_times[i] > b_times[j+1])){
			j += 1;
		}
		n_elts_per_string[j] = n_elts_per_string[j] + 1;
	}

	// 0.3 Allocate memory in preparation for the grouping.
	for(j=0; j<n_strings; j++){
		if(n_elts_per_string[j]>0){
			grouped_s_times[j]=malloc(n_elts_per_string[j]*sizeof(double));
			grouped_s_idx[j]=malloc(n_elts_per_string[j]*sizeof(int));
		}else{
			grouped_s_times[j]=malloc(sizeof(double));
			grouped_s_times[j][0]=DBL_MAX;
			grouped_s_idx[j]=malloc(sizeof(int));
			grouped_s_idx[j][0]=INT_MAX;	
		}
	}
	
	// 0.4 Do the actual grouping.
	counter=0;
	int k;
	for(j=0; j<n_strings; j++){
		if(n_elts_per_string[j]>0){
			for(k=0; k<n_elts_per_string[j]; k++){
				grouped_s_times[j][k]=s_times[counter];
				grouped_s_idx[j][k]=counter;
				counter+=1; 
			}
		}
	}
	
	// 1. Compute the covariance matrix at the boundary conditions.
	bound_cov = new_zeros(2*n_b_times, 2*n_b_times);
	string_boundaries_deriv_cov(bound_cov, b_times, n_b_times, thetas, n_mixts, type);
	
	// 2. Compute the global covariance matrix.
	int p,q;
	double a_p_1, a_p, a_q_1, a_q, **_unc_bound_cov_p, **inv_unc_bound_cov_p, **b_cov,
		**_unc_bound_cov_q, **inv_unc_bound_cov_q, **_unc_times_cov, **_unc_times_bound_cov,
		**cond_cov_p_q, **_unc_times_p_bound_cov, **_unc_times_q_bound_cov, **lambda_p, 
		**lambda_q, **cov_p_q, **tmp, **tmp2, *tmp3;

	tmp3=malloc(2*sizeof(double));
		
	for(p=1; p<n_b_times; p++){
		a_p_1=b_times[p-1];
		a_p=b_times[p];
		
		// Unconditional covariance matrix of the p-th String at its boundaries 
		_unc_bound_cov_p=new_zeros(4,4);
		
		tmp3[0]=a_p_1;
		tmp3[1]=a_p;
		deriv_cov(_unc_bound_cov_p, tmp3, 2, tmp3, 2, thetas[p-1], n_mixts, type);
		
		inv_unc_bound_cov_p=new_zeros(4,4);
		invert(inv_unc_bound_cov_p, _unc_bound_cov_p, 4);
		
		free_mem_matrix(_unc_bound_cov_p, 4);
		_unc_bound_cov_p=NULL;

		for(q=1; q<n_b_times; q++){
			b_cov=new_zeros(4,4);
			
			for(i=0; i<4; i++){
				for(j=0; j<4; j++){
					b_cov[i][j]=bound_cov[2*p-2+i][2*q-2+j];
				}	
			}
			
			// If there is no string time on this string just move on.
			if(n_elts_per_string[q-1] == 0) continue;
			
			a_q_1=b_times[q-1];
			a_q=b_times[q];
			
			// Unconditional covariance matrix of the q-th String at its boundaries   
			_unc_bound_cov_q=new_zeros(4,4);
			
			tmp3[0]=a_q_1;
			tmp3[1]=a_q;
			deriv_cov(_unc_bound_cov_q, tmp3, 2, tmp3, 2, thetas[q-1], n_mixts, type);
			
			inv_unc_bound_cov_q=new_zeros(4,4);
			invert(inv_unc_bound_cov_q, _unc_bound_cov_q, 4);
			
			// Cleanup _unc_bound_cov_q
			free_mem_matrix(_unc_bound_cov_q, 4);
			_unc_bound_cov_q=NULL;

			
 			if(p==q){
				// Unconditional covariance matrix of the j-th String GP at the string times times_p.
				_unc_times_cov = new_zeros(2*n_elts_per_string[p-1], 2*n_elts_per_string[p-1]);
				deriv_cov(_unc_times_cov, grouped_s_times[p-1], n_elts_per_string[p-1], grouped_s_times[p-1], n_elts_per_string[p-1], thetas[p-1], n_mixts, type);
 				
				// Unconditional cross-covariance matrix of the j-th String GP between the String times and the boundary times.
				_unc_times_bound_cov = new_zeros(2*n_elts_per_string[p-1], 4);
				
				tmp3[0]=a_p_1;
				tmp3[1]=a_p;
				deriv_cov(_unc_times_bound_cov, grouped_s_times[p-1], n_elts_per_string[p-1], tmp3, 2, thetas[p-1], n_mixts, type);
				
				
				// Covariance matrix of the values of the j-th String GP at the string times s_times, conditional on its values at the boundary conditions.
				cond_cov_p_q = new_zeros(2*n_elts_per_string[p-1], 2*n_elts_per_string[p-1]);
				
				tmp=new_zeros(4, 2*n_elts_per_string[p-1]);
				transpose_copy(tmp, _unc_times_bound_cov, 2*n_elts_per_string[p-1], 4); // tmp = _unc_times_bound_cov.T
				
				tmp2=new_zeros(4, 2*n_elts_per_string[p-1]); // Allocate memory for tmp2
				matrix_prod(tmp2, inv_unc_bound_cov_p, 4, 4, tmp, 2*n_elts_per_string[p-1]); // tmp2 = np.dot(inv_unc_bound_cov_p, _unc_times_bound_cov.T)
				
				free_mem_matrix(tmp, 4); // Clear tmp to be able to use it with different dimensions later.
				tmp=NULL;
				
				tmp=new_zeros(2*n_elts_per_string[p-1], 2*n_elts_per_string[p-1]);
				matrix_prod(tmp, _unc_times_bound_cov, 2*n_elts_per_string[p-1], 4, tmp2, 2*n_elts_per_string[p-1]); // tmp=np.dot(_unc_times_bound_cov, np.dot(inv_unc_bound_cov_p, _unc_times_bound_cov.T))
				matrix_sub(cond_cov_p_q, _unc_times_cov, tmp, 2*n_elts_per_string[p-1], 2*n_elts_per_string[p-1]); // cond_cov_p_q = _unc_times_cov - np.dot(_unc_times_bound_cov, np.dot(inv_unc_bound_cov_p, _unc_times_bound_cov.T))
				
				// Clear both tmp and tmp2 as they aren't needed any longer (for now)
				free_mem_matrix(tmp, 2*n_elts_per_string[p-1]);
				tmp=NULL;
				free_mem_matrix(tmp2, 4);
				tmp2=NULL;
				
				// Cleanup other temporary variables
				free_mem_matrix(_unc_times_cov, 2*n_elts_per_string[p-1]);
				_unc_times_cov=NULL;
				free_mem_matrix(_unc_times_bound_cov, 2*n_elts_per_string[p-1]);
				_unc_times_bound_cov=NULL;
			}else{
				cond_cov_p_q = new_zeros(2*n_elts_per_string[p-1], 2*n_elts_per_string[q-1]);
			}
			
			_unc_times_p_bound_cov = new_zeros(2*n_elts_per_string[p-1], 4);
			tmp3[0]=a_p_1;
			tmp3[1]=a_p;
			deriv_cov(_unc_times_p_bound_cov, grouped_s_times[p-1], n_elts_per_string[p-1], tmp3, 2, thetas[p-1], n_mixts, type);
			
			_unc_times_q_bound_cov = new_zeros(2*n_elts_per_string[q-1], 4);
			tmp3[0]=a_q_1;
			tmp3[1]=a_q;
			deriv_cov(_unc_times_q_bound_cov, grouped_s_times[q-1], n_elts_per_string[q-1], tmp3, 2, thetas[q-1], n_mixts, type);
			
			lambda_p = new_zeros(2*n_elts_per_string[p-1], 4);
			matrix_prod(lambda_p, _unc_times_p_bound_cov, 2*n_elts_per_string[p-1], 4, inv_unc_bound_cov_p, 4); // lambda_p = np.dot(_unc_times_p_bound_cov, inv_unc_bound_cov_p)
			
			lambda_q = new_zeros(2*n_elts_per_string[q-1], 4);
			matrix_prod(lambda_q, _unc_times_q_bound_cov, 2*n_elts_per_string[q-1], 4, inv_unc_bound_cov_q, 4); // lambda_q = np.dot(_unc_times_q_bound_cov, inv_unc_bound_cov_q)
			
			// Cleanup inv_unc_bound_cov_q
			free_mem_matrix(inv_unc_bound_cov_q, 4);
			inv_unc_bound_cov_q=NULL;
			
			cov_p_q=new_zeros(2*n_elts_per_string[p-1], 2*n_elts_per_string[q-1]);
			tmp=new_zeros(4, 2*n_elts_per_string[q-1]);
			
			transpose_copy(tmp, lambda_q, 2*n_elts_per_string[q-1], 4); // tmp=lambda_q.T;
			tmp2=new_zeros(4, 2*n_elts_per_string[q-1]);
			matrix_prod(tmp2, b_cov, 4, 4, tmp, 2*n_elts_per_string[q-1]); // tmp2=np.dot(b_cov, lambda_q.T);
			
			// Cleanup b_cov
			free_mem_matrix(b_cov, 4);
			b_cov=NULL;
			
			// Cleanup tmp
			free_mem_matrix(tmp, 4);
			tmp=NULL;
			
			tmp=new_zeros(2*n_elts_per_string[p-1], 2*n_elts_per_string[q-1]);
			matrix_prod(tmp, lambda_p, 2*n_elts_per_string[p-1], 4, tmp2, 2*n_elts_per_string[q-1]); // tmp = np.dot(lambda_p, np.dot(b_cov, lambda_q.T));
			
			// Cleanup tmp2
			free_mem_matrix(tmp2, 4);
			tmp2=NULL;
			
			// Global covariance matrix between times_p and times_q
			matrix_add(cov_p_q, cond_cov_p_q, tmp, 2*n_elts_per_string[p-1], 2*n_elts_per_string[q-1]); // cov_p_q = cond_cov_p_q + np.dot(lambda_p, np.dot(b_cov, lambda_q.T))
			
			// Cleanup tmp
			free_mem_matrix(tmp, 2*n_elts_per_string[p-1]);
			tmp=NULL;
			
			for(i=0; i<n_elts_per_string[p-1]; i++){
				for(j=0; j<n_elts_per_string[q-1]; j++){
					res[2*grouped_s_idx[p-1][i]][2*grouped_s_idx[q-1][j]] = cov_p_q[2*i][2*j];
					res[1+2*grouped_s_idx[p-1][i]][2*grouped_s_idx[q-1][j]] = cov_p_q[1+2*i][2*j];
					res[2*grouped_s_idx[p-1][i]][1+2*grouped_s_idx[q-1][j]] = cov_p_q[2*i][1+2*j];
					res[1+2*grouped_s_idx[p-1][i]][1+2*grouped_s_idx[q-1][j]] = cov_p_q[1+2*i][1+2*j];
				}
			}
			
			free_mem_matrix(cov_p_q, 2*n_elts_per_string[p-1]);
			cov_p_q=NULL;
			
			free_mem_matrix(cond_cov_p_q, 2*n_elts_per_string[p-1]);
			cond_cov_p_q=NULL;
			
			free_mem_matrix(_unc_times_p_bound_cov, 2*n_elts_per_string[p-1]);
			_unc_times_p_bound_cov=NULL;
			
			free_mem_matrix(_unc_times_q_bound_cov, 2*n_elts_per_string[q-1]);
			_unc_times_q_bound_cov=NULL;
			
			free_mem_matrix(lambda_p, 2*n_elts_per_string[p-1]);
			lambda_p=NULL;
			
			free_mem_matrix(lambda_q, 2*n_elts_per_string[q-1]);
			lambda_q=NULL;
			
		}

		free_mem_matrix(inv_unc_bound_cov_p, 4);
		inv_unc_bound_cov_p=NULL;
	}

	// Cleanup
	free(n_elts_per_string);
	for(j=0; j<n_strings; j++){
		free(grouped_s_times[j]);
		free(grouped_s_idx[j]);
	}
	free(grouped_s_times);
	grouped_s_times=NULL;
	free(grouped_s_idx);
	grouped_s_idx=NULL;
	free(tmp3);
	tmp3=NULL;
	free_mem_matrix(bound_cov, 2*n_b_times);
	bound_cov=NULL;
}


/* ==== Compute the covariance matrix of a string GP at string times, under a given expert kernel. ==== */
/* 
	res: pointer to the matrix to be updated in place. The matrix should be of size n_s_times x n_s_times.
	n_s_times: array of string times.
	b_times: array of boundary times, should be sorted (the function will sort it otherwise, and the matrix indices will correspond to sorted boundary times).
	n_b_times: number of boundary times. Also 1 + the number of strings.
	thetas: matrix of string hyper-parameters.
	n_mixts: Number of elements in the spectral mixture for each local expert kernel.
	type: type of kernel (se, rq, ma32, ma52, sm)
*/
void string_cov(double **res, double *s_times, unsigned int n_s_times, double *b_times, unsigned int n_b_times, double **thetas, unsigned int n_mixts, char *type){
	double **_deriv_cov;
	int i,j;
	
	// Compute the covariance matrix of the derivative Gaussian process in-place.
	_deriv_cov=new_zeros(2*n_s_times, 2*n_s_times);
	string_deriv_cov(_deriv_cov, s_times, n_s_times, b_times, n_b_times, thetas, n_mixts, type);
	
	for(i=0; i<n_s_times; i++){
		for(j=i; j<n_s_times; j++){
			res[i][j]=_deriv_cov[2*i][2*j];
			res[j][i]=_deriv_cov[2*j][2*i];
		}
	}
	
	free_mem_matrix(_deriv_cov, 2*n_s_times);
	_deriv_cov=NULL;
}


/* ==== Compute the covariance matrix of a string derivative GP at string times, under a spectral mixture kernel. ==== */
/* 
	res: pointer to the matrix to be updated in place. The matrix should be of size 2*n_s_times x 2*n_s_times.
	n_s_times: array of string times.
	b_times: array of boundary times, should be sorted (the function will sort it otherwise, and the matrix indices will correspond to sorted boundary times).
	n_b_times: number of boundary times. Also 1 + the number of strings.
	thetas: matrix of string hyper-parameters.
	n_mixts: Number of elements in the spectral mixture for each local expert kernel.
*/
void string_sm_deriv_cov(double **res, double *s_times, unsigned int n_s_times, double *b_times, unsigned int n_b_times, double **thetas, unsigned int n_mixts){
	string_deriv_cov(res, s_times, n_s_times, b_times, n_b_times, thetas, n_mixts, "sse");
}
