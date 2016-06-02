import org.jblas.Decompose;
import org.jblas.DoubleMatrix;
import org.jblas.Singular;

public class RandomizedSVD {
	// Compute a (truncated) randomized SVD of a JBLAS DoubleMatrix
	public int numComponents = 2;
	public int niters = 5;

	private int numOversamples = 10;
	private boolean transpose = false;
	public DoubleMatrix[] rsvd = new DoubleMatrix[3];

	public RandomizedSVD(int numComponents, int niters) {
		this.numComponents = numComponents;
		this.niters = niters;
	}

	public void fit(DoubleMatrix A) {
		transpose = A.rows > A.columns;
		rsvd[0] = new DoubleMatrix(A.rows, numComponents);
		rsvd[1] = new DoubleMatrix(numComponents);
		rsvd[2] = new DoubleMatrix(A.columns, numComponents);
		if (transpose) {
			A = A.transpose();
		}

		DoubleMatrix C = A.mmul(A.transpose());
		DoubleMatrix Q = DoubleMatrix.randn(A.rows, Math.min(A.rows, numComponents + numOversamples));
		for (int i = 0; i < niters; i++) {
			C.mmuli(Q, Q);
			Q = Decompose.lu(Q).l;
		}

		C.mmuli(Q, Q);
		Q = Decompose.qr(Q).q;
		DoubleMatrix[] svd = Singular.fullSVD(Q.transpose().mmul(A));
		DoubleMatrix W = Q.mmul(svd[0]);

		if (transpose) {
			for (int i = 0; i < numComponents; i++) {
				rsvd[0].putColumn(i, svd[2].getColumn(i));
				rsvd[1].put(i, svd[1].get(i));
				rsvd[2].putColumn(i, W.getColumn(i));
			}
		} else {
			for (int i = 0; i < numComponents; i++) {
				rsvd[0].putColumn(i, W.getColumn(i));
				rsvd[1].put(i, svd[1].get(i));
				rsvd[2].putColumn(i, svd[2].getColumn(i));
			}
		}
	}
}
