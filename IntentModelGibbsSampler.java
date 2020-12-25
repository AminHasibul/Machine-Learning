/*
 * (C) Copyright 2011, Arjun Mukherjee 
 */

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.io.*;
import java.util.*;

public class IntentModelGibbsSampler {

	/**
	 * document data (term lists)
	 */
	int[][] documents;

	/**
	 * vocabulary size
	 */
	int V;
	
	HashMap<Integer, String> V_Map;
	/**
	 * number of topics
	 */
	int T;

	/**
	 * number of intents
	 */
	int I;
    
		/**
	 * Dirichlet parameter (document--topic associations)
	 */
	double alpha;
	double alpha_i; // Dirichlet prior for theta_i (per document distribution of intents)

	/**
	 * Dirichlet parameter (topic--term associations)
	 */
	double beta;
	double beta_i; //  Dirichlet prior for phi_i (per intent distribution of words)

	double gamma_a, gamma_b; // Asymmetric Beta priors
	/**
	 * topic assignments for each word.
	 */
	int z[][];
	boolean r[][]; // indicator variable r for each word assignment. Takes true for topics and false for intents
	/**
	 * cwt[i][j] number of instances of word i (term?) assigned to topic j.
	 */
	int[][] nw;
	int [][] nw_i; // no. of times word w is assigned to intent i

	/**
	 * na[i][j] number of words in document i assigned to topic j.
	 */
	int[][] nd;
	int [][] nd_i; // no. of words in doc d assigned to intent i

	/**
	 * nwsum[j] total number of words assigned to topic j.
	 */
	int[] nwsum;
	int [] nwsum_i; // total no. of words assigned to intent i

	/**
	 * ndsum[i] total number of words in document d assigned to topics.
	 */
	int[] ndsum;
	int [] ndsum_i; // total no. of words in doc d assigned to intents

	/**
	 * cumulative statistics of theta
	 */
	double[][] thetasum;
	double [][] thetasum_i; // cumulative stats for theta_i

	/**
	 * cumulative statistics of phi
	 */
	double[][] phisum;
	double [][] phisum_i; // cumulative stats for phi_i
	/**
	 * size of statistics
	 */
	int numstats;

	/**
	 * sampling lag (?)
	 */
	private static int THIN_INTERVAL = 20;

	/**
	 * burn-in period
	 */
	private static int BURN_IN = 100;

	/**
	 * max iterations
	 */
	private static int ITERATIONS = 1000;

	/**
	 * sample lag (if -1 only one sample taken)
	 */
	private static int SAMPLE_LAG;

	private static int dispcol = 0;

	/**
	 * Initialise the Gibbs sampler with data.
	 * 
	 * @param V
	 *            vocabulary size
	 * @param data
	 */
	public IntentModelGibbsSampler(int[][] documents, int V) {

		this.documents = documents;
		this.V = V;
	}

	/**
	 * Initialisation: Must start with an assignment of observations to topics ?
	 * Many alternatives are possible, I chose to perform random assignments
	 * with equal probabilities
	 * 
	 * @param T
	 *            number of topics
	 * @param I
	 *            number of intents
	 * @return z, assignments of words to topics and intents
	 */
	public void initialState(int T, int I) {

		int M = documents.length;

		// initialise count variables.
		nw = new int[V][T];
		nw_i = new int[V][I];
		nd = new int[M][T];
		nd_i = new int[M][I];
		nwsum = new int[T];
		nwsum_i = new int[I];
		ndsum = new int[M];
		ndsum_i = new int[M];

		/*
		 The z_i are are initialised to values in [0,T-1] for topics and
		-[1, I] for intents to determine the initial state of the Markov chain.
		for each word I first choose to assign an intent or a topic (with a prior
		of 0.3 for intents as intent expressions generally constitute about 30%. This
		is my prior belief of intent distribution to intitialize the markov state. Clearly,
		other beliefs are also possible). Then randomly allocate one of the topics/intents
		from the set of all topics/intents
		 */
		z = new int[M][];
		r = new boolean [M][];
		for (int m = 0; m < M; m++) {
			int N = documents[m].length;
			z[m] = new int[N];
			r[m] = new boolean[N];
			for (int n = 0; n < N; n++)
			{
				r[m][n] = (Math.random()<0.19)?false:true; // 0.19 = E(1-V), V~Beta(gamma) is the success prob of choosing a topic
				if(r[m][n]){ // z_m,n is a topic
					int topic = (int) (Math.random() * T);
					z[m][n] = topic;
					// number of instances of word v assigned to topic t
					nw[documents[m][n]][topic]++;
					// number of words in document d assigned to topic t.
					nd[m][topic]++;
					// total number of words assigned to topic t.
					nwsum[topic]++;
					// total number of words in document d assigned to topics
					ndsum[m]++;	
					 
				
					
				}
				else{ // z_m,n is an intent
					int intent = (int) (Math.random() * I);
					z[m][n] = -(intent+1);
					//System.out.println(z[m][n]);
					// number of instances of word v assigned to intent i
					nw_i[documents[m][n]][intent]++;
					//System.out.println("new docs"+ nw_i[documents[m][n]][intent]);
					// number of words in document d assigned to intent i.
					nd_i[m][intent]++;
					// total number of words assigned to intent i.
					nwsum_i[intent]++;
					// total number of words in document d assigned to intents
					ndsum_i[m]++;
					
				}
			}
		}
	}

	/**
	 * Main method: Select initial state ? Repeat a large number of times: 1.
	 * Select an element 2. Update conditional on other elements. If
	 * appropriate, output summary for each run.
	 * 
	 * @param T
	 *            number of topics
	 * @param I
	 * 			  number of intents
	 * @param alpha
	 *            symmetric prior parameter on document--topic associations
	 * @param alpha_i
	 *            symmetric prior parameter on document--intent associations
	 * @param beta
	 *            symmetric prior parameter on topic--term associations
	 * @param beta_i
	 *            symmetric prior parameter on intent--term associations
	 * @param gamma_a
	 *            asymmetric prior for per document topic-intent distribution
	 * @param gamma_b
	 *            asymmetric prior for per document topic-intent distribution.
	 */
	private void gibbs(int T, int I, double alpha, double alpha_i, double beta, double beta_i, double gamma_a, double gamma_b) {
		this.T = T;
		this.I = I;
		this.alpha = alpha;
		this.alpha_i = alpha_i;
		this.beta = beta;
		this.beta_i = beta_i;
		this.gamma_a = gamma_a;
		this.gamma_b = gamma_b;

		// init sampler statistics
		if (SAMPLE_LAG > 0) {
			thetasum = new double[documents.length][T];
			thetasum_i = new double[documents.length][I];
			phisum = new double[T][V];
			phisum_i = new double [I][V];
			numstats = 0;
		}

		// initial state of the Markov chain:
		initialState(T, I);

		System.out.println("Sampling " + ITERATIONS
				+ " iterations with burn-in of " + BURN_IN + " (B/S="
				+ THIN_INTERVAL + ").");

		for (int i = 0; i < ITERATIONS; i++) 
		{

			// for all z_i
			for (int m = 0; m < z.length; m++) {
				for (int n = 0; n < z[m].length; n++) {

					// (z_i = z[m][n])
					int topic;
					topic = sampleFullConditional(m, n); // returns the sampled topic [0, T-1] or intent -[1, I]
					z[m][n] = topic;
					r[m][n] = (topic>=0)?true:false; // also assign the sampled indicator variable r_m,n
				}
			}


			if(i==BURN_IN)
				
				System.out.println("\nBURN_IN Time: " + new Date().toString());

			if ((i < BURN_IN) && (i % THIN_INTERVAL == 0)) {
				System.out.print("B");
				System.out.println("Iteration: " + i + ", Likelihood: " + getLogLiklihood());
				
				printCurrentPhi(i);
				dispcol++;
			}
			// display progress
			if ((i > BURN_IN) && (i % THIN_INTERVAL == 0)) {
				System.out.print("S");
				System.out.println("Iteration: " + i + ", Likelihood: " + getLogLiklihood());
				
				
				dispcol++;
			}
			// get statistics after burn-in
			if ((i > BURN_IN) && (SAMPLE_LAG > 0) && (i % SAMPLE_LAG == 0)) {
				updateParams();
				System.out.print("|");
				
				printCurrentPhi(i);
				System.out.println("Iteration: " + i + ", Likelihood: " + getLogLiklihood());
				
				if (i % THIN_INTERVAL != 0)
					dispcol++;
			}
			if (dispcol >= 100) {
				System.out.println();
				dispcol = 0;
			}
		}
	}
	
	
	/**
	 * Compute the current log-liklihood
	 * P(W|Z) = Sum_k=1_K( log( B(beta + n_k)/ B(beta) ) )
	 * @return
	 * log-liklihood
	 */
	private double getLogLiklihood(){
		//System.out.println("Computing liklihood");
		double log_liklihood = 0;
		double log_liklihood_i = 0;
		double log_liklihood_t = 0;
		double [] beta_v = new double[V]; 
		double [] beta_v_i = new double[V]; //// beta_1 to V which is symmetric
		double [][] beta_n_k = new double [T][V]; // beta + n_k
		double [][] beta_n_k_i = new double [I][V];
		
		// beta + n_k
		for(int k=0; k<T; k++)
			for(int v=0; v<V; v++)
				beta_n_k[k][v] = beta + nw[v][k];
		
		for(int k=0; k<I; k++)
			for(int v=0; v<V; v++)
				beta_n_k_i[k][v] = beta_i + nw_i[v][k];
		
		for(int v=0; v<V; v++)
			beta_v[v] = beta;
		
		for(int v=0; v<V; v++)
			beta_v_i[v] = beta_i;
		
		
		for(int k=0; k<T; k++){
			//System.out.println("beta_n_k : " + ArrayToString(beta_n_k[k]));
			double log_B_beta_n_k = GammaUtils.ldelta(beta_n_k[k]);
			//double log_B_beta_n_k_i = GammaUtils.ldelta(beta_n_k_i[k]);
			//System.out.println("B(beta_n_k): " + B_beta_n_k);
			double log_B_beta = GammaUtils.ldelta(beta_v);
			//double log_B_beta_i = GammaUtils.ldelta(beta_v_i);
			//System.out.println("B(beta): " + B_beta);
			double val = (log_B_beta_n_k)-(log_B_beta);
//			double val = (log_B_beta_n_k+log_B_beta_n_k_i)-(log_B_beta+log_B_beta_i);
			//System.out.println("liklihood comp sum: " + val);
			log_liklihood_t += val;
		}
		for(int k=0; k<I; k++){
			//System.out.println("beta_n_k : " + ArrayToString(beta_n_k[k]));
			//double log_B_beta_n_k = GammaUtils.ldelta(beta_n_k[k]);
			double log_B_beta_n_k_i = GammaUtils.ldelta(beta_n_k_i[k]);
			//System.out.println("B(beta_n_k): " + B_beta_n_k);
			//double log_B_beta = GammaUtils.ldelta(beta_v);
			double log_B_beta_i = GammaUtils.ldelta(beta_v_i);
			//System.out.println("B(beta): " + B_beta);
			double val = (log_B_beta_n_k_i)-(log_B_beta_i);
//			
			//System.out.println("liklihood comp sum: " + val);
			log_liklihood_i += val;
		}
		log_liklihood = log_liklihood_i + log_liklihood_t; 
		return log_liklihood;
	}
	/**
	 * Sample based on the Gibbs Updating Rule.
	 * @param m
	 *            document
	 * @param n
	 *            word
	 */
	private int sampleFullConditional(int m, int n) {

		
		if(r[m][n] == true)
		{
			int topic = z[m][n];
			nw[documents[m][n]][topic]--;
			nd[m][topic]--;
			nwsum[topic]--;
			ndsum[m]--;

			double[] p = new double[T];
			int [] labels = new int[T];
			for (int t = 0; t < T; t++) {
				p[t] = (nw[documents[m][n]][t] + beta) / (nwsum[t] + V * beta)
				* (nd[m][t] + alpha) / (ndsum[m] + T * alpha);
				labels[t] = t;

			}

			for (int t = 1; t < p.length; t++) {
				p[t] += p[t - 1];
			}

			
			double u = Math.random() * p[T - 1];
			for (topic = 0; topic< p.length; topic++) {
				if (u < p[topic])
					break;
			}

		
			nw[documents[m][n]][topic]++;
			nd[m][topic]++;
			nwsum[topic]++;
			ndsum[m]++;

			return topic;
		}
		else
		{
			int intent = -(z[m][n]);
			intent = intent -1;
		
			nw_i[documents[m][n]][intent]--;
			nd_i[m][intent]--;
			nwsum_i[intent]--;
			ndsum_i[m]--;
           
			
			double[] p = new double[I];
			int [] labels = new int[I];
			for (int i = 0; i < I; i++) {
				p[i] = (nw_i[documents[m][n]][i] + beta_i) / (nwsum_i[i] + V * beta_i)
				* (nd_i[m][i] + alpha_i) / (ndsum_i[m] + I * alpha_i);
				labels[i] = i;

			}

		
			for (int k = 1; k < p.length; k++) {
				p[k] += p[k - 1];
			}

			
			double u = Math.random() * p[I - 1];
			for (intent = 0; intent < p.length; intent++) {
				if (u < p[intent])
					break;
			}

			
			nw_i[documents[m][n]][intent]++;
			nd_i[m][intent]++;
			nwsum_i[intent]++;
			ndsum_i[m]++;
            intent = intent+1;
			return -intent;
		}
	}

	/**
	 * Add to the statistics the values of theta, phi, theta_i and phi_i  for the current state.
	 */
	private void updateParams() {
		// topics
		for (int m = 0; m < documents.length; m++) 
		{
			for (int t = 0; t < T; t++) 
			{
				thetasum[m][t] += (nd[m][t] + alpha) / (ndsum[m] + T * alpha);
			}
			
			for (int i = 0; i < I; i++) 
			{
				thetasum_i[m][i] += (nd_i[m][i] + alpha_i) / (ndsum_i[m] + I * alpha_i);
			}
			
		}
		
		for (int t = 0; t < T; t++) 
		{
			for (int w = 0; w < V; w++) {
				phisum[t][w] += (nw[w][t] + beta) / (nwsum[t] + V * beta);
			}
		}
		
//		// intention
//		for (int m = 0; m < documents.length; m++)
//		{
//			for (int i = 0; i < I; i++) 
//			{
//				thetasum_i[m][i] += (nd_i[m][i] + alpha_i) / (ndsum_i[m] + I * alpha_i);
//			}
//		}
		
		for (int i = 0; i < I; i++)
		{
			for (int w = 0; w < V; w++)
			{
				phisum_i[i][w] += (nw_i[w][i] + beta_i) / (nwsum_i[i] + V * beta_i);
			}
		}
		numstats++;
		// To-Do
	}

	/**
	 * Retrieve estimated document--topic associations. If sample lag > 0 then
	 * the mean value of all sampled statistics for theta[][] is taken.
	 * 
	 * @return theta multinomial mixture of document topics (M x K)
	 */
	public double[][] getTheta() {
		double[][] theta = new double[documents.length][T];
		
		if (SAMPLE_LAG > 0) {
			for (int m = 0; m < documents.length; m++)
			{
				for (int k = 0; k < T; k++) 
				{
					theta[m][k] = thetasum[m][k] / numstats;
				}
			}

		} else {
			for (int m = 0; m < documents.length; m++) {
				for (int k = 0; k < T; k++) {
					theta[m][k] = (nd[m][k] + alpha) / (ndsum[m] + T * alpha);
				}
			}
		}
		
		

		return theta;
	}




	/**
	 * Retrieve estimated topic--word associations. If sample lag > 0 then the
	 * mean value of all sampled statistics for phi[][] is taken.
	 * 
	 * @return phi multinomial mixture of topic words (K x V)
	 */
	public double[][] getPhi() {
		double[][] phi = new double[T][V];
		if (SAMPLE_LAG > 0) {
			for (int t = 0; t < T; t++) {
				for (int w = 0; w < V; w++) {
					phi[t][w] = phisum[t][w] / numstats;
				}
			}
		} else {
			for (int t = 0; t < T; t++) {
				for (int w = 0; w < V; w++) {
					phi[t][w] = (nw[w][t] + beta) / (nwsum[t] + V * beta);
				}
			}
		}
		
		// To-Do
		return phi;
	}
	
	public double[][] getPhi_i() {
		double[][] phi_i = new double[I][V];
		if (SAMPLE_LAG > 0) {
			for (int i = 0; i < I; i++) {
				for (int w = 0; w < V; w++) {
					phi_i[i][w] = phisum_i[i][w] / numstats;
				}
			}
		} else {
			for (int i = 0; i < I; i++) {
				for (int w = 0; w < V; w++) {
					phi_i[i][w] = (nw_i[w][i] + beta_i) / (nwsum_i[i] + V * beta_i);
				}
			}
		}
		return phi_i;
	}

	public void printCurrentPhi(int i){
		double [][] phi = new double [T][V];
		double [][] phi_i = new double [I][V];
		double [][] combined_phi = new double [T+I][];
		//compute phi (per topic word distribution)
		if(i<BURN_IN)
			for (int k = 0; k < T; k++) {
				for (int w = 0; w < V; w++) {
					phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
				}
			}
		else
			for (int k = 0; k < T; k++) {
				for (int w = 0; w < V; w++) {
					phi[k][w] = phisum[k][w] / numstats;
				}
			}
		
		//compute phi_i (per intent word distribution)
		if(i<BURN_IN)
			for (int k = 0; k < I; k++) {
				for (int w = 0; w < V; w++) {
					phi_i[k][w] = (nw_i[w][k] + beta_i) / (nwsum_i[k] + V * beta_i);
				}
			}
		else
			for (int k = 0; k < I; k++) {
				for (int w = 0; w < V; w++) {
					phi_i[k][w] = phisum_i[k][w] / numstats;
				}
			}
		
		// combine the topic and intent distributions
		for(int t=0; t<T; t++)
			combined_phi[t] = phi[t];
		for(int k=0; k<I; k++)
			combined_phi[T+k] = phi_i[k];
		
		System.out.println("\nPrinting Topic Distribution from print f\n");
		for(int j=0; j<V; j++)
			System.out.printf("%4.1f %c",(float)j, ' ' );
		System.out.println();
		for(int k=0; k<T+I; k++){
			System.out.println();
			for(int j=0; j<V; j++)
				System.out.printf("%4.1f %c",combined_phi[k][j], ' ' );
			System.out.println();
		}
		 
		new GibbsVisualizer().saveDist(Integer.toString(i), combined_phi);
		
	}

	/**
	 * Configure the gibbs sampler
	 * 
	 * @param iterations
	 *            number of total iterations
	 * @param burnIn
	 *            number of burn-in iterations
	 * @param thinInterval
	 *            update statistics interval
	 * @param sampleLag
	 *            sample interval (-1 for just one sample at the end)
	 */
	public void configure(int iterations, int burnIn, int thinInterval,
			int sampleLag) {
		ITERATIONS = iterations;
		BURN_IN = burnIn;
		THIN_INTERVAL = thinInterval;
		SAMPLE_LAG = sampleLag;
	}

	/**
	 * Driver with example data.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {

		System.out.println("\nStart Time: " + new Date().toString());
		// load the documents
		//String path_prefix = "";
		int[][] documents = new IntentModelSyntheticData().getDocuments();
		// vocabulary
		//System.out.println("Jointly Modeling Topics "+ documents);
		int V = 25;
		// # topics, intents
		int T = 7, I = 3;
		
		
		double alpha = 50.0/T, alpha_i = 50.0/I;
		double beta = 0.1, beta_i = 0.1;
		double gamma_a = 4, gamma_b = 1.0; // my prior belief of the distribution of topics and intents in docs
		// good values
		//alpha = 2.0; beta = .5;
		// Bootstrap
		System.out.println("Jointly Modeling Topics and Intentions using Beta Switch Samplers.");
		double[][] theta;
		
		double[][] phi = new double[T][V];
		double [][] phi_i = new double[I][V];
		
		// Learn the model form data
		IntentModelGibbsSampler model = new IntentModelGibbsSampler(documents, V);
		model.configure(2000, 25, 50, 10);
		model.gibbs(T, I, alpha, alpha_i, beta, beta_i, gamma_a, gamma_b);
		// Obtain the posterior estimates
		theta = model.getTheta();
		
		phi = model.getPhi();
		phi_i = model.getPhi_i();
		UseSerialization.doSave(phi, "phi.ser");
		UseSerialization.doSave(phi_i, "phi_i.ser");
		System.out.println("\nFinish Time: " + new Date().toString());
		System.out.println();
		

		
		// Code snippets to print the estimated latent variable dist. (To ensure in value and not just in image)
		
//		System.out.println("Printing Topic Distribution\n");
//		for(int j=0; j<V; j++)
//			System.out.printf("%4.1f %c",(float)j, ' ' );
//		System.out.println();
//		for(int i=0; i<T; i++){
//			System.out.println();
//			for(int j=0; j<V; j++)
//				System.out.printf("%4.1f %c",phi[i][j], ' ' );
//		}
	    
		
//		phi = (double [][])UseSerialization.doLoad("phi.ser");
//		for(int i=0; i<T; i++){
//			Phi_k_w [] phi_j = new Phi_k_w [V];
//			for(int l=0; l<V; l++)
//				phi_j[l] = new Phi_k_w(l, phi[i][l]);
//			Arrays.sort(phi_j);
//
//			System.out.println();
//			System.out.print("Topic " + i+1 + ". ");
//			for(int j=0; j<25; j++)
//				System.out.print(model.V_Map.get(phi_j[j].w) + "  " );
//		}
//		for(int i=0; i<I; i++){
//			Phi_k_w [] phi_k = new Phi_k_w [V];
//			for(int l=0; l<V; l++)
//				phi_k[l] = new Phi_k_w(l, phi[i][l]);
//			Arrays.sort(phi_k);
//
//			System.out.println();
//			System.out.print("intention " + i+1 + ". ");
//			for(int j=0; j<25; j++)
//				System.out.print(model.V_Map.get(phi_k[j].w) + "  " );
//		}
		
	}


}