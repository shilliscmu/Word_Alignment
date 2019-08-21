package edu.berkeley.nlp.assignments.align.student;

import com.sun.javafx.scene.control.SizeLimitedList;
import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAlignerFactory;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.Arrays;
import java.util.List;
import java.util.stream.StreamSupport;

public class HmmAlignerFactory implements WordAlignerFactory
{

	public HmmWordAligner newAligner(Iterable<SentencePair> trainingData) {
		int trainingDataSize = (int) StreamSupport.stream(trainingData.spliterator(), false).count();
		System.out.println("Training data size: " + trainingDataSize);

		//indexedEnglishTrainingData[instance][pos] = indexWord
		int[][] indexedEnglishTrainingData = new int[trainingDataSize][];
		int[][] indexedFrenchTrainingData = new int[trainingDataSize][];

		StringIndexer englishWordIndex = new StringIndexer();
		StringIndexer frenchWordIndex = new StringIndexer();

		double epsilon = 0.6;

		int instance = 0;
		int longestEnglishSentenceLength = 0;
		for(SentencePair datum : trainingData) {
			List<String> englishWords = datum.getEnglishWords();
//			int[] sentenceOfIndexedEnglishTrainingData = new int[englishWords.size()];
//			for(int pos = 0; pos < englishWords.size(); pos++) {
//				sentenceOfIndexedEnglishTrainingData[pos] = englishWordIndex.addAndGetIndex(englishWords.get(pos));
//			}
//			indexedEnglishTrainingData[instance] = sentenceOfIndexedEnglishTrainingData;
			int[] sentenceOfIndexedEnglishTrainingData = new int[englishWords.size()+1];
			sentenceOfIndexedEnglishTrainingData[0] = englishWordIndex.addAndGetIndex("NULL");
			for(int pos = 1; pos <= englishWords.size(); pos++) {
				sentenceOfIndexedEnglishTrainingData[pos] = englishWordIndex.addAndGetIndex(englishWords.get(pos-1));
			}
			indexedEnglishTrainingData[instance] = sentenceOfIndexedEnglishTrainingData;

			if(sentenceOfIndexedEnglishTrainingData.length > longestEnglishSentenceLength) {
				longestEnglishSentenceLength = sentenceOfIndexedEnglishTrainingData.length;
			}

			List<String> frenchWords = datum.getFrenchWords();
			int[] sentenceOfIndexedFrenchTrainingData = new int[frenchWords.size()];
			for(int pos = 0; pos < frenchWords.size(); pos++) {
				sentenceOfIndexedFrenchTrainingData[pos] = frenchWordIndex.addAndGetIndex(frenchWords.get(pos));
			}
			indexedFrenchTrainingData[instance] = sentenceOfIndexedFrenchTrainingData;

			instance++;
		}

		int englishVocabSize = englishWordIndex.size();
		int frenchVocabSize = frenchWordIndex.size();

		//translation probs
		double uniformThetaInit = Math.log((double)1/(double)frenchVocabSize);
		double[][] theta = new double[frenchVocabSize][englishVocabSize];
		for(int frenchWord=0; frenchWord < frenchVocabSize; frenchWord++) {
			Arrays.fill(theta[frenchWord], uniformThetaInit);
		}
		//edge marginals
		//longest possible jump is from first word to last word in english sentence
		double uniformPsiInit = Math.log((double) 1 / (double) longestEnglishSentenceLength);
		double[] psi = new double[longestEnglishSentenceLength];
		Arrays.fill(psi, uniformPsiInit);
		double[][] alpha = new double[frenchVocabSize][englishVocabSize];
		double[][] beta = new double[frenchVocabSize][englishVocabSize];
		//d^t_f,e(theta); mStep numerator for theta
		double[][] jointCounts;
		//mStep denom for theta
		double[] totalJointCountsByEnglishWord;
		//mStep numerator for psi
		double[] psiCounts;
		//mStep denom for psi
		double totalPsiCounts;

		long startTime = System.currentTimeMillis();

		int iterations = 0;
		while(iterations < 10) {
			jointCounts = new double[frenchVocabSize][englishVocabSize];
			for(int frenchWord = 0; frenchWord < frenchVocabSize; frenchWord++) {
				Arrays.fill(jointCounts[frenchWord], Double.NEGATIVE_INFINITY);
			}
			totalJointCountsByEnglishWord = new double[englishVocabSize];
			Arrays.fill(totalJointCountsByEnglishWord, Double.NEGATIVE_INFINITY);
			psiCounts = new double[longestEnglishSentenceLength];
			Arrays.fill(psiCounts, Double.NEGATIVE_INFINITY);
			totalPsiCounts = Double.NEGATIVE_INFINITY;

			for (instance = 0; instance < trainingDataSize; instance++) {
				int[] englishSentenceIndexedWords = indexedEnglishTrainingData[instance];
				int[] frenchSentenceIndexedWords = indexedFrenchTrainingData[instance];

//				System.out.println("English:");
//				for(int englishWord = 0; englishWord < englishSentenceIndexedWords.length; englishWord++) {
//					System.out.print(englishWordIndex.get(englishWord) + " ");
//				}
//				System.out.println();
//
//				System.out.println("French:");
//				for(int frenchWord = 0; frenchWord < frenchSentenceIndexedWords.length; frenchWord++) {
//					System.out.print(frenchWordIndex.get(frenchWord) + " ");
//				}
//				System.out.println();

				double initialNull = Math.log(epsilon);
				double initialNotNull = Math.log((1 - epsilon) / englishSentenceIndexedWords.length);
				double toNull = Math.log(epsilon);
				double fromNull = Math.log((1-epsilon) / englishSentenceIndexedWords.length);
				double[] fromNotNullToNotNull = new double[psi.length];
				for(int jump = 0; jump < psi.length; jump++) {
					fromNotNullToNotNull[jump] = Math.log(1-epsilon) + psi[jump];
				}

				//compute alphaIJ when J=0
				for (int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
					int englishWord = englishSentenceIndexedWords[englishWordPos];
					if(englishWord == 0) {
						//add because they're in log space
						alpha[0][englishWord] = (theta[0][englishWord] + initialNull);
					} else {
						//add because they're in log space. Use transitionFromNullToNotNull because we're going from nothing to not null.
						alpha[0][englishWord] = (theta[0][englishWord] + initialNotNull);
					}
				}

				//compute alphaIJs when J>0
				for (int frenchWordPos = 1; frenchWordPos < frenchSentenceIndexedWords.length; frenchWordPos++) {
					int frenchWord = frenchSentenceIndexedWords[frenchWordPos];
					int frenchWordMinus1 = frenchSentenceIndexedWords[(frenchWordPos-1)];
					for (int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
						int englishWord = englishSentenceIndexedWords[englishWordPos];
						double alphaIJ = Double.NEGATIVE_INFINITY;
						for(int englishWordPosPrime = 0; englishWordPosPrime < englishSentenceIndexedWords.length; englishWordPosPrime++) {
							int englishWordPrime = englishSentenceIndexedWords[englishWordPosPrime];
							int jump = Math.abs(englishWordPos - englishWordPosPrime);
							if (englishWord == 0) {
								//to null
								alphaIJ = SloppyMath.logAdd(alphaIJ, (alpha[frenchWordMinus1][englishWordPrime] + toNull));
//								alphaIJ += (alpha[frenchWordMinus1][englishWordPrime] + toNull);
							} else if (englishWordPrime == 0) {
								//from null
								alphaIJ = SloppyMath.logAdd(alphaIJ, (alpha[frenchWordMinus1][englishWordPrime] + fromNull));
//								alphaIJ += (alpha[frenchWordMinus1][englishWordPrime] + fromNull);
							} else {
								alphaIJ = SloppyMath.logAdd(alphaIJ, (alpha[frenchWordMinus1][englishWordPrime] + fromNotNullToNotNull[jump]));
//								alphaIJ += (alpha[frenchWordMinus1][englishWordPrime] + fromNotNullToNotNull[jump]);
							}
						}
						alphaIJ += theta[frenchWord][englishWord];
						alpha[frenchWord][englishWord] = alphaIJ;
					}
				}

//				System.out.println("Alpha: ");
//				for(int frenchWord = 0; frenchWord < frenchVocabSize; frenchWord++) {
//					if(alpha[frenchWord][0] !=0) {
//						System.out.print(frenchWordIndex.get(frenchWord) + ": ");
//					}
//					for(int englishWord = 0; englishWord < alpha[frenchWord].length; englishWord++) {
//						if(alpha[frenchWord][englishWord] != 0) {
//							System.out.print(englishWordIndex.get(englishWord) + " -> " + alpha[frenchWord][englishWord] + ", ");
//						}
//					}
//					System.out.println();
//				}

				//compute betaIJ when J=frenchSentenceIndexedWords.length-1
				for(int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
					int englishWord = englishSentenceIndexedWords[englishWordPos];
					int lastFrenchWord = frenchSentenceIndexedWords[frenchSentenceIndexedWords.length-1];
					//Math.log(1) = 0;
					beta[lastFrenchWord][englishWord] = 0.0;
				}

				//compute betaIJ when J<frenchSentenceIndexedWords.length-1
				for (int frenchWordPos = frenchSentenceIndexedWords.length - 2; frenchWordPos >= 0; frenchWordPos--) {
					int frenchWord = frenchSentenceIndexedWords[frenchWordPos];
					int frenchWordPlus1 = frenchSentenceIndexedWords[(frenchWordPos + 1)];
					for (int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
						int englishWord = englishSentenceIndexedWords[englishWordPos];
						//init -inf because of logAdd
						double betaIJ = Double.NEGATIVE_INFINITY;
						//here, englishWordPrime is the word the next french word is aligned to.
						for(int englishWordPosPrime = 0; englishWordPosPrime < englishSentenceIndexedWords.length; englishWordPosPrime++) {
							int englishWordPrime = englishSentenceIndexedWords[englishWordPosPrime];
							if(englishWord == 0) {
								//from null
								betaIJ = SloppyMath.logAdd(betaIJ, (theta[frenchWordPlus1][englishWordPrime] + fromNull + beta[frenchWordPlus1][englishWordPrime]));
//								beta[frenchWord][englishWord] += (theta[frenchWordPlus1][englishWordPrime] + fromNull + beta[frenchWordPlus1][englishWordPrime]);
							} else if (englishWordPrime == 0) {
								//to null
								betaIJ = SloppyMath.logAdd(betaIJ, (theta[frenchWordPlus1][englishWordPrime] + toNull + beta[frenchWordPlus1][englishWordPrime]));
//								beta[frenchWord][englishWord] += (theta[frenchWordPlus1][englishWordPrime] + toNull + beta[frenchWordPlus1][englishWordPrime]);
							} else {
								int jump = Math.abs(englishWordPos - englishWordPosPrime);
								betaIJ = SloppyMath.logAdd(betaIJ, (theta[frenchWordPlus1][englishWordPrime] + fromNotNullToNotNull[jump] + beta[frenchWordPlus1][englishWordPrime]));
//								beta[frenchWord][englishWord] += (theta[frenchWordPlus1][englishWordPrime] + fromNotNullToNotNull[jump] + beta[frenchWordPlus1][englishWordPrime]);
							}
						}
						beta[frenchWord][englishWord] = betaIJ;
					}
				}
//				System.out.println("Beta: ");
//				for(int frenchWord = 0; frenchWord < frenchVocabSize; frenchWord++) {
//					System.out.print(frenchWordIndex.get(frenchWord) + ": ");
//					for(double entry : beta[frenchWord]) {
//						System.out.print(entry + ", ");
//					}
//					System.out.println();
//				}

				double[] z = new double[frenchSentenceIndexedWords.length];
				for(int frenchWordPos = 0; frenchWordPos < frenchSentenceIndexedWords.length; frenchWordPos++) {
					int frenchWord = frenchSentenceIndexedWords[frenchWordPos];
					double zFrenchWord = Double.NEGATIVE_INFINITY;
					for(int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
						int englishWord = englishSentenceIndexedWords[englishWordPos];
						zFrenchWord = SloppyMath.logAdd(zFrenchWord, (alpha[frenchWord][englishWord] + beta[frenchWord][englishWord]));
					}
					z[frenchWordPos] = zFrenchWord;
				}

				//eStep: compute expected counts for theta
				for (int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
					int englishWord = englishSentenceIndexedWords[englishWordPos];
					for (int frenchWordPos = 0; frenchWordPos < frenchSentenceIndexedWords.length; frenchWordPos++) {
						int frenchWord = frenchSentenceIndexedWords[frenchWordPos];
						double alphaIJ = alpha[frenchWord][englishWord];
						double betaIJ = beta[frenchWord][englishWord];
						//add because they're in log space
						double update = (alphaIJ + betaIJ - z[frenchWordPos]);
						jointCounts[frenchWord][englishWord] = SloppyMath.logAdd(jointCounts[frenchWord][englishWord], update);
						totalJointCountsByEnglishWord[englishWord] = SloppyMath.logAdd(totalJointCountsByEnglishWord[englishWord], update);
//						totalJointCountsByEnglishWord[englishWord] += update;
					}
				}

				//eStep: compute expected counts for psi
				for (int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
					int englishWord = englishSentenceIndexedWords[englishWordPos];
					for (int englishWordPrimePos = 0; englishWordPrimePos < englishSentenceIndexedWords.length; englishWordPrimePos++) {
						int englishWordPrime = englishSentenceIndexedWords[englishWordPrimePos];
						int jump = Math.abs(englishWordPos - englishWordPrimePos);
						double psiJump = psi[jump];
						//frenchWordPos = 1 because we need j-1
						for (int frenchWordPos = 1; frenchWordPos < frenchSentenceIndexedWords.length; frenchWordPos++) {
							int frenchWord = frenchSentenceIndexedWords[frenchWordPos];
							int frenchWordMinus1 = frenchSentenceIndexedWords[frenchWordPos-1];
							double alphaIprimeJminus1 = alpha[frenchWordMinus1][englishWordPrime];
							double betaIJ = beta[frenchWord][englishWord];
							double thetaIJ = theta[frenchWord][englishWord];
							//add because they're in log space
							double update = (alphaIprimeJminus1 + betaIJ + psiJump + thetaIJ - z[frenchWordPos]);
//							psiCounts[jump] += update;
							psiCounts[jump] = SloppyMath.logAdd(psiCounts[jump], update);
//							totalPsiCounts += update;
							totalPsiCounts = SloppyMath.logAdd(totalPsiCounts, update);
						}
					}
				}

//				if ((instance + 1) % 10000 == 0) {
//					System.out.println("We have learned from " + (instance + 1) + " sentence pairs so far this iteration.");
//				}
			}

			//mStep: reestimate theta
			for (int englishWord = 0; englishWord < englishVocabSize; englishWord++) {
				double totalJointCountByEnglishWord = totalJointCountsByEnglishWord[englishWord];
				for (int frenchWord = 0; frenchWord < frenchVocabSize; frenchWord++) {
					if(jointCounts[frenchWord][englishWord] != Double.NEGATIVE_INFINITY) {
						//subtracting because we're in log space
						double updatedTheta = (jointCounts[frenchWord][englishWord] - totalJointCountByEnglishWord);
						theta[frenchWord][englishWord] = updatedTheta;
					}
				}
			}

			//mStep: reestimate psi
			for (int jump = 0; jump < longestEnglishSentenceLength; jump++) {
				if(psiCounts[jump] != Double.NEGATIVE_INFINITY) {
					psi[jump] = (psiCounts[jump] - totalPsiCounts);
				}
			}

			iterations++;
//			if (iterations % 1 == 0) {
				System.out.println("We have completed " + (iterations) + " iterations of EM. ");
//			}

		}
//		System.out.println("Thetas:");
//		for(int frenchWord = 0; frenchWord < theta.length; frenchWord++) {
//			System.out.println(frenchWordIndex.get(frenchWord) + ": ");
//			for(int englishWord = 0; englishWord < theta[frenchWord].length; englishWord++) {
//				System.out.print(englishWordIndex.get(englishWord) + "= " + theta[frenchWord][englishWord] + "; ");
//			}
//			System.out.println('\n');
//		}
		System.out.println("It took " + iterations + " iterations and " + ((System.currentTimeMillis() - startTime)/1000) + " seconds to converge.");
		return new HmmWordAligner(theta, psi, epsilon, englishWordIndex, frenchWordIndex);
	}
}
