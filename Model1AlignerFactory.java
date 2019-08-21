package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAlignerFactory;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.Arrays;
import java.util.List;
import java.util.stream.StreamSupport;

public class Model1AlignerFactory implements WordAlignerFactory
{

	public Model1WordAligner newAligner(Iterable<SentencePair> trainingData) {
		int trainingDataSize = (int) StreamSupport.stream(trainingData.spliterator(), false).count();
		System.out.println("Training data size: " + trainingDataSize);
		
		//indexedEnglishTrainingData[instance][pos] = indexWord
		int[][] indexedEnglishTrainingData = new int[trainingDataSize][];
		int[][] indexedFrenchTrainingData = new int[trainingDataSize][];

		StringIndexer englishWordIndex = new StringIndexer();
		StringIndexer frenchWordIndex = new StringIndexer();

		int instance = 0;
		for(SentencePair datum : trainingData) {
			List<String> englishWords = datum.getEnglishWords();

			int[] sentenceOfIndexedEnglishTrainingData = new int[englishWords.size()+1];
			sentenceOfIndexedEnglishTrainingData[0] = englishWordIndex.addAndGetIndex("NULL");
			for(int pos = 1; pos <= englishWords.size(); pos++) {
				sentenceOfIndexedEnglishTrainingData[pos] = englishWordIndex.addAndGetIndex(englishWords.get(pos-1));
			}
			indexedEnglishTrainingData[instance] = sentenceOfIndexedEnglishTrainingData;

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

		//t: translation probabilities. theta[frenchWord][englishWord].
		double uniformThetaInit = Math.log((double)1/(double)frenchVocabSize);
		double[][] theta = new double[frenchVocabSize][englishVocabSize];
		for(int frenchWord=0; frenchWord < frenchVocabSize; frenchWord++) {
			Arrays.fill(theta[frenchWord], uniformThetaInit);
		}

		//mStepNumers. jointCounts[frenchWord][englishWord]
		double[][] jointCounts;
		//mStepDenoms
		double[] totalJointCountsByEnglishWord;
		//each entry is the sum of all (theta[f][e] + prior)'s for a particular f
		double[] eStepDenoms;

		System.out.println("The size of the english vocab is " + englishVocabSize);
		System.out.println("The size of the french vocab is " + frenchVocabSize);

		int iterations = 0;
		while(iterations < 10) {
			//initialize jointCounts to neg inf for each iteration
			jointCounts = new double[frenchVocabSize][englishVocabSize];
			for(int frenchWord = 0; frenchWord < frenchVocabSize; frenchWord++) {
				Arrays.fill(jointCounts[frenchWord], Double.NEGATIVE_INFINITY);
			}
			totalJointCountsByEnglishWord = new double[englishVocabSize];
			//init -inf because logAdd
			Arrays.fill(totalJointCountsByEnglishWord, Double.NEGATIVE_INFINITY);

			for(instance = 0; instance < trainingDataSize; instance++) {
				int[] englishSentenceIndexedWords = indexedEnglishTrainingData[instance];
				int[] frenchSentenceIndexedWords = indexedFrenchTrainingData[instance];

				double epsilon = 0.75;
				double nullPrior = Math.log(epsilon);
				double notNullPrior = Math.log((1-epsilon)/((double)englishSentenceIndexedWords.length));

				//compute eStep normalization
				eStepDenoms = new double[frenchVocabSize];
				for(int frenchWordPos = 0; frenchWordPos < frenchSentenceIndexedWords.length; frenchWordPos++) {
					int frenchWord = frenchSentenceIndexedWords[frenchWordPos];
					//init -inf because of logAdd
					double eStepDenom = Double.NEGATIVE_INFINITY;
					for(int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
						int englishWord = englishSentenceIndexedWords[englishWordPos];
						if(englishWord == 0) {
							//SloppyMath.logAdd() because we want to sum two log probabilities
							eStepDenom = SloppyMath.logAdd(eStepDenom, (theta[frenchWord][englishWord] + nullPrior));
						} else {
							//SloppyMath.logAdd() because we want to sum two log probabilities
							eStepDenom = SloppyMath.logAdd(eStepDenom, (theta[frenchWord][englishWord] + notNullPrior));
						}
					}
					eStepDenoms[frenchWord] = eStepDenom;
				}

				//eStep: collect expected counts
				for(int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
					int englishWord = englishSentenceIndexedWords[englishWordPos];
					for(int frenchWordPos = 0; frenchWordPos < frenchSentenceIndexedWords.length; frenchWordPos++) {
						int frenchWord = frenchSentenceIndexedWords[frenchWordPos];
						double update;
						if(englishWord == 0) {
							update = ((theta[frenchWord][englishWord] + nullPrior) - eStepDenoms[frenchWord]);
						} else {
							update = ((theta[frenchWord][englishWord] + notNullPrior) - eStepDenoms[frenchWord]);
						}
						jointCounts[frenchWord][englishWord] = SloppyMath.logAdd(jointCounts[frenchWord][englishWord], update);
						//SloppyMath.logAdd() because we want to sum two log probabilities
						totalJointCountsByEnglishWord[englishWord] = SloppyMath.logAdd(totalJointCountsByEnglishWord[englishWord], update);
					}
				}
			}
			//M step: reestimate alignment probabilities after seeing all training sentence pairs for an iteration
			for(int englishWord = 0; englishWord < englishVocabSize; englishWord++) {
				for(int frenchWord = 0; frenchWord < frenchVocabSize; frenchWord++) {
					if(jointCounts[frenchWord][englishWord] != Double.NEGATIVE_INFINITY) {
						//subtract because jointCounts and totalJointCountsByEnglishWord are both in log space: they come from theta
						double updatedTheta = (jointCounts[frenchWord][englishWord] - totalJointCountsByEnglishWord[englishWord]);
						theta[frenchWord][englishWord] = updatedTheta;
					} else {
						theta[frenchWord][englishWord] = Double.NEGATIVE_INFINITY;
					}
				}
			}
			iterations++;
//			if(iterations % 1 == 0) {
			System.out.println("We have completed " + (iterations) + " iterations of EM.");
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

		return new Model1WordAligner(theta, englishWordIndex, frenchWordIndex);
	}
}
