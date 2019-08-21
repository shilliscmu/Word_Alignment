package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.Arrays;
import java.util.List;

public class HmmWordAligner implements WordAligner {
    private double[][] theta;
    private double[] psi;
    private double epsilon;
    private StringIndexer englishWordIndex;
    private StringIndexer frenchWordIndex;

    public HmmWordAligner(double[][] theta, double[] psi, double epsilon, StringIndexer englishWordIndex, StringIndexer frenchWordIndex) {
        this.theta = theta;
        this.psi = psi;
        this.epsilon = epsilon;
        this.englishWordIndex = englishWordIndex;
        this.frenchWordIndex = frenchWordIndex;
    }

    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignment = new Alignment();

        List<String> englishWords = sentencePair.getEnglishWords();

        int[] englishSentenceIndexedWords = new int[englishWords.size()+1];
        englishSentenceIndexedWords[0]=englishWordIndex.indexOf("NULL");
        for(int pos = 1; pos <= englishWords.size(); pos++) {
            englishSentenceIndexedWords[pos] = englishWordIndex.addAndGetIndex(englishWords.get(pos-1));
        }

        List<String> frenchWords = sentencePair.getFrenchWords();
        int[] frenchSentenceIndexedWords = new int[frenchWords.size()];
        for(int pos = 0; pos < frenchWords.size(); pos++) {
            frenchSentenceIndexedWords[pos] = frenchWordIndex.addAndGetIndex(frenchWords.get(pos));
        }

//        System.out.println("English:");
//        for(int englishWord = 0; englishWord < englishSentenceIndexedWords.length; englishWord++) {
//            System.out.print(englishWordIndex.get(englishWord) + " ");
//        }
//        System.out.println();
//
//        System.out.println("French:");
//        for(int frenchWord = 0; frenchWord < frenchSentenceIndexedWords.length; frenchWord++) {
//            System.out.print(frenchWordIndex.get(frenchWord) + " ");
//        }
//        System.out.println();

        double initialNull = Math.log(epsilon);
        double initialNotNull = Math.log((1 - epsilon) / englishSentenceIndexedWords.length);
        double toNull = Math.log(epsilon);
        double fromNull = Math.log((1-epsilon) / englishSentenceIndexedWords.length);
        double[] fromNotNullToNotNull = new double[psi.length];
        for(int jump = 0; jump < psi.length; jump++) {
            fromNotNullToNotNull[jump] = Math.log(1-epsilon) + psi[jump];
        }

        double[][] viterbiScore = new double[frenchSentenceIndexedWords.length][englishSentenceIndexedWords.length];
        for(int frenchWordPos = 0; frenchWordPos < frenchSentenceIndexedWords.length; frenchWordPos++) {
            Arrays.fill(viterbiScore[frenchWordPos], Double.NEGATIVE_INFINITY);
        }
        int[][][] backPointers = new int[frenchSentenceIndexedWords.length][englishSentenceIndexedWords.length][1];

        //compute viterbiScoreIJ when J=0
        for(int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
            int englishWord = englishSentenceIndexedWords[englishWordPos];
            if(englishWord == 0) {
                viterbiScore[0][englishWordPos] = theta[0][englishWord] + initialNull;
            } else {
                viterbiScore[0][englishWordPos] = theta[0][englishWord] + initialNotNull;
            }
        }

        //compute viterbiScoreIJ when J>0
        for (int frenchWordPos = 1; frenchWordPos < frenchSentenceIndexedWords.length; frenchWordPos++) {
            int frenchWord = frenchSentenceIndexedWords[frenchWordPos];
            int frenchWordPosMinus1 = frenchWordPos-1;
            for(int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
                int englishWord = englishSentenceIndexedWords[englishWordPos];
                int argmaxEnglishWordPosPrime = -1;
                double maxEnglishWordPrimeViterbiScore = Double.NEGATIVE_INFINITY;
                //english word aligned to previous french word
                for(int englishWordPosPrime = 0; englishWordPosPrime < englishSentenceIndexedWords.length; englishWordPosPrime++) {
                    int englishWordPrime = englishSentenceIndexedWords[englishWordPosPrime];
                    double englishWordPrimeViterbiScore;
                    if (englishWord == 0) {
                        //to null
                        englishWordPrimeViterbiScore = (viterbiScore[frenchWordPosMinus1][englishWordPosPrime] + toNull);
                    } else if (englishWordPrime == 0) {
                        //from null
                        englishWordPrimeViterbiScore = (viterbiScore[frenchWordPosMinus1][englishWordPosPrime] + fromNull);
                    } else {
                        int jump = Math.abs(englishWordPos - englishWordPosPrime);
                        englishWordPrimeViterbiScore = (viterbiScore[frenchWordPosMinus1][englishWordPosPrime] + fromNotNullToNotNull[jump]);
                    }
                    if(englishWordPrimeViterbiScore > maxEnglishWordPrimeViterbiScore) {
                        maxEnglishWordPrimeViterbiScore = englishWordPrimeViterbiScore;
                        argmaxEnglishWordPosPrime = englishWordPosPrime;
                    }
                }
                viterbiScore[frenchWordPos][englishWordPos] = maxEnglishWordPrimeViterbiScore + theta[frenchWord][englishWord];
                int[] backPointer = new int[]{argmaxEnglishWordPosPrime};
                backPointers[frenchWordPos][englishWordPos] = backPointer;
            }
        }

        //find the highest score in the last column
        double maxLastColumnViterbiScore = Double.NEGATIVE_INFINITY;
        int argmaxEnglishWordPosForLastFrenchWord = -1;
        int frenchWordPos = frenchSentenceIndexedWords.length-1;
        for(int englishWordPos = 0; englishWordPos < englishSentenceIndexedWords.length; englishWordPos++) {
            double score = viterbiScore[frenchWordPos][englishWordPos];
            if(score > maxLastColumnViterbiScore) {
                maxLastColumnViterbiScore = score;
                argmaxEnglishWordPosForLastFrenchWord = englishWordPos;
            }
        }
        //follow the backpointers
        int frenchWord = frenchSentenceIndexedWords[frenchWordPos];
        int englishWordPos = argmaxEnglishWordPosForLastFrenchWord;
//        System.out.println("French word " + frenchWordIndex.get(frenchWord) + " aligns with " + englishWordIndex.get(englishSentenceIndexedWords[englishWordPos]));
        alignment.addAlignment(englishWordPos-1, frenchWordPos, true);
        int englishWordPosPrime = backPointers[frenchWordPos][englishWordPos][0];
        frenchWordPos--;
//        System.out.println("French word " + frenchWordIndex.get(frenchSentenceIndexedWords[frenchWordPos]) + " aligns with " + englishWordIndex.get(englishSentenceIndexedWords[englishWordPosPrime]));
        for(int previousFrenchWordPos = frenchWordPos; previousFrenchWordPos > 0; previousFrenchWordPos--) {
            alignment.addAlignment(englishWordPosPrime-1, previousFrenchWordPos, true);
            int[] backPointer = backPointers[previousFrenchWordPos][englishWordPosPrime];
            englishWordPosPrime = backPointer[0];
//            System.out.println("French word " + frenchWordIndex.get(frenchSentenceIndexedWords[previousFrenchWordPos]) + " aligns with " + englishWordIndex.get(englishSentenceIndexedWords[englishWordPosPrime]));
        }
        alignment.addAlignment(englishWordPosPrime-1, 0, true);

        return alignment;
    }
}
