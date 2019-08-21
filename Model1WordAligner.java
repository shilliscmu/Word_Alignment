package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.List;

public class Model1WordAligner implements WordAligner {
    private double[][] theta;
    private StringIndexer englishWordIndex;
    private StringIndexer frenchWordIndex;

    public Model1WordAligner(double[][] theta, StringIndexer englishWordIndex, StringIndexer frenchWordIndex) {
        this.theta = theta;
        this.englishWordIndex = englishWordIndex;
        this.frenchWordIndex = frenchWordIndex;
    }

    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignment = new Alignment();

        List<String> englishWords = sentencePair.getEnglishWords();

//        int[] sentenceOfIndexedEnglishTrainingData = new int[englishWords.size()];
//        for(int pos = 0; pos < englishWords.size(); pos++) {
//            sentenceOfIndexedEnglishTrainingData[pos] = englishWordIndex.indexOf(englishWords.get(pos));
//        }
        int[] sentenceOfIndexedEnglishTrainingData = new int[englishWords.size()+1];
        sentenceOfIndexedEnglishTrainingData[0]=englishWordIndex.indexOf("NULL");
        for(int pos = 1; pos <= englishWords.size(); pos++) {
            sentenceOfIndexedEnglishTrainingData[pos] = englishWordIndex.addAndGetIndex(englishWords.get(pos-1));
        }

        List<String> frenchWords = sentencePair.getFrenchWords();
        int[] sentenceOfIndexedFrenchTrainingData = new int[frenchWords.size()];
        for(int pos = 0; pos < frenchWords.size(); pos++) {
            sentenceOfIndexedFrenchTrainingData[pos] = frenchWordIndex.indexOf(frenchWords.get(pos));
        }

        for(int frenchWordPos = 0; frenchWordPos < sentenceOfIndexedFrenchTrainingData.length; frenchWordPos++) {
            int frenchWord = sentenceOfIndexedFrenchTrainingData[frenchWordPos];
            double bestAlignmentScore = Double.NEGATIVE_INFINITY;
            int englishArgmaxWordPos = -1;
            for(int englishWordPos = 0; englishWordPos < sentenceOfIndexedEnglishTrainingData.length; englishWordPos++) {
                int englishWord = sentenceOfIndexedEnglishTrainingData[englishWordPos];
                double alignmentScore = theta[frenchWord][englishWord];
                if(alignmentScore > bestAlignmentScore) {
                    bestAlignmentScore = alignmentScore;
                    englishArgmaxWordPos = englishWordPos;
                }
            }
            alignment.addAlignment(englishArgmaxWordPos-1,frenchWordPos, true);
        }

        return alignment;
    }
}
