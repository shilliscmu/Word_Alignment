package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.Pair;

import java.util.HashMap;
import java.util.List;

public class HeuristicWordAligner implements WordAligner {
    private HashMap<String, Integer> englishCount;
    private HashMap<String, Integer> frenchCount;
    private HashMap<Pair<String, String>, Integer> jointCount;

    public HeuristicWordAligner(HashMap<String, Integer> englishCount, HashMap<String, Integer> frenchCount, HashMap<Pair<String, String>, Integer> jointCount) {
        this.englishCount = englishCount;
        this.frenchCount = frenchCount;
        this.jointCount = jointCount;
    }

    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignment = new Alignment();
        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();

        for(int i = 0; i < frenchWords.size(); i++) {
            double bestAlignmentScore = Double.NEGATIVE_INFINITY;
            int englishArgmaxIndex = -1;
            for(int j = 0; j < englishWords.size(); j++) {
                String englishWord = englishWords.get(j);
                String frenchWord = frenchWords.get(i);
                Pair<String, String> joint = new Pair(frenchWord, englishWord);

                int englishWordCount = 0;
                if(englishCount.containsKey(englishWord)) {
                    englishWordCount += englishCount.get(englishWord);
                }
                int frenchWordCount = 0;
                if(frenchCount.containsKey(frenchWord)) {
                    frenchWordCount += frenchCount.get(frenchWord);
                }
                int jointOccurenceCount = 0;
                if(jointCount.containsKey(joint)) {
                    jointOccurenceCount += jointCount.get(joint);
                }

                double alignmentScore = ((double)jointOccurenceCount / ((double)englishWordCount * (double)(frenchWordCount)));

                if(alignmentScore > bestAlignmentScore) {
                    bestAlignmentScore = alignmentScore;
                    englishArgmaxIndex = j;
                }
            }
            alignment.addAlignment(englishArgmaxIndex, i, true);
        }

        return alignment;
    }
}
